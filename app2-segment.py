import os
from datetime import datetime
from rapidfuzz import fuzz
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from groq import Groq
import json
from flask_cors import CORS
from io import BytesIO
import time
import nltk
from sentence_transformers import SentenceTransformer, util
import threading
from functools import lru_cache

# =============================================================================
# GLOBAL STATE & HELPERS
# =============================================================================

# Global state for managing call audio buffers for reprocessing.
# This in-memory cache holds the complete audio for ongoing calls to enable
# re-transcription of fragmented segments at chunk boundaries.
# For production, a more robust distributed cache like Redis could be used.
call_audio_buffers = {}

def slice_audio(audio_bytes: bytes, start_time_s: float, end_time_s: float = None) -> BytesIO:
    """
    Slices a WAV audio file provided as bytes using torchaudio.
    
    Args:
        audio_bytes: The raw bytes of the WAV audio.
        start_time_s: The start time for the slice in seconds.
        end_time_s: The end time for the slice in seconds. If None, slices to the end.
        
    Returns:
        A BytesIO object containing the sliced WAV audio.
    """
    if not audio_bytes or len(audio_bytes) < 44:  # Smallest possible valid WAV
        print("[AUDIO_SLICE_WARNING] Audio bytes are empty or too small. Returning empty buffer.")
        return BytesIO()
        
    try:
        buffer = BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buffer)
        
        start_frame = int(start_time_s * sample_rate)
        end_frame = int(end_time_s * sample_rate) if end_time_s is not None else waveform.shape[1]
            
        # Ensure slice indices are within the valid range of the audio tensor
        start_frame = max(0, start_frame)
        end_frame = min(waveform.shape[1], end_frame)
        
        if start_frame >= end_frame:
            return BytesIO()
            
        sliced_waveform = waveform[:, start_frame:end_frame]
        
        sliced_buffer = BytesIO()
        torchaudio.save(sliced_buffer, sliced_waveform, sample_rate, format="wav")
        sliced_buffer.seek(0)
        
        return sliced_buffer
        
    except Exception as e:
        print(f"[AUDIO_SLICE_ERROR] Failed to slice audio: {e}. Returning original audio as fallback.")
        fallback_buffer = BytesIO(audio_bytes)
        fallback_buffer.seek(0)
        return fallback_buffer

def append_wav_chunks(existing_wav_bytes: bytes, new_wav_chunk_bytes: bytes) -> bytes:
    """
    Combines two WAV byte streams by decoding them, concatenating their
    audio data, and encoding back into a single WAV byte stream.
    This prevents issues with concatenated WAV headers.
    """
    if not new_wav_chunk_bytes:
        return existing_wav_bytes
    if not existing_wav_bytes:
        return new_wav_chunk_bytes

    try:
        existing_buffer = BytesIO(existing_wav_bytes)
        new_chunk_buffer = BytesIO(new_wav_chunk_bytes)

        existing_waveform, sr1 = torchaudio.load(existing_buffer)
        new_waveform, sr2 = torchaudio.load(new_chunk_buffer)

        if sr1 != sr2:
            print(f"[AUDIO_APPEND_WARNING] Sample rate mismatch: {sr1} vs {sr2}. Resampling to {sr1}.")
            resampler = torchaudio.transforms.Resample(sr2, sr1)
            new_waveform = resampler(new_waveform)

        combined_waveform = torch.cat((existing_waveform, new_waveform), dim=1)
        
        output_buffer = BytesIO()
        torchaudio.save(output_buffer, combined_waveform, sr1, format="wav")
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    except Exception as e:
        print(f"[AUDIO_APPEND_ERROR] Failed to combine WAV chunks: {e}. Returning original audio only.")
        return existing_wav_bytes

def resegment_based_on_punctuation(words):
    """
    Resegments text based on punctuation ('.', '?', '!') using word-level timestamps.
    """
    if not words:
        return []

    new_segments = []
    current_words = []
    sentence_ending_punctuations = ".?!"

    for word_info in words:
        current_words.append(word_info)
        word_text = word_info['word'].strip()
        
        if word_text and word_text[-1] in sentence_ending_punctuations:
            if current_words:
                sentence_text = " ".join([w['word'] for w in current_words]).strip()
                start_time = current_words[0]['start']
                end_time = current_words[-1]['end']
                
                new_segments.append({
                    'text': sentence_text,
                    'start': start_time,
                    'end': end_time,
                })
                current_words = []

    if current_words:
        sentence_text = " ".join([w['word'] for w in current_words]).strip()
        start_time = current_words[0]['start']
        end_time = current_words[-1]['end']
        new_segments.append({
            'text': sentence_text,
            'start': start_time,
            'end': end_time,
        })

    print(f"[RESEGMENT] Created {len(new_segments)} new segments from {len(words)} words.")
    return new_segments

# =============================================================================
# VERISYNC ANALYSIS BACKEND - SEGMENT-BASED VERSION
# =============================================================================
# 
# Segment-based version using Whisper segments as sentences:
# - POST /update_call: Process audio chunks with segment-based analysis
# - POST /final_update: Complete call analysis with final metrics
#
# Key Features:
# - Segment-Based Processing: Uses Whisper segments directly as sentences
# - Adaptive Windows: Tests window sizes 1, 2, and 3 automatically
# - SBERT Optimization: Cached embeddings + GPU support
# - Complete Analysis: Audio processing, transcription, adherence, emotions
# - Custom Script Support: XML-like format with automatic parsing
# - Script Fallback: Loads from audioscript.txt if no transcript provided
#
# Script Format:
# <strict>The line of the script</strict>
# <semantic>The line of the script</semantic>
# <topic>The line of the script</topic>
# =============================================================================

# MongoDB imports
from db_config import get_db
from call_operations import get_call_operations

# Silero VAD imports
import torch
import torchaudio
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global cache for prompt embeddings to improve performance
_prompt_embedding_cache = {}
_cache_lock = threading.Lock()

# Initialize Silero VAD model
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    VAD_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] Failed to load Silero VAD model: {e}")
    VAD_AVAILABLE = False
    model = None

# Initialize Sentence Transformer model with optimizations
try:
    print("[INFO] Attempting to load Sentence Transformer model...")
    
    model_options = [
        'all-MiniLM-L6-v2',
        'paraphrase-MiniLM-L3-v2',
        'all-MiniLM-L12-v2'
    ]
    
    sbert_model = None
    for model_name in model_options:
        try:
            print(f"[INFO] Trying to load model: {model_name}")
            sbert_model = SentenceTransformer(model_name)
            
            if torch.cuda.is_available():
                sbert_model = sbert_model.to('cuda')
                print(f"[INFO] SBERT model moved to GPU")
            else:
                print(f"[INFO] SBERT model running on CPU")
            
            print(f"[INFO] Sentence Transformer model '{model_name}' loaded successfully.")
            break
        except Exception as model_error:
            print(f"[WARNING] Failed to load {model_name}: {model_error}")
            continue
    
    if sbert_model is not None:
        SBERT_AVAILABLE = True
    else:
        print("[ERROR] All sentence transformer models failed to load")
        SBERT_AVAILABLE = False
        
except Exception as e:
    print(f"[ERROR] Failed to initialize Sentence Transformer: {e}")
    SBERT_AVAILABLE = False
    sbert_model = None

def _precompute_prompt_embeddings():
    """Pre-compute embeddings for default script prompts to improve performance"""
    try:
        print("[CACHE] Pre-computing default prompt embeddings...")
        script_checkpoints = get_current_call_script()  # This will use default or file-based script
        
        with _cache_lock:
            for checkpoint in script_checkpoints:
                prompt_sentences = checkpoint.get("prompt_sentences", [])
                checkpoint_id = checkpoint.get("checkpoint_id", "unknown")
                for sent_idx, sentence in enumerate(prompt_sentences):
                    if sentence:
                        clean_prompt_preserve = clean_text_for_matching(sentence, preserve_structure=True)
                        clean_prompt_remove = clean_text_for_matching(sentence, preserve_structure=False)
                        
                        for prompt_version, version_key in [(clean_prompt_preserve, "preserve"), (clean_prompt_remove, "remove")]:
                            if prompt_version.strip():
                                sentence_cache_id = f"{checkpoint_id}_{sent_idx}"
                                cache_key = f"{sentence_cache_id}_{version_key}"
                                if cache_key not in _prompt_embedding_cache:
                                    embedding = sbert_model.encode(prompt_version, convert_to_tensor=True)
                                    _prompt_embedding_cache[cache_key] = embedding
                                
        print(f"[CACHE] Cached {len(_prompt_embedding_cache)} default prompt embeddings")
        
    except Exception as e:
        print(f"[CACHE] Failed to pre-compute prompt embeddings: {e}")

def get_cached_prompt_embedding(checkpoint_id, prompt_text, preserve_structure=True):
    """Get cached prompt embedding or compute and cache if not available"""
    try:
        clean_prompt = clean_text_for_matching(prompt_text, preserve_structure=preserve_structure)
        version_key = "preserve" if preserve_structure else "remove"
        cache_key = f"{checkpoint_id}_{version_key}"
        
        with _cache_lock:
            if cache_key in _prompt_embedding_cache:
                return _prompt_embedding_cache[cache_key]
            
            if clean_prompt.strip() and SBERT_AVAILABLE and sbert_model is not None:
                embedding = sbert_model.encode(clean_prompt, convert_to_tensor=True)
                _prompt_embedding_cache[cache_key] = embedding
                return embedding
        
        return None
        
    except Exception as e:
        print(f"[CACHE] Error getting cached embedding for {checkpoint_id}: {e}")
        return None

def encode_sentences_with_memory_optimization(sentences, batch_size=32):
    """Encode sentences in batches to manage memory usage during concurrent requests"""
    try:
        if not SBERT_AVAILABLE or sbert_model is None:
            return None
            
        if len(sentences) <= batch_size:
            return sbert_model.encode(sentences, convert_to_tensor=True)
        
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = sbert_model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
        
    except Exception as e:
        print(f"[MEMORY] Error in batch encoding: {e}")
        return None
        
# Download NLTK data for sentence tokenization
try:
    nltk.download('punkt')
    print("[INFO] NLTK 'punkt' model downloaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to download NLTK 'punkt' model: {e}")

def get_semantic_match(current_text, script_sentences, model):
    """
    Compare current text against script sentences using semantic similarity
    """
    if not SBERT_AVAILABLE or model is None:
        print(f"[SEMANTIC] SBERT not available, returning 0.0 for: '{current_text}'")
        return 0.0
    
    try:
        print(f"[SEMANTIC] Comparing: '{current_text}' against {len(script_sentences)} script sentences")
        
        # Encode current text
        current_embedding = model.encode(current_text.lower())
        
        # Encode all script sentences
        script_embeddings = model.encode([s.lower() for s in script_sentences])
        
        # Calculate similarities
        similarities = []
        for idx, script_emb in enumerate(script_embeddings):
            similarity = util.cos_sim(current_embedding, script_emb).item()
            similarities.append(similarity)
            script_text = script_sentences[idx]
            print(f"[SEMANTIC] - Script {idx+1}: '{script_text[:40]}{'...' if len(script_text) > 40 else ''}' → {similarity:.3f} ({similarity*100:.1f}%)")
        
        max_similarity = max(similarities)
        best_match_idx = similarities.index(max_similarity)
        best_script = script_sentences[best_match_idx]
        
        print(f"[SEMANTIC] Best match: Script {best_match_idx+1} '{best_script[:50]}{'...' if len(best_script) > 50 else ''}' with {max_similarity:.3f} ({max_similarity*100:.1f}%)")
        
        return max_similarity
    except Exception as e:
        print(f"[ERROR] Semantic matching failed: {e}")
        return 0.0

def process_whisper_segments(segments):
    """
    Process Whisper segments into sentence-like structures for analysis
    """
    if not segments:
        print("[SEGMENTS] No segments provided, returning empty list")
        return []
    
    print(f"[SEGMENTS] ========== Processing Whisper Segments ==========")
    print(f"[SEGMENTS] Processing {len(segments)} segments from Whisper")
    
    processed_segments = []
    
    for i, segment in enumerate(segments):
        # Extract segment data
        if hasattr(segment, 'text') and hasattr(segment, 'start') and hasattr(segment, 'end'):
            # Object format
            text = segment.text.strip()
            start_time = segment.start
            end_time = segment.end
        elif isinstance(segment, dict):
            # Dictionary format
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
        else:
            print(f"[SEGMENTS] Warning: Unknown segment format at index {i}")
            continue
        
        if not text:
            print(f"[SEGMENTS] Skipping empty segment {i+1}")
            continue
        
        word_count = len(text.split())
        duration = end_time - start_time
        
        print(f"[SEGMENTS] Segment {i+1}: '{text}' ({word_count} words, {duration:.2f}s)")
        
        processed_segment = {
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'word_count': word_count,
            'duration': duration,
                'metadata': {
                'segment_index': i,
                'source': 'whisper_segment',
                'confidence': segment.get('confidence', 1.0) if isinstance(segment, dict) else 1.0
            }
        }
        
        processed_segments.append(processed_segment)
    
    print(f"[SEGMENTS] ========== Segment Processing Complete ==========")
    print(f"[SEGMENTS] Processed {len(processed_segments)} valid segments")
    
    return processed_segments

# =============================================================================
# CALL SCRIPT CONFIGURATION
# =============================================================================

DEFAULT_CALL_SCRIPT = """
<semantic> Hello, thank you for calling Customer Service </semantic>
<semantic> My name is [Agent Name], how can I assist you today? </semantic>
<semantic> I understand how frustrating that must be. Let me check available options for you </semantic>
<semantic> Can you provide your flight details? </semantic>
<semantic> I see your flight was canceled due to [Reason] </semantic>
<semantic> The next available flights are at [Time] </semantic>
<semantic> Would you like to proceed with one of these? </semantic>
<semantic> Since this was due to [reason], our policy does not include compensation. However, I can offer [something]. </semantic>
<semantic> I've rebooked your flight. </semantic>
<semantic> A confirmation email has been sent </semantic>
<strict> Is there anything else I can assist with? </strict>
<semantic> Thank you for choosing. Safe travels! </semantic>
"""

def parse_script_from_text(script_text):
    """
    Parse script from custom XML-like format:
    <strict>The line of the script</strict>
    <semantic>The line of the script</semantic>
    <topic>The line of the script</topic>
    
    Returns a list of checkpoint dictionaries
    """
    if not script_text or not script_text.strip():
        return []
    
    import re
    
    # Pattern to match tags: <type>content</type> (case insensitive)
    pattern = r'<(strict|semantic|topic)>(.*?)</\1>'
    matches = re.findall(pattern, script_text, re.IGNORECASE | re.DOTALL)
    
    if not matches:
        return []
    
    checkpoints = []
    
    for i, (adherence_type, prompt_text) in enumerate(matches):
        adherence_type_upper = adherence_type.upper()
        prompt_text_clean = prompt_text.strip()
        
        if not prompt_text_clean:
            continue

        try:
            sentences = nltk.sent_tokenize(prompt_text_clean)
        except Exception as e:
            print(f"[SCRIPT PARSER] NLTK sentence tokenization failed: {e}. Treating as single sentence.")
            sentences = [prompt_text_clean]
        
        # Assign weights based on adherence type and position
        # Higher weights for stricter types and important positions
        base_weights = {"STRICT": 25, "SEMANTIC": 20, "TOPIC": 15}
        
        weight = base_weights.get(adherence_type_upper, 15)
        
        # Determine if mandatory - STRICT types are always mandatory
        # First and last checkpoints are also typically mandatory regardless of type
        is_mandatory = (
            adherence_type_upper == "STRICT" or 
            i == 0 or  # First checkpoint (greeting)
            i == len(matches) - 1  # Last checkpoint (closing)
        )
            
        checkpoint = {
            "checkpoint_id": f"checkpoint_{i+1}_{adherence_type.lower()}",
            "prompt_text": prompt_text_clean,
            "prompt_sentences": sentences,
            "adherence_type": adherence_type_upper,
            "is_mandatory": is_mandatory,
            "weight": weight
        }
        
        checkpoints.append(checkpoint)
    
    return checkpoints

def load_script_from_file(file_path="audioscript.txt"):
    """Load script from file in the analysis folder"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), file_path)
        
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as file:
                script_content = file.read()
                return script_content
        else:
            return None
            
    except Exception as e:
        print(f"[SCRIPT LOADER] Error loading script file: {e}")
        return None

def get_current_call_script(script_text=None):
    """
    Get the current call script for adherence checking.
    
    Args:
        script_text: Custom script text in XML format, or None to use default
    
    Returns:
        List of checkpoint dictionaries (always parsed from XML format)
    """
    if script_text:
        checkpoints = parse_script_from_text(script_text)
        if checkpoints:
            return checkpoints
    
    # Try to load from file as fallback
    file_script = load_script_from_file()
    if file_script:
        checkpoints = parse_script_from_text(file_script)
        if checkpoints:
            return checkpoints
    
    # Use default built-in script
    checkpoints = parse_script_from_text(DEFAULT_CALL_SCRIPT)
    
    # Final safety check - if even default script fails, create a minimal fallback
    if not checkpoints:
        print("[SCRIPT] ERROR: All scripts failed to parse, creating minimal fallback")
        return [{
            "checkpoint_id": "fallback_greeting",
            "prompt_text": "Hello, how can I help you?",
            "adherence_type": "SEMANTIC",
            "is_mandatory": True,
            "weight": 100
        }]
    
    return checkpoints

def clean_text_for_matching(text, preserve_structure=False):
    """Clean text for better semantic matching"""
    if preserve_structure:
        cleaned = text.replace('[Company Name]', 'company')
        cleaned = cleaned.replace('[Agent Name]', 'agent')
        cleaned = cleaned.replace('[Customer Name]', 'customer')
        cleaned = cleaned.replace('[', '').replace(']', '')
    else:
        cleaned = text.replace('[Company Name]', '').replace('[Agent Name]', '')
        cleaned = cleaned.replace('[Customer Name]', '').replace('[', '').replace(']', '')
    
    cleaned = ' '.join(cleaned.split()).lower().strip()
    return cleaned

def check_script_adherence_adaptive_windows(agent_text, agent_segments, script_checkpoints=None, script_text=None):
    """
    ADAPTIVE WINDOW APPROACH - CONCATENATED
    Tests window sizes 1, 2, and 3, concatenating segment text within the window
    for a more robust comparison. Automatically selects the best performing window size.
    
    Args:
        agent_text: The agent's spoken text
        agent_segments: Pre-processed segments from Whisper (not sentences)
        script_checkpoints: Pre-parsed checkpoint list (optional)
        script_text: Raw script text in XML format (optional)
    """
    if script_checkpoints is None:
        script_checkpoints = get_current_call_script(script_text)

    if not agent_segments:
        return {"real_time_adherence_score": 0, "script_completion_percentage": 0, "checkpoint_results": []}

    # Extract segment texts for processing
    segment_texts = []
    for segment in agent_segments:
        text = segment.get('text', '') if isinstance(segment, dict) else str(segment)
        if text.strip():
            segment_texts.append(text.strip())

    checkpoint_results = []
    
    for i, checkpoint in enumerate(script_checkpoints):
        checkpoint_id = checkpoint.get("checkpoint_id", f"unknown_{i}")
        prompt_sentences = checkpoint.get("prompt_sentences", [])
        if not prompt_sentences:
            prompt_text = checkpoint.get("prompt_text", "")
            if prompt_text:
                prompt_sentences = [prompt_text]
            else:
                continue

        adherence_type = checkpoint.get("adherence_type", "SEMANTIC")
        weight = checkpoint.get("weight", 1)
        is_mandatory = checkpoint.get("is_mandatory", True)
        threshold_map = {'STRICT': 85, 'SEMANTIC': 60, 'TOPIC': 40}
        threshold = threshold_map.get(adherence_type, 60)

        print(f"\n[ADHERENCE] Analyzing Checkpoint {i+1}: {checkpoint_id} ({adherence_type})")

        sentence_level_results = []
        all_sentence_scores = []

        for sent_idx, script_sentence in enumerate(prompt_sentences):
            clean_prompt_preserve = clean_text_for_matching(script_sentence, preserve_structure=True)
            clean_prompt_remove = clean_text_for_matching(script_sentence, preserve_structure=False)
            
            prompt_embedding_preserve, prompt_embedding_remove = None, None
            if SBERT_AVAILABLE and sbert_model:
                try:
                    sentence_cache_id = f"{checkpoint_id}_{sent_idx}"
                    prompt_embedding_preserve = get_cached_prompt_embedding(sentence_cache_id, script_sentence, True)
                    prompt_embedding_remove = get_cached_prompt_embedding(sentence_cache_id, script_sentence, False)
                except Exception as e:
                    print(f"[CONCAT] [ERROR] Failed to get cached prompt embeddings: {e}")

            best_score_for_sentence = 0
            best_match_info = None

            # Test each window size (1, 2, 3)
            for window_size in range(1, 4):
                # Create all possible windows of the current size
                for start_idx in range(len(segment_texts) - window_size + 1):
                    end_idx = start_idx + window_size
                    window_segments = segment_texts[start_idx:end_idx]
                    window_text = " ".join(window_segments)
                    
                    score_for_window = 0
                    method_used = ""

                    if adherence_type == "STRICT":
                        # Check for exact match first
                        if clean_prompt_preserve in window_text or clean_prompt_remove in window_text:
                            score_for_window = 100
                            method_used = "exact"
                        else: # Fallback to fuzzy matching
                            score_preserve = fuzz.ratio(clean_prompt_preserve, window_text)
                            score_remove = fuzz.ratio(clean_prompt_remove, window_text)
                            score_for_window = max(score_preserve, score_remove)
                            method_used = "fuzzy"
                    
                    elif adherence_type == "SEMANTIC" and SBERT_AVAILABLE and sbert_model:
                        try:
                            # Generate embedding for the concatenated window text on the fly
                            window_embedding = sbert_model.encode(window_text, convert_to_tensor=True)
                            
                            sim_preserve, sim_remove = 0, 0
                            if prompt_embedding_preserve is not None:
                                sim_preserve = util.cos_sim(prompt_embedding_preserve, window_embedding).item()
                            if prompt_embedding_remove is not None:
                                sim_remove = util.cos_sim(prompt_embedding_remove, window_embedding).item()
                            
                            score_for_window = round(max(sim_preserve, sim_remove) * 100, 2)
                            method_used = "sbert"
                        except Exception as e:
                            print(f"[CONCAT] [ERROR] Failed to compute semantic similarity for window: {e}")
                            score_for_window = 0 # Assign 0 if embedding fails

                    elif adherence_type == "TOPIC":
                        score_preserve = fuzz.ratio(clean_prompt_preserve, window_text)
                        score_remove = fuzz.ratio(clean_prompt_remove, window_text)
                        score_for_window = max(score_preserve, score_remove)
                        method_used = "fuzzy"

                    # Track the best score found so far for this sentence
                    if score_for_window > best_score_for_sentence:
                        best_score_for_sentence = score_for_window
                        best_match_info = {
                            "window_size": window_size,
                            "window_start_idx": start_idx,
                            "window_end_idx": end_idx - 1,
                            "method": method_used,
                            "matched_text": window_text
                        }
                    
                    # Optimization: if a perfect score is found, no need to check other windows
                    if best_score_for_sentence >= 100:
                        break
                if best_score_for_sentence >= 100:
                    break

            sentence_status = "PASS" if best_score_for_sentence >= threshold else "FAIL"
            
            # NEW LOGIC: If sentence passes threshold, use 100 for averaging; otherwise use actual score
            adjusted_score_for_averaging = 100 if sentence_status == "PASS" else best_score_for_sentence
            
            agent_match_text = best_match_info['matched_text'] if best_match_info else "No match found"

            print(f"  - Script: '{script_sentence}'")
            print(f"    Agent Match (Window): '{agent_match_text}' -> Score: {best_score_for_sentence:.2f}% ({sentence_status})")
            print(f"    Adjusted Score for Averaging: {adjusted_score_for_averaging:.2f}%")

            sentence_level_results.append({
                "script_sentence": script_sentence,
                "agent_match_text": agent_match_text,
                "score": best_score_for_sentence,
                "adjusted_score": adjusted_score_for_averaging,
                "status": sentence_status,
                "match_details": best_match_info
            })
            all_sentence_scores.append(adjusted_score_for_averaging)

        final_checkpoint_score = (sum(all_sentence_scores) / len(all_sentence_scores)) if all_sentence_scores else 0
        # NEW LOGIC: Checkpoint passes if average adjusted score > 60
        final_checkpoint_status = "PASS" if final_checkpoint_score > 60 else "FAIL"
        
        print(f"[ADHERENCE] Checkpoint Result: {final_checkpoint_status} (Avg Score: {final_checkpoint_score:.2f}%)")

        checkpoint_results.append({
            "checkpoint_id": checkpoint_id,
            "status": final_checkpoint_status,
            "score": final_checkpoint_score,
            "weight": weight,
            "is_mandatory": is_mandatory,
            "sentence_results": sentence_level_results
        })

    # (The final scoring logic remains the same)
    last_passed_index = -1
    passed_checkpoints = sum(1 for r in checkpoint_results if r["status"] == "PASS")
    
    for i, result in enumerate(checkpoint_results):
        if result["status"] == "PASS":
            last_passed_index = i
            
    script_completion_percentage = round((passed_checkpoints / len(checkpoint_results)) * 100, 2) if checkpoint_results else 0
    
    if last_passed_index >= 0:
        relevant_checkpoints = checkpoint_results[:last_passed_index + 1]
        total_weight = sum(r["weight"] for r in relevant_checkpoints)
        weighted_score_sum = sum(r["score"] * r["weight"] for r in relevant_checkpoints)
        real_time_adherence_score = round(weighted_score_sum / total_weight, 2) if total_weight > 0 else 0
    else:
        real_time_adherence_score = 0
    
    return {
        "real_time_adherence_score": real_time_adherence_score,
        "script_completion_percentage": script_completion_percentage,
        "checkpoint_results": checkpoint_results,
        "method": "adaptive_windows_concatenated",
        "total_checkpoints": len(checkpoint_results)
    }

def get_quality(emotions):
    """Calculate call quality based on emotions"""
    positive = ["joy", "neutral", "gratitude", "admiration", "optimism", "relief", "caring"]
    negative = ["anger", "annoyance", "disappointment", "disapproval", "sadness", "confusion", "embarrassment"]
    
    positive_score = sum(emotions[e] for e in positive if e in emotions)
    negative_score = sum(emotions[e] for e in negative if e in emotions)
    total_score = positive_score + negative_score
    
    if total_score == 0:
        call_quality = 100
    else:
        call_quality = (positive_score / total_score) * 100
    result = round(call_quality, 2)
    return result

def detect_voice_activity(audio_file, min_speech_duration=0.5, min_silence_duration=0.1):
    """Detect voice activity in audio using Silero VAD"""
    if not VAD_AVAILABLE:
        return {
            'has_speech': True,
            'speech_segments': [],
            'total_speech_duration': 0,
            'total_audio_duration': 0,
            'speech_ratio': 1.0,
            'vad_available': False
        }
    
    try:
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)
            
        if isinstance(audio_file, BytesIO):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio_file.seek(0)
                temp_file.write(audio_file.read())
                temp_path = temp_file.name
                audio_file.seek(0)
            
            wav, sr = torchaudio.load(temp_path)
            
            import os
            os.unlink(temp_path)
        else:
            wav, sr = torchaudio.load(audio_file)
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000
        
        wav = wav.squeeze()
        total_duration = len(wav) / sr
        
        speech_timestamps = get_speech_timestamps(
            wav, 
            model,
            sampling_rate=sr,
            min_speech_duration_ms=int(min_speech_duration * 1000),
            min_silence_duration_ms=int(min_silence_duration * 1000),
            window_size_samples=512,
            speech_pad_ms=30
        )
        
        total_speech_duration = 0
        speech_segments = []
        
        for segment in speech_timestamps:
            start_time = segment['start'] / sr
            end_time = segment['end'] / sr
            duration = end_time - start_time
            total_speech_duration += duration
            speech_segments.append([start_time, end_time])
        
        speech_ratio = total_speech_duration / total_duration if total_duration > 0 else 0
        has_speech = len(speech_timestamps) > 0 and total_speech_duration >= min_speech_duration
        
        return {
            'has_speech': has_speech,
            'speech_segments': speech_segments,
            'total_speech_duration': total_speech_duration,
            'total_audio_duration': total_duration,
            'speech_ratio': speech_ratio,
            'vad_available': True
        }
        
    except Exception as e:
        return {
            'has_speech': True,
            'speech_segments': [],
            'total_speech_duration': 0,
            'total_audio_duration': 0,
            'speech_ratio': 1.0,
            'vad_available': True,
            'error': str(e)
        }

def speech_to_text(audio):
    """Convert audio to text using Groq Whisper API with segment-based processing"""
    try:
        audio_copy = BytesIO()
        audio.seek(0)
        audio_copy.write(audio.read())
        audio_copy.seek(0)
        
        buffer_data = audio_copy.read()
        audio_copy.seek(0)
        
        if len(buffer_data) < 44:
            return "", 0, [], []
        
        # Voice Activity Detection
        vad_result = detect_voice_activity(audio_copy, min_speech_duration=0.3, min_silence_duration=0.1)
        
        if not vad_result['has_speech']:
            return "", vad_result['total_audio_duration'], [], []
        
        # Groq Whisper API transcription with segment granularity
        audio_copy.name = "audio.wav"
        
        print(f"[API REQUEST] Calling Groq Whisper API for segment-based transcription.")
        response = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_copy,
            language="en",
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"]
        )
        
        print("[API RESPONSE] Groq Whisper segment transcription completed")
        
        if hasattr(response, 'text'):
            transcription = response.text or ""
        else:
            transcription = response.get('text', '') if isinstance(response, dict) else ""
        
        duration = vad_result['total_audio_duration']
        
        # Process segment-level data
        segment_data = []
        segments = None
        if hasattr(response, 'segments') and response.segments:
            segments = response.segments
        elif isinstance(response, dict) and 'segments' in response and response['segments']:
            segments = response['segments']
        
        if segments:
            for segment in segments:
                if hasattr(segment, 'text') and hasattr(segment, 'start') and hasattr(segment, 'end'):
                    segment_data.append({
                        'text': segment.text.strip(),
                        'start': segment.start,
                        'end': segment.end,
                        'confidence': getattr(segment, 'avg_logprob', 0.0)
                    })
                elif isinstance(segment, dict):
                    segment_data.append({
                        'text': segment.get('text', '').strip(),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'confidence': segment.get('avg_logprob', 0.0)
                    })
        
        # Process word-level data
        word_data = []
        words = None
        if hasattr(response, 'words') and response.words:
            words = response.words
        elif isinstance(response, dict) and 'words' in response and response['words']:
            words = response['words']

        if words:
            for word in words:
                word_data.append({
                    'word': word.get('word', '').strip(),
                    'start': word.get('start', 0),
                    'end': word.get('end', 0)
                })

        print(f"[API RESPONSE] Extracted {len(segment_data)} segments and {len(word_data)} words from Whisper")
        
        return transcription, duration, segment_data, word_data
        
    except Exception as e:
        error_message = str(e)
        
        if "rate_limit" in error_message.lower() or "429" in error_message:
            print(f"[API ERROR] Groq API Rate Limit: {error_message}")
            time.sleep(30)
            try:
                audio_copy.seek(0)
                audio_copy.name = "audio.wav"
                print("[API RETRY] Retrying Groq Whisper API call")
                response = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_copy,
                    language="en",
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"]
                )
                
                if hasattr(response, 'text'):
                    transcription = response.text or ""
                else:
                    transcription = response.get('text', '') if isinstance(response, dict) else ""
                
                print("[API RETRY] Groq API retry successful")
                return transcription, 0, [], []
                
            except Exception as retry_error:
                print(f"[API ERROR] Groq API retry failed: {retry_error}")
                return "", 0, [], []
        else:
            print(f"[API ERROR] Groq API error: {error_message}")
            return "", 0, [], []

def get_response(text):
    """Get emotion analysis from external API"""
    print(f"[API REQUEST] Calling emotion model API")
    
    url = os.getenv('EMOTION_MODEL')
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {"input": text}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"[API RESPONSE] Emotion API call successful")
            return result
        else:
            print(f"[API ERROR] Emotion API returned status {response.status_code}")
            return {"predictions": [{"label": "neutral", "score": 1.0}]}
            
    except Exception as e:
        print(f"[API ERROR] Exception calling emotion API: {e}")
        return {"predictions": [{"label": "neutral", "score": 1.0}]}

def calculate_cqs(predictions):
    """Calculate Customer Quality Score based on emotions"""
    emotion_weights = {
        "joy": 2.5, "gratitude": 2.0, "admiration": 1.75, "optimism": 1.5, "relief": 1.25, "caring": 1.0,
        "neutral": 0, "curiosity": 0, "realization": 0,
        "anger": -2.5, "disapproval": -2.25, "disappointment": -2.0, "annoyance": -1.75, "sadness": -1.5, "embarrassment": -1.0, "confusion": -0.5
    }
    cqs = 0
    emotions = {}
    for emotion in predictions:
        label = emotion["label"].lower()
        score = emotion["score"]
        if label in emotion_weights:
            emotions[label] = round(score, 2)
            weighted_contribution = emotion_weights[label] * score
            cqs += weighted_contribution
    final_cqs = round(cqs, 2)
    result = [final_cqs, emotions]
    return result

def agent_scores(transcription):
    """Evaluate agent performance using Groq AI"""
    prompt = f'''
You are an AI assistant trained to evaluate call center agent performance based on customer experience criteria. Given a transcript of a conversation between an agent and a customer, analyze the interaction and answer the following questions. Provide the results in JSON format with scores for "Yes," "No," and "N/A." by giving percentage of how much you are confident that agent followed what he or she is asked to do in question.

Each question must be evaluated based on the transcript, and responses should be categorized as:

"Yes" (if the agent meets the criterion),
"No" (if the agent does not meet the criterion),
"N/A" (if the criterion is not applicable in the given conversation).

Questions to Evaluate:
1. Did the agent use the proper greeting by identifying the company and themselves by name?
2. Did the agent show empathy with a commitment to help?
3. Did the agent ask the customer to verify their name, address, and phone number?
4. Did the agent show confidence that the customer's issue would be resolved?
5. Was the agent courteous and professional?
6. Was the agent proactive in identifying the customer's needs?
7. Did the agent actively listen to the customer?
8. Did the agent address the customer by name throughout the conversation?
9. Did the agent offer additional assistance?

JSON Output Format:

{{
    "q1": {{"yes":...,"no":...,"na":...}},
    "q2": {{"yes":...,"no":...,"na":...}},
    "q3": {{"yes":...,"no":...,"na":...}},
    "q4": {{"yes":...,"no":...,"na":...}},
    ...
}}

Transcription:
{transcription}
'''
    
    print("[API REQUEST] Calling Groq AI for agent performance evaluation")
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            response_format={"type": "json_object"}
        )

        response_content = completion.choices[0].message.content
        result = json.loads(response_content)
        print("[API RESPONSE] Groq AI agent evaluation completed")
        
        return result
            
    except Exception as e:
        print(f"[API ERROR] Exception calling Groq AI: {e}")
        result = {}
        for i in range(1, 10):
            result[f'q{i}'] = {"yes": 0, "no": 0, "na": 100}
        return result

def call_summary(agent_text, client_text):
    """Generate call summary using Groq AI"""
    prompt = f'''
Here is conversation between agent and customer in call center:
Agent:
{agent_text}

Customer:
{client_text}

I want you to generate summary of this call. Make sure it is in paragraph format. Output must be in json format:
{{"summary":...}}
'''
    
    print("[API REQUEST] Calling Groq AI for call summary generation")
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            response_format={"type": "json_object"}
        )

        response_content = completion.choices[0].message.content
        
        try:
            parsed_response = eval(response_content)
            summary = parsed_response['summary']
            print("[API RESPONSE] Groq AI call summary completed")
            return summary
            
        except Exception as e:
            print(f"[API ERROR] Failed to parse summary response: {e}")
            return "Summary generation failed - unable to parse AI response"
            
    except Exception as e:
        print(f"[API ERROR] Exception calling Groq AI for summary: {e}")
        return "Summary generation failed - AI service unavailable"

def get_tags(conversation):
    """Generate conversation tags using Groq AI"""
    prompt = f'''
here is conversation between client and call center agent:

{conversation}

I want you to generate maximum of 3 tags which summarizes whole conversation. Output must be in json format in following way:
{{
 "Tags": [tag_1, tag2, tag_3]
}}
'''
    
    print("[API REQUEST] Calling Groq AI for conversation tags")
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            response_format={"type": "json_object"}
        )

        response_content = completion.choices[0].message.content
        
        try:
            result = json.loads(response_content)
            print("[API RESPONSE] Groq AI conversation tags completed")
            return result
            
        except json.JSONDecodeError as e:
            print(f"[API ERROR] Failed to parse tags response: {e}")
            return {"Tags": []}
            
    except Exception as e:
        print(f"[API ERROR] Exception calling Groq AI for tags: {e}")
        return {"Tags": []}

# Initialize prompt embeddings cache after functions are defined
if SBERT_AVAILABLE and sbert_model is not None:
    try:
        _precompute_prompt_embeddings()
    except Exception as e:
        print(f"[ERROR] Failed to precompute prompt embeddings: {e}")
else:
    print("[INFO] SBERT not available, skipping prompt embeddings cache")

# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)
CORS(app)

load_dotenv()

# Load environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
mongodb_uri = os.getenv('MONGODB_URI')
mongodb_host = os.getenv('MONGODB_HOST', 'localhost')
mongodb_db_name = os.getenv('MONGODB_DB_NAME', 'verisync_analysis')
emotion_model_url = os.getenv('EMOTION_MODEL')
main_backend_url = os.getenv('MAIN_BACKEND_URL', 'http://localhost:3000')

if not groq_api_key:
    print("[ERROR] GROQ_API_KEY not found!")

try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    print(f"[ERROR] Failed to initialize Groq client: {e}")

@app.route('/create_call', methods=['POST'])
def create_call():
    print("[ROUTE] POST /create_call")
    # Use request.form to handle form data, consistent with other endpoints
    if 'call_id' not in request.form:
        return jsonify({"error": "call_id is a required field"}), 400
    
    call_id = request.form.get('call_id')
    agent_id = request.form.get('agent_id')
    customer_id = request.form.get('customer_id')
    script_text = request.form.get('transcript')

    call_ops = get_call_operations()
    success = call_ops.create_call(call_id, agent_id, customer_id, script_text)

    if success:
        return jsonify({"status": "success", "message": f"Call record for {call_id} created or already exists."}), 201
    else:
        return jsonify({"error": "Failed to create call record in database"}), 500

@app.route('/update_call', methods=['POST'])
def update_call():
    print("[ROUTE] POST /update_call")
    
    client_audio = request.files.get('client_audio')
    agent_audio = request.files.get('agent_audio')
    call_id = request.form.get('call_id')
    transcript = request.form.get('transcript')
    
    if not transcript:
        print("[UPDATE_CALL] No transcript provided, attempting to load from audioscript.txt")
        transcript = load_script_from_file("audioscript.txt")

    try:
        if call_id not in call_audio_buffers:
            call_audio_buffers[call_id] = {'agent_audio': BytesIO(), 'client_audio': BytesIO()}
        
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)
        if not existing_call:
            return jsonify({"error": "No record found for the given CallID for updating"}), 404
        
        existing_transcription = existing_call.get('transcription', {})

        # --- Process Agent Audio ---
        agent_chunk_bytes = agent_audio.read()
        agent_audio.seek(0)
        previous_agent_bytes = call_audio_buffers[call_id]['agent_audio'].getvalue()
        full_agent_audio_bytes = append_wav_chunks(previous_agent_bytes, agent_chunk_bytes)
        call_audio_buffers[call_id]['agent_audio'] = BytesIO(full_agent_audio_bytes)
        
        total_agent_segments = []
        if full_agent_audio_bytes:
            _, _, _, all_words = speech_to_text(BytesIO(full_agent_audio_bytes))
            total_agent_segments = resegment_based_on_punctuation(all_words)
        
        total_agent_text = " ".join([s['text'] for s in total_agent_segments]).strip()


        # --- Process Client Audio ---
        client_chunk_bytes = client_audio.read()
        client_audio.seek(0)
        previous_client_bytes = call_audio_buffers[call_id]['client_audio'].getvalue()
        full_client_audio_bytes = append_wav_chunks(previous_client_bytes, client_chunk_bytes)
        call_audio_buffers[call_id]['client_audio'] = BytesIO(full_client_audio_bytes)
        
        total_client_segments = []
        if full_client_audio_bytes:
            _, _, _, all_words = speech_to_text(BytesIO(full_client_audio_bytes))
            total_client_segments = resegment_based_on_punctuation(all_words)
        
        total_client_text = " ".join([s['text'] for s in total_client_segments]).strip()

        
        # --- Analysis and Response ---
        # Finalize and Analyze
        if not total_agent_text and not total_client_text:
            return jsonify({"status": "success", "message": "Audio silent, no analysis."}), 200

        # This duration is now an approximation of the chunk, not the whole call
        duration = max(len(agent_chunk_bytes), len(client_chunk_bytes)) / 32000 
        
        script_checkpoints = get_current_call_script(transcript)
        
        combined_transcription = f"Agent: {total_agent_text}\nClient: {total_client_text}".strip()
        
        processed_agent_segments = process_whisper_segments(total_agent_segments)
        processed_client_segments = process_whisper_segments(total_client_segments)
        
        agent_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Agent', 'text': s['text']} for s in processed_agent_segments]
        client_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Client', 'text': s['text']} for s in processed_client_segments]
        all_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])

        all_transcription = {
            "agent": total_agent_text,
            "client": total_client_text,
            "combined": combined_transcription,
            "timestamped_dialogue": all_timestamped_dialogue,
            "agent_segments": total_agent_segments,
            "client_segments": total_client_segments
        }

        stored_audio_result = call_ops.store_audio_and_update_call(call_id, agent_audio, client_audio, is_final=False)
        
        results = check_script_adherence_adaptive_windows(total_agent_text, processed_agent_segments, script_checkpoints)
        overall_adherence = results.get("real_time_adherence_score", 0)
        script_completion = results.get("script_completion_percentage", 0)
        
        adherence_data = {
            'overall': overall_adherence,
            'script_completion': script_completion,
            'details': results.get("checkpoint_results", []),
            'window_size_usage': results.get("window_size_usage", {})
        }

        response = get_response(total_client_text)
        CQS, emotions = calculate_cqs(response["predictions"])
        quality = get_quality(emotions)

        call_ops.insert_partial_update(call_id, duration, CQS, adherence_data, emotions, all_transcription, quality)

        return jsonify({
            "status": "success",
            "call_id": call_id,
            "overall_adherence": overall_adherence,
            "script_completion": script_completion,
            "adherence_details": results.get("checkpoint_results", [])
        })
    
    except Exception as e:
        print(f"[UPDATE_CALL ERROR] An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected server error occurred"}), 500

@app.route('/final_update', methods=['POST'])
def final_update():
    print("[ROUTE] POST /final_update")
    
    client_audio = request.files.get('client_audio')
    agent_audio = request.files.get('agent_audio')
    call_id = request.form.get('call_id')
    transcript = request.form.get('transcript')
    
    if not transcript:
        print("[FINAL_UPDATE] No transcript provided, using default")
        transcript = load_script_from_file("audioscript.txt")

    try:
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)
        if not existing_call:
            return jsonify({"error": "No record found for the given CallID for updating"}), 404

        existing_transcription = existing_call.get('transcription', {})

        # --- Process Agent Audio ---
        agent_chunk_bytes = agent_audio.read()
        agent_audio.seek(0)
        previous_agent_bytes = call_audio_buffers.get(call_id, {}).get('agent_audio', BytesIO()).getvalue()
        full_agent_audio_bytes = append_wav_chunks(previous_agent_bytes, agent_chunk_bytes)
        
        final_agent_segments = []
        if full_agent_audio_bytes:
            _, _, _, all_words = speech_to_text(BytesIO(full_agent_audio_bytes))
            final_agent_segments = resegment_based_on_punctuation(all_words)

        final_agent_text = " ".join([s['text'] for s in final_agent_segments]).strip()


        # --- Process Client Audio ---
        client_chunk_bytes = client_audio.read()
        client_audio.seek(0)
        previous_client_bytes = call_audio_buffers.get(call_id, {}).get('client_audio', BytesIO()).getvalue()
        full_client_audio_bytes = append_wav_chunks(previous_client_bytes, client_chunk_bytes)
        
        final_client_segments = []
        if full_client_audio_bytes:
            _, _, _, all_words = speech_to_text(BytesIO(full_client_audio_bytes))
            final_client_segments = resegment_based_on_punctuation(all_words)

        final_client_text = " ".join([s['text'] for s in final_client_segments]).strip()

        
        # --- Final Analysis ---
        script_checkpoints = get_current_call_script(transcript)
        results = check_script_adherence_adaptive_windows(final_agent_text, final_agent_segments, script_checkpoints)
        
        combined_transcription = f"Agent: {final_agent_text}\nClient: {final_client_text}".strip()
        processed_agent_segments = process_whisper_segments(final_agent_segments)
        processed_client_segments = process_whisper_segments(final_client_segments)
        
        agent_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Agent', 'text': s['text']} for s in processed_agent_segments]
        client_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Client', 'text': s['text']} for s in processed_client_segments]
        final_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])

        response = get_response(final_client_text)
        CQS, emotions = calculate_cqs(response["predictions"])
        quality = get_quality(emotions)
        agent_quality = agent_scores(final_agent_text)
        summary = call_summary(final_agent_text, final_client_text)
        tags_obj = get_tags(f"Client:\n{final_client_text}\nAgent:\n{final_agent_text}")
        tags = ', '.join(tags_obj.get('Tags', []))

        chunk_duration = max(len(agent_chunk_bytes), len(client_chunk_bytes)) / 32000
        total_duration = existing_call.get('duration', 0) + chunk_duration

        final_adherence_data = {
            "overall": results.get("real_time_adherence_score", 0),
            "script_completion": results.get("script_completion_percentage", 0),
            "details": results.get("checkpoint_results", []),
            "window_size_usage": results.get("window_size_usage", {}),
            "method": "adaptive_windows_segments"
        }
        
        call_ops.complete_call_update(
            call_id=call_id,
            agent_text=final_agent_text, client_text=final_client_text,
            combined=combined_transcription, cqs=CQS,
            overall_adherence=final_adherence_data, agent_quality=agent_quality,
            summary=summary, emotions=emotions, duration=total_duration,
            quality=quality, tags=tags, timestamped_dialogue=final_timestamped_dialogue,
            agent_segments=final_agent_segments, client_segments=final_client_segments
        )

        if call_id in call_audio_buffers:
            del call_audio_buffers[call_id]
            print(f"[BUFFER_CLEANUP] Cleared audio buffer for completed call {call_id}")

        return jsonify({
            "status": "success",
            "message": "Call analysis completed.",
            "call_id": call_id,
            "final_adherence": final_adherence_data
        })
    
    except Exception as e:
        print(f"[FINAL_UPDATE ERROR] An unexpected error occurred: {e}")
        return jsonify({"error": "An unexpected server error occurred"}), 500

if __name__ == '__main__':
    # Configure Flask for production deployment or local development
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = '0.0.0.0'
    
    print(f"[SERVER] Starting Flask server at http://{host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        print(f"[SERVER ERROR] Failed to start server: {e}")
    finally:
        print("[SERVER] Server stopped") 