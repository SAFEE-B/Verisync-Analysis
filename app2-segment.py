import os
from datetime import datetime, timezone
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
import traceback # Make sure to import traceback at the top of your file for better error logging
import soundfile as sf
import numpy as np
import torchaudio

# =============================================================================
# GLOBAL STATE & HELPERS
# =============================================================================

# REMOVED: Global in-memory buffers - now using database-backed storage for concurrent processing
# All audio chunks and segments are stored in MongoDB with GridFS for thread-safe operations

# Silent Audio Detection Configuration
SILENT_AUDIO_THRESHOLD = 1e-12  # no_speech_prob threshold above which segments are filtered out



def resample_audio(audio_chunk, target_sr=24000):
    """
    Resamples an audio chunk to the target sample rate.

    Args:
        audio_chunk (FileStorage): The audio chunk from the request.
        target_sr (int): The target sample rate.

    Returns:
        BytesIO: The resampled audio data in a BytesIO object.
    """
    if not audio_chunk:
        return None

    try:
        # Read the audio data and its original sample rate
        audio_data, original_sr = sf.read(audio_chunk)

        # If the audio is already at the target sample rate, no need to resample
        if original_sr == target_sr:
            audio_chunk.seek(0)
            return BytesIO(audio_chunk.read())

        print(f"[RESAMPLE] Resampling audio from {original_sr} Hz to {target_sr} Hz")

        # Convert to tensor for torchaudio
        audio_tensor = torch.from_numpy(audio_data).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0) # Add channel dimension if mono

        # Create the resampler
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        resampled_tensor = resampler(audio_tensor)

        # Convert back to numpy array
        resampled_data = resampled_tensor.squeeze(0).numpy()

        # Save the resampled audio to a BytesIO object
        resampled_buffer = BytesIO()
        sf.write(resampled_buffer, resampled_data, target_sr, format='WAV', subtype='PCM_16')
        resampled_buffer.seek(0)

        return resampled_buffer

    except Exception as e:
        print(f"[RESAMPLE ERROR] Failed to resample audio: {e}")
        # Fallback: return the original audio chunk if resampling fails
        audio_chunk.seek(0)
        return BytesIO(audio_chunk.read())

def merge_transcription_segments(previous_segments, new_segments, overlap_start_time):
    """
    Merges new transcription segments with a previous set by replacing the overlapping portion.

    Args:
        previous_segments (list): The list of segments from the prior transcription.
        new_segments (list): The list of segments from the new, overlapping transcription.
        overlap_start_time (float): The timestamp where the re-transcription began.

    Returns:
        list: The final, merged list of segments.
    """
    if not previous_segments:
        return new_segments

    # Remove all previous segments that start at or after the overlap point
    base_segments = [seg for seg in previous_segments if seg.get('start', 0) < overlap_start_time]

    # The new segments' timestamps should already be absolute (adjusted in transcription)
    # So we just append them directly
    final_segments = base_segments + new_segments

    print(f"[MERGE_SEGMENTS] Merged {len(base_segments)} base segments with {len(new_segments)} new segments. Total: {len(final_segments)}")
    return final_segments


# REMOVED: slice_audio - now handled by database get_audio_slice method

# REMOVED: append_wav_chunks - now handled by database-backed storage with proper WAV concatenation

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
        
        'paraphrase-MiniLM-L3-v2',
        'all-MiniLM-L12-v2',
        'all-MiniLM-L6-v2'
    ]
    
    sbert_model = None
    for model_name in model_options:
        try:
            print(f"[INFO] Trying to load model: {model_name}")
            sbert_model = SentenceTransformer(model_name)
            
            if torch.cuda.is_available():
                # sbert_model = sbert_model.to('cuda')
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
        
        print(f"[SEGMENTS] Segment {i+1}: '{text}' ({word_count} words, {duration:.2f}s), start {start_time}, end {end_time}")
        
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

        # print(f"\n[ADHERENCE] Analyzing Checkpoint {i+1}: {checkpoint_id} ({adherence_type})")

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
            for window_size in range(1, 2):
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

            # print(f"  - Script: '{script_sentence}'")
            # print(f"    Agent Match (Window): '{agent_match_text}' -> Score: {best_score_for_sentence:.2f}% ({sentence_status})")
            # print(f"    Adjusted Score for Averaging: {adjusted_score_for_averaging:.2f}%")

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

    print("EMOTION RECEIVED:", emotions)
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
    """TEMPORARY FIX: Disable VAD processing, just return duration."""
    try:
        # Read audio and get duration using torchaudio, but skip all VAD/model logic
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
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            wav = resampler(wav)
            sr = 24000
        wav = wav.squeeze()
        total_duration = len(wav) / sr
        return {
            'has_speech': True,
            'speech_segments': [],
            'total_speech_duration': 0,
            'total_audio_duration': total_duration,
            'speech_ratio': 1.0,
            'vad_available': False
        }
    except Exception as e:
        return {
            'has_speech': True,
            'speech_segments': [],
            'total_speech_duration': 0,
            'total_audio_duration': 0,
            'speech_ratio': 1.0,
            'vad_available': False,
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
        
        print("[API RESPONSE] Groq Whisper segment transcription completed",response)
        
        if hasattr(response, 'text'):
            transcription = response.text or ""
        else:
            transcription = response.get('text', '') if isinstance(response, dict) else ""
        
        duration = vad_result['total_audio_duration']
        print('Duration is',duration)
        # Process segment-level data
        segment_data = []
        segments = None
        if hasattr(response, 'segments') and response.segments:
            segments = response.segments
        elif isinstance(response, dict) and 'segments' in response and response['segments']:
            segments = response['segments']
        
        if segments:
            for segment in segments:
                # Extract segment data including no_speech_prob
                # Handle dictionary-style segments first (Groq format)
                if isinstance(segment, dict):
                    no_speech_prob = segment.get('no_speech_prob', 0.0)
                    segment_info = {
                        'text': segment.get('text', '').strip(),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'confidence': segment.get('avg_logprob', 0.0),
                        'no_speech_prob': no_speech_prob
                    }
                # Handle object-style segments (backup for other APIs)
                elif hasattr(segment, 'text') and hasattr(segment, 'start') and hasattr(segment, 'end'):
                    no_speech_prob = getattr(segment, 'no_speech_prob', 0.0)
                    segment_info = {
                        'text': segment.text.strip(),
                        'start': segment.start,
                        'end': segment.end,
                        'confidence': getattr(segment, 'avg_logprob', 0.0),
                        'no_speech_prob': no_speech_prob
                    }
                else:
                    continue
                
                # Apply silent audio detection: filter out segments with high no_speech_prob
                if no_speech_prob > SILENT_AUDIO_THRESHOLD:
                    print(f"[SILENT_DETECTION] Filtering out silent segment: '{segment_info['text'][:50]}...' "
                          f"(no_speech_prob: {no_speech_prob:.2e}, duration: {segment_info['end'] - segment_info['start']:.2f}s)")
                else:
                    segment_data.append(segment_info)
                    print(f"[SEGMENT_KEPT] Keeping segment: '{segment_info['text'][:50]}...' "
                          f"(no_speech_prob: {no_speech_prob:.2e})")
        
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

        # Calculate silent detection statistics
        total_segments = len(segments) if segments else 0
        kept_segments = len(segment_data)
        filtered_segments = total_segments - kept_segments
        
        print(f"[API RESPONSE] Extracted {kept_segments}/{total_segments} segments and {len(word_data)} words from Whisper")
        if filtered_segments > 0:
            print(f"[SILENT_DETECTION] Filtered out {filtered_segments} silent segments (threshold: {SILENT_AUDIO_THRESHOLD})")
        
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
    
    url = "https://f9a470925488.ngrok-free.app/emotion"
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload, timeout=2.5)
        
        if response.status_code == 200:
            print("[API RESPONSE] Emotion model API response received")
            # print(f"{response}")
            return response.json()
        else:
            print("Sentiment server error:", response.text)
    except Exception as e:
        print("Sentiment server not responding:", str(e))
    return None

# def calculate_cqs(predictions):
#     """Calculate Customer Quality Score based on emotions"""
#     emotion_weights = {
#         "joy": 2.5, "gratitude": 2.0, "admiration": 1.75, "optimism": 1.5, "relief": 1.25, "caring": 1.0,
#         "neutral": 0, "curiosity": 0, "realization": 0,
#         "anger": -2.5, "disapproval": -2.25, "disappointment": -2.0, "annoyance": -1.75, "sadness": -1.5, "embarrassment": -1.0, "confusion": -0.5
#     }
#     cqs = 0
#     emotions = predictions
#     emotions = dict(emotions[0])

#     print(emotions,"------------------------")
#     confidence = emotions['score']
#     emotion_recieved = emotions['label'].lower()
#     if emotion_recieved in emotion_weights:
#         default_cqs = emotion_weights[emotion_recieved] * confidence
#     else:
#         default_cqs = 0.1    # for emotion in predictions:
#     #     label = emotion["label"].lower()
#     #     # score = emotion["score"]
#     #     if label in emotion_weights:
#     #         # emotions[label] = round(score, 2)
#     #         weighted_contribution = emotion_weights[label] * score
#     #         cqs += weighted_contribution
#     cqs = default_cqs
#     final_cqs = abs(round(cqs, 2))
#     result = [final_cqs, emotions]
#     return result

emotions_CACHE = []
def calculate_cqs(predictions, script_completion=0.0):
    """Calculate Call Quality Score (CQS) using emotions + script adherence"""
    print('Calculating call quality...')
    # Emotion weight mapping — positive emotions get higher influence
    emotion_weights = {
        "admiration": 0.7, "amusement": 0.8, "anger": -0.9, "annoyance": -0.7,
        "approval": 0.6, "caring": 0.6, "confusion": -0.5, "curiosity": 0.5,
        "desire": 0.4, "disappointment": -0.8, "disapproval": -0.6, "disgust": -0.9,
        "embarrassment": -0.5, "excitement": 1.0, "fear": -0.7, "gratitude": 1.2,
        "grief": -1.0, "joy": 1.0, "love": 1.2, "nervousness": -0.5, "optimism": 0.8,
        "pride": 0.5, "realization": 0.3, "relief": 0.5, "remorse": -0.4,
        "sadness": -0.6, "surprise": 0.2, "neutral": 0.0,
    }
    total_weighted_score = 0.0
    total_confidence = 0.0
    if not predictions:
        print("No predictions found.")
        return [0.0, None]
    current_emotion = dict(predictions[0])
    print("Processing emotion_CACHEs...")
    for emotion_group in emotions_CACHE:
        if not emotion_group or not isinstance(emotion_group, list):
            continue
        top_emotion = emotion_group[0]
        label = top_emotion['label'].lower()
        score = top_emotion['score']
        weight = emotion_weights.get(label, 0.0)
        total_weighted_score += weight * score
        total_confidence += score
    if total_confidence == 0:
        total_confidence = 1  # prevent division by zero
    # Calculate emotion score scaled to 0–100
    emotion_cqs = round((total_weighted_score / total_confidence) * 100, 2)
    # Normalize script_completion
    script_score = round(script_completion * 100, 2) if script_completion <= 1 else round(script_completion, 2)
    # Final CQS: Weighted sum (Emotion: 60%, Script: 40%)
    final_cqs = round((emotion_cqs * 0.6) + (script_score * 0.4), 2)
    print(f"Emotion CQS: {emotion_cqs} | Script Score: {script_score} | Final CQS: {final_cqs}")
    return [final_cqs, current_emotion]





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
    sip_id = request.form.get('sip_id')
    script_text = request.form.get('transcript')

    call_ops = get_call_operations()
    success = call_ops.create_call(call_id, agent_id, sip_id, script_text)

    if success:
        return jsonify({"status": "success", "message": f"Call record for {call_id} created or already exists."}), 201
    else:
        return jsonify({"error": "Failed to create call record in database"}), 500

@app.route('/update_call', methods=['POST'])
def update_call():
    """
    Processes an audio chunk for an ongoing call using the new optimized flow:
    1. Store chunks in DB and get audio from last segment start to chunk end
    2. Transcribe this audio segment
    3. Process analysis on the transcribed content
    4. Update segments and other data via insert_partial_update
    """
    print("[ROUTE] POST /update_call")
    
    client_audio_chunk = request.files.get('client_audio')
    agent_audio_chunk = request.files.get('agent_audio')

    # Resample audio to 16k Hz
    resampled_client_audio = resample_audio(client_audio_chunk)
    resampled_agent_audio = resample_audio(agent_audio_chunk)
    call_id = request.form.get('call_id')
    transcript = request.form.get('transcript')
    
    if not transcript:
        print("[UPDATE_CALL] No transcript provided, attempting to load from audioscript.txt")
        transcript = load_script_from_file("audioscript.txt")

    try:
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)
        if not existing_call:
            return jsonify({"error": "No record found for the given CallID for updating"}), 404
        
        # STEP 1: Store audio chunks and get processing data using the new optimized approach
        print("[UPDATE_CALL] Step 1: Storing chunks and getting processing audio...")
        processing_data = call_ops.store_audio_chunk_and_process(call_id, resampled_client_audio, resampled_agent_audio)
        
        # Extract processing results
        previous_agent_segments = processing_data.get("agent_segments", [])
        previous_client_segments = processing_data.get("client_segments", [])
        last_agent_segment_start = processing_data.get("agent_overlap_start", 0.0)
        last_client_segment_start = processing_data.get("client_overlap_start", 0.0)
        agent_audio_for_transcription = processing_data.get("agent_audio_for_transcription")
        client_audio_for_transcription = processing_data.get("client_audio_for_transcription")
        processing_context = processing_data.get("processing_context", {})
        
        print(f"[UPDATE_CALL] Processing context: {processing_context}")
        
        # Handle BytesIO objects for size reporting
        agent_audio_size = 0
        client_audio_size = 0
        
        if agent_audio_for_transcription:
            if hasattr(agent_audio_for_transcription, 'getvalue'):
                agent_audio_size = len(agent_audio_for_transcription.getvalue())
            else:
                agent_audio_size = len(agent_audio_for_transcription)
        
        if client_audio_for_transcription:
            if hasattr(client_audio_for_transcription, 'getvalue'):
                client_audio_size = len(client_audio_for_transcription.getvalue())
            else:
                client_audio_size = len(client_audio_for_transcription)
        
        print(f"[UPDATE_CALL] Agent audio from {last_agent_segment_start}s: {agent_audio_size} bytes")
        print(f"[UPDATE_CALL] Client audio from {last_client_segment_start}s: {client_audio_size} bytes")
        
        # STEP 2: Transcribe the audio segment from last segment start to chunk end
        print("[UPDATE_CALL] Step 2: Transcribing audio segment...")
        import concurrent.futures
        
        def transcribe_agent_audio():
            """Transcribe agent audio segment in parallel"""
            if agent_audio_for_transcription:
                audio_size = agent_audio_for_transcription.getvalue().__len__() if hasattr(agent_audio_for_transcription, 'getvalue') else len(agent_audio_for_transcription)
                print(f"[AGENT_TRANSCRIBE] Processing {audio_size} bytes from {last_agent_segment_start}s onward")
                
                # Handle both BytesIO objects and raw bytes
                if hasattr(agent_audio_for_transcription, 'seek'):
                    agent_audio_for_transcription.seek(0)  # Reset position if it's a BytesIO
                    audio_input = agent_audio_for_transcription
                else:
                    audio_input = BytesIO(agent_audio_for_transcription)
                
                _, _, _, all_words = speech_to_text(audio_input)
                # Adjust timestamps to account for the segment start time
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_agent_segment_start
                    segment['end'] += last_agent_segment_start
                return adjusted_segments
            return []
        
        def transcribe_client_audio():
            """Transcribe client audio segment in parallel"""
            if client_audio_for_transcription:
                audio_size = client_audio_for_transcription.getvalue().__len__() if hasattr(client_audio_for_transcription, 'getvalue') else len(client_audio_for_transcription)
                print(f"[CLIENT_TRANSCRIBE] Processing {audio_size} bytes from {last_client_segment_start}s onward")
                
                # Handle both BytesIO objects and raw bytes
                if hasattr(client_audio_for_transcription, 'seek'):
                    client_audio_for_transcription.seek(0)  # Reset position if it's a BytesIO
                    audio_input = client_audio_for_transcription
                else:
                    audio_input = BytesIO(client_audio_for_transcription)
                
                _, _, _, all_words = speech_to_text(audio_input)
                # Adjust timestamps to account for the segment start time
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_client_segment_start
                    segment['end'] += last_client_segment_start
                return adjusted_segments
            return []
        
        # Execute transcriptions in parallel
        import time
        start_time = time.time()
        print("[UPDATE_CALL] Starting parallel transcription...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            agent_future = executor.submit(transcribe_agent_audio)
            client_future = executor.submit(transcribe_client_audio)
            
            # Wait for both to complete
            new_agent_segments = agent_future.result()
            new_client_segments = client_future.result()
        
        end_time = time.time()
        transcription_time = end_time - start_time
        print(f"[UPDATE_CALL] Transcription completed in {transcription_time:.2f}s")
        print(f"[UPDATE_CALL] New agent segments: {len(new_agent_segments)}")
        print(f"[UPDATE_CALL] New client segments: {len(new_client_segments)}")
        
        # STEP 3: Combine previous segments with new segments for complete conversation
        print("[UPDATE_CALL] Step 3: Building complete conversation...")
        # Combine existing segments with new ones (new segments replace overlapping ones)
        all_agent_segments = merge_transcription_segments(previous_agent_segments, new_agent_segments, last_agent_segment_start)
        all_client_segments = merge_transcription_segments(previous_client_segments, new_client_segments, last_client_segment_start)
        print("All client segments after processing", all_agent_segments)
        #  sort by time
        all_agent_segments = sorted(all_agent_segments, key=lambda x: x.get('start', 0))
        all_client_segments = sorted(all_client_segments, key=lambda x: x.get('start', 0))
        
        # Generate text from segments
        total_agent_text = " ".join([s['text'] for s in all_agent_segments]).strip()
        total_client_text = " ".join([s['text'] for s in all_client_segments]).strip()
        
        print(f"[UPDATE_CALL] Total agent text length: {len(total_agent_text)} chars")
        print(f"[UPDATE_CALL] Total client text length: {len(total_client_text)} chars")
        
        # STEP 4: Perform analysis on the complete conversation
        print("[UPDATE_CALL] Step 4: Performing analysis...")
        
        if not total_agent_text and not total_client_text:
            print("[UPDATE_CALL] No text to analyze, returning success")
            return jsonify({"status": "success", "message": "Audio silent, no analysis performed."}), 200

        # Calculate duration (approximate from audio size) - reuse the sizes we calculated above
        duration = max(agent_audio_size, client_audio_size) / 32000  # Approximate duration
        print("second calculated duration",duration)
        # Prepare data for analysis
        script_checkpoints = get_current_call_script(transcript)
        combined_transcription = f"Agent: {total_agent_text}\nClient: {total_client_text}".strip()
        
        processed_agent_segments = process_whisper_segments(all_agent_segments)
        processed_client_segments = process_whisper_segments(all_client_segments)
        
        agent_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Agent', 'text': s['text']} for s in processed_agent_segments]
        client_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Client', 'text': s['text']} for s in processed_client_segments]
        all_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])

        all_transcription = {
            "agent": total_agent_text,
            "client": total_client_text,
            "combined": combined_transcription,
            "timestamped_dialogue": all_timestamped_dialogue,
            "agent_segments": all_agent_segments,
            "client_segments": all_client_segments
        }
        print("Total Transcription is: ",all_transcription)
        # Perform analysis
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
        emotions_CACHE.append(response)
        CQS, emotions = calculate_cqs(response,adherence_data.get('script_completion', 0.0))
        quality = get_quality(emotions)

        # STEP 5: Update database with all the processed data
        print("[UPDATE_CALL] Step 5: Updating database with processed results...")
        call_ops.insert_partial_update(call_id, duration, CQS, adherence_data, emotions, all_transcription, quality)

        print("[UPDATE_CALL] Successfully completed update_call")
        return jsonify({
            "status": "success",
            "call_id": call_id,
            "overall_adherence": overall_adherence,
            "script_completion": script_completion,
            "adherence_details": results.get("checkpoint_results", []),
            "processing_stats": {
                "transcription_time": transcription_time,
                "agent_segments": len(all_agent_segments),
                "client_segments": len(all_client_segments),
                "audio_processed": {
                    "agent_bytes": agent_audio_size,
                    "client_bytes": client_audio_size
                }
            }
        })
    
    except Exception as e:
        print(f"[UPDATE_CALL ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred"}), 500

@app.route('/final_update', methods=['POST'])
def final_update():
    """
    Processes the final audio chunk for a call using the same optimized flow as update_call:
    1. Store chunks in DB and get audio from last segment start to chunk end
    2. Transcribe this audio segment (or complete audio if no previous segments)
    3. Process analysis on the complete conversation
    4. Update segments and finalize call data
    """
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

        # STEP 1: Store final audio chunks and get processing data using the same optimized approach as update_call
        print("[FINAL_UPDATE] Step 1: Storing final chunks and getting processing audio...")
        processing_data = call_ops.store_audio_chunk_and_process(call_id, client_audio, agent_audio)
        
        # Extract processing results
        previous_agent_segments = processing_data.get("agent_segments", [])
        previous_client_segments = processing_data.get("client_segments", [])
        last_agent_segment_start = processing_data.get("agent_overlap_start", 0.0)
        last_client_segment_start = processing_data.get("client_overlap_start", 0.0)
        agent_audio_for_transcription = processing_data.get("agent_audio_for_transcription")
        client_audio_for_transcription = processing_data.get("client_audio_for_transcription")
        processing_context = processing_data.get("processing_context", {})
        
        print(f"[FINAL_UPDATE] Processing context: {processing_context}")
        
        # Handle BytesIO objects for size reporting
        agent_audio_size = 0
        client_audio_size = 0
        
        if agent_audio_for_transcription:
            if hasattr(agent_audio_for_transcription, 'getvalue'):
                agent_audio_size = len(agent_audio_for_transcription.getvalue())
            else:
                agent_audio_size = len(agent_audio_for_transcription)
        
        if client_audio_for_transcription:
            if hasattr(client_audio_for_transcription, 'getvalue'):
                client_audio_size = len(client_audio_for_transcription.getvalue())
            else:
                client_audio_size = len(client_audio_for_transcription)
        
        print(f"[FINAL_UPDATE] Agent audio from {last_agent_segment_start}s: {agent_audio_size} bytes")
        print(f"[FINAL_UPDATE] Client audio from {last_client_segment_start}s: {client_audio_size} bytes")
        
        # STEP 2: Transcribe the audio segment (or complete audio if no previous segments)
        print("[FINAL_UPDATE] Step 2: Transcribing final audio segment...")
        import concurrent.futures
        
        def transcribe_agent_audio():
            """Transcribe agent audio segment in parallel"""
            if agent_audio_for_transcription:
                audio_size = agent_audio_for_transcription.getvalue().__len__() if hasattr(agent_audio_for_transcription, 'getvalue') else len(agent_audio_for_transcription)
                print(f"[FINAL_AGENT_TRANSCRIBE] Processing {audio_size} bytes from {last_agent_segment_start}s onward")
                
                # Handle both BytesIO objects and raw bytes
                if hasattr(agent_audio_for_transcription, 'seek'):
                    agent_audio_for_transcription.seek(0)  # Reset position if it's a BytesIO
                    audio_input = agent_audio_for_transcription
                else:
                    audio_input = BytesIO(agent_audio_for_transcription)
                
                _, _, _, all_words = speech_to_text(audio_input)
                # Adjust timestamps to account for the segment start time
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_agent_segment_start
                    segment['end'] += last_agent_segment_start
                return adjusted_segments
            return []
        
        def transcribe_client_audio():
            """Transcribe client audio segment in parallel"""
            if client_audio_for_transcription:
                audio_size = client_audio_for_transcription.getvalue().__len__() if hasattr(client_audio_for_transcription, 'getvalue') else len(client_audio_for_transcription)
                print(f"[FINAL_CLIENT_TRANSCRIBE] Processing {audio_size} bytes from {last_client_segment_start}s onward")
                
                # Handle both BytesIO objects and raw bytes
                if hasattr(client_audio_for_transcription, 'seek'):
                    client_audio_for_transcription.seek(0)  # Reset position if it's a BytesIO
                    audio_input = client_audio_for_transcription
                else:
                    audio_input = BytesIO(client_audio_for_transcription)
                
                _, _, _, all_words = speech_to_text(audio_input)
                # Adjust timestamps to account for the segment start time
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_client_segment_start
                    segment['end'] += last_client_segment_start
                return adjusted_segments
            return []
        
        # Execute transcriptions in parallel
        import time
        start_time = time.time()
        print("[FINAL_UPDATE] Starting parallel transcription...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            agent_future = executor.submit(transcribe_agent_audio)
            client_future = executor.submit(transcribe_client_audio)
            
            # Wait for both to complete
            new_agent_segments = agent_future.result()
            new_client_segments = client_future.result()
        
        end_time = time.time()
        transcription_time = end_time - start_time
        print(f"[FINAL_UPDATE] Transcription completed in {transcription_time:.2f}s")
        print(f"[FINAL_UPDATE] New agent segments: {len(new_agent_segments)}")
        print(f"[FINAL_UPDATE] New client segments: {len(new_client_segments)}")
        
        # STEP 3: Merge segments using the same logic as update_call
        print("[FINAL_UPDATE] Step 3: Building complete conversation...")
        
        # Use merge_transcription_segments for consistency
        final_agent_segments = merge_transcription_segments(previous_agent_segments, new_agent_segments, last_agent_segment_start)
        final_client_segments = merge_transcription_segments(previous_client_segments, new_client_segments, last_client_segment_start)
        
        # Sort by time
        final_agent_segments = sorted(final_agent_segments, key=lambda x: x.get('start', 0))
        final_client_segments = sorted(final_client_segments, key=lambda x: x.get('start', 0))
        
        # Generate text from segments
        final_agent_text = " ".join([s['text'] for s in final_agent_segments]).strip()
        final_client_text = " ".join([s['text'] for s in final_client_segments]).strip()
        
        print(f"[FINAL_UPDATE] Final agent text length: {len(final_agent_text)} chars")
        print(f"[FINAL_UPDATE] Final client text length: {len(final_client_text)} chars")
        
        # STEP 4: Perform final analysis on the complete conversation
        print("[FINAL_UPDATE] Step 4: Performing final analysis...")
        
        if not final_agent_text and not final_client_text:
            print("[FINAL_UPDATE] No text to analyze, completing with minimal data")
            # Still complete the call even if no text
            call_ops.complete_call_update(
                call_id=call_id,
                agent_text="", client_text="",
                combined="", cqs=0,
                overall_adherence={}, agent_quality={},
                summary="", emotions={}, duration=existing_call.get('duration', 0),
                quality=0, tags="", timestamped_dialogue=[],
                agent_segments=[], client_segments=[]
            )
            return jsonify({"status": "success", "message": "Call completed with no transcription data."})

        # Calculate duration (approximate from audio size) - reuse the sizes we calculated above
        chunk_duration = max(agent_audio_size, client_audio_size) / 32000  # Approximate duration
        total_duration = existing_call.get('duration', 0) + chunk_duration
        
        # Prepare data for analysis
        script_checkpoints = get_current_call_script(transcript)
        combined_transcription = f"Agent: {final_agent_text}\nClient: {final_client_text}".strip()
        
        processed_agent_segments = process_whisper_segments(final_agent_segments)
        processed_client_segments = process_whisper_segments(final_client_segments)
        
        agent_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Agent', 'text': s['text']} for s in processed_agent_segments]
        client_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Client', 'text': s['text']} for s in processed_client_segments]
        final_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])

        # Perform final analysis
        results = check_script_adherence_adaptive_windows(final_agent_text, processed_agent_segments, script_checkpoints)
        
        final_adherence_data = {
            "overall": results.get("real_time_adherence_score", 0),
            "script_completion": results.get("script_completion_percentage", 0),
            "details": results.get("checkpoint_results", []),
            "window_size_usage": results.get("window_size_usage", {}),
            "method": "adaptive_windows_segments"
        }

        # Get final emotion analysis for complete client transcript
        print(f"[FINAL_UPDATE] Analyzing final emotions for complete client transcript ({len(final_client_text)} chars)")
        final_emotion_response = get_response(final_client_text)
        final_CQS, final_emotions = calculate_cqs(final_emotion_response, final_adherence_data.get('script_completion', 0.0))
        final_quality = get_quality(final_emotions)
        
        print(f"[FINAL_UPDATE] Final emotion analysis complete:")
        print(f"[FINAL_UPDATE] - Final CQS: {final_CQS}")
        print(f"[FINAL_UPDATE] - Final Quality: {final_quality}")
        print(f"[FINAL_UPDATE] - Final Emotions: {final_emotions}")
        
        # Keep the last chunk emotions for compatibility
        response = get_response(final_client_text)
        CQS, emotions = calculate_cqs(response, final_adherence_data.get('script_completion', 0.0))
        quality = get_quality(emotions)
        
        agent_quality = agent_scores(final_agent_text)
        summary = call_summary(final_agent_text, final_client_text)
        tags_obj = get_tags(f"Client:\n{final_client_text}\nAgent:\n{final_agent_text}")
        tags = ', '.join(tags_obj.get('Tags', []))

        # STEP 5: Complete the call using the same structure as update_call but with complete_call_update
        print("[FINAL_UPDATE] Step 5: Completing call with final data...")
        
        call_ops.complete_call_update(
            call_id=call_id,
            agent_text=final_agent_text, client_text=final_client_text,
            combined=combined_transcription, cqs=final_CQS,
            overall_adherence=final_adherence_data, agent_quality=agent_quality,
            summary=summary, emotions=emotions, duration=total_duration,
            quality=final_quality, tags=tags, timestamped_dialogue=final_timestamped_dialogue,
            agent_segments=final_agent_segments, client_segments=final_client_segments,
            final_emotions=final_emotions
        )

        print("[FINAL_UPDATE] Successfully completed final_update")
        return jsonify({
            "status": "success",
            "message": "Call analysis completed.",
            "call_id": call_id,
            "final_adherence": final_adherence_data,
            "final_emotions": {
                "emotions": final_emotions,
                "cqs": final_CQS,
                "quality": final_quality
            },
            "total_duration": total_duration,
            "chunk_emotions_count": len(existing_call.get('emotions', [])) if existing_call else 0,
            "processing_stats": {
                "transcription_time": transcription_time,
                "agent_segments": len(final_agent_segments),
                "client_segments": len(final_client_segments),
                "audio_processed": {
                    "agent_bytes": agent_audio_size,
                    "client_bytes": client_audio_size
                }
            }
        })
    
    except Exception as e:
        print(f"[FINAL_UPDATE ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred"}), 500

@app.route('/cleanup_call', methods=['POST'])
def cleanup_call():
    """
    Clean up resources for a completed call (optional endpoint for storage optimization)
    """
    print("[ROUTE] POST /cleanup_call")
    
    call_id = request.form.get('call_id')
    
    if not call_id:
        return jsonify({"error": "call_id is required"}), 400
    
    try:
        call_ops = get_call_operations()
        call_ops.cleanup_call_resources(call_id)
        
        return jsonify({
            "status": "success",
            "message": f"Resources cleaned up for call {call_id}"
        })
        
    except Exception as e:
        print(f"[CLEANUP_ERROR] Error cleaning up call {call_id}: {e}")
        return jsonify({"error": "Failed to cleanup call resources"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for load balancer and monitoring
    """
    try:
        # Test database connection
        call_ops = get_call_operations()
        db_status = "connected" if call_ops.db.client.admin.command('ping') else "disconnected"
        
        return jsonify({
            "status": "healthy",
            "database": db_status,
            "sbert_available": SBERT_AVAILABLE,
            "vad_available": VAD_AVAILABLE,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

if __name__ == '__main__':
    # Configure Flask for production deployment with concurrent processing
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = '0.0.0.0'
    
    # Enable threading for concurrent request handling
    threaded = True
    processes = 1  # Use threading instead of multiprocessing for better resource sharing
    
    print(f"[SERVER] Starting Flask server at http://{host}:{port}")
    print(f"[SERVER] Threading enabled: {threaded}")
    print(f"[SERVER] Database-backed storage: Enabled")
    print(f"[SERVER] Concurrent processing: Ready")
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=threaded, processes=processes, use_reloader=False)
    except Exception as e:
        print(f"[SERVER ERROR] Failed to start server: {e}")
    finally:
        # Cleanup on server shutdown
        try:
            mongodb = get_db()
            mongodb.close()
            print("[SERVER] Database connection closed")
        except:
            pass
        print("[SERVER] Server stopped") 