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
# VERISYNC ANALYSIS BACKEND - SIMPLIFIED VERSION
# =============================================================================
# 
# Streamlined version with only core functionality:
# - POST /update_call: Process audio chunks with adaptive analysis
# - POST /final_update: Complete call analysis with final metrics
#
# Key Features:
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
                prompt_text = checkpoint.get("prompt_text", "")
                if prompt_text:
                    clean_prompt_preserve = clean_text_for_matching(prompt_text, preserve_structure=True)
                    clean_prompt_remove = clean_text_for_matching(prompt_text, preserve_structure=False)
                    
                    for prompt_version, version_key in [(clean_prompt_preserve, "preserve"), (clean_prompt_remove, "remove")]:
                        if prompt_version.strip():
                            cache_key = f"{checkpoint.get('checkpoint_id', 'unknown')}_{version_key}"
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

def detect_complete_thought(text):
    """
    Detect if text contains markers indicating a complete thought
    """
    original_text = text
    text = text.lower()
    
    # Check for sentence-final punctuation
    if any(punct in text for punct in ['.', '?', '!']):
        found_punct = [punct for punct in ['.', '?', '!'] if punct in text]
        print(f"[THOUGHT] ✓ Complete thought detected by punctuation: '{original_text}' (found: {found_punct})")
        return True, 'punctuation'
        
    # Check minimum word count for potential complete thought
    word_count = len(text.split())
    if word_count >= 5:
        print(f"[THOUGHT] ✓ Complete thought detected by length: '{original_text}' ({word_count} words >= 5)")
        return True, 'length'
    
    print(f"[THOUGHT] ✗ Incomplete thought: '{original_text}' ({word_count} words < 5, no punctuation)")
    return False, None

def create_adherence_optimized_sentences(words, script_sentences=None, min_confidence=0.7, use_semantic_matching=True):
    """
    Create sentences optimized for script adherence checking by combining
    multiple signals: semantic similarity, dialog acts, and pause detection
    If use_semantic_matching is False, semantic similarity is not used in boundary decisions.
    """
    if not words:
        print("[SENTENCE] No words provided, returning empty list")
        return []
    
    print(f"[SENTENCE] ========== Starting Sentence Creation ==========")
    print(f"[SENTENCE] Processing {len(words)} words with min_confidence={min_confidence}, use_semantic_matching={use_semantic_matching}")
    
    # Default script sentences if none provided
    if script_sentences is None:
        script_sentences = [
            "hello thanks for calling",
            "how can i help you today",
            "would you like to confirm",
            "thank you for choosing"
        ]
        print(f"[SENTENCE] Using default script sentences ({len(script_sentences)} phrases)")
    else:
        print(f"[SENTENCE] Using provided script sentences ({len(script_sentences)} phrases)")
        for idx, script in enumerate(script_sentences[:3]):  # Show first 3
            print(f"[SENTENCE] - Script {idx+1}: '{script[:50]}{'...' if len(script) > 50 else ''}'")
        if len(script_sentences) > 3:
            print(f"[SENTENCE] - ... and {len(script_sentences) - 3} more scripts")
    
    sentence_boundaries = []
    current_start = words[0]['timestamp'] if words else 0
    current_words = []
    
    print(f"[SENTENCE] Starting analysis at timestamp {current_start:.2f}s")
    print("-" * 60)
    
    for i in range(len(words)):
        current_words.append(words[i])
        current_text = ' '.join(w['word'] for w in current_words)
        current_word = words[i]['word']
        
        print(f"\n[SENTENCE] Word {i+1}/{len(words)}: '{current_word}' (timestamp: {words[i]['timestamp']:.2f}s)")
        print(f"[SENTENCE] Current sentence: '{current_text}' ({len(current_words)} words)")
        
        # Calculate pause duration to the *next* word, if it exists
        time_gap = (words[i + 1]['timestamp'] - words[i]['timestamp']) if i < len(words) - 1 else float('inf')
        print(f"[SENTENCE] Time gap to next word: {time_gap:.2f}s")
        
        # Get semantic similarity with script sentences (if enabled)
        if use_semantic_matching:
            semantic_score = get_semantic_match(current_text, script_sentences, sbert_model)
            print(f"[SENTENCE] Semantic similarity score: {semantic_score:.3f} ({semantic_score*100:.1f}%)")
            good_semantic_match = semantic_score > min_confidence
        else:
            semantic_score = 0.0
            good_semantic_match = False
            print(f"[SENTENCE] Semantic similarity skipped (use_semantic_matching=False)")
        
        # Check for complete thought
        is_complete, thought_type = detect_complete_thought(current_text)
        
        # Decision factors
        long_pause = time_gap > 0.7
        complete_thought = is_complete
        end_of_words = (i == len(words) - 1)
        max_length_reached = len(current_words) > 20

        print(f"[SENTENCE] Decision factors:")
        print(f"[SENTENCE] - Long pause (>0.7s): {long_pause} (gap: {time_gap:.2f}s)")
        print(f"[SENTENCE] - Good semantic match (>{min_confidence}): {good_semantic_match} (score: {semantic_score:.3f})")
        print(f"[SENTENCE] - Complete thought: {complete_thought} (type: {thought_type})")
        print(f"[SENTENCE] - End of words: {end_of_words}")
        print(f"[SENTENCE] - Max length reached (>20): {max_length_reached} (current: {len(current_words)})")

        # Decision logic
        should_break = end_of_words or \
                       (long_pause and len(current_words) > 3) or \
                       (good_semantic_match and complete_thought) or \
                       max_length_reached
        
        if should_break:
            # Determine break reason
            if end_of_words:
                break_reason = 'end_of_audio'
            elif long_pause and len(current_words) > 3:
                break_reason = 'pause'
            elif good_semantic_match and complete_thought:
                break_reason = 'semantic'
            elif max_length_reached:
                break_reason = 'length'
            else:
                break_reason = 'unknown'
            
            print(f"\n[SENTENCE] 🎯 BREAKING SENTENCE - Reason: {break_reason.upper()}")
            print(f"[SENTENCE] Final sentence: '{current_text}'")
            print(f"[SENTENCE] Duration: {words[i]['timestamp'] - current_start:.2f}s")
            print(f"[SENTENCE] Word count: {len(current_words)}")
            
            sentence_boundaries.append({
                'start_time': current_start,
                'end_time': words[i]['timestamp'],
                'text': current_text,
                'word_count': len(current_words),
                'metadata': {
                    'semantic_score': round(semantic_score * 100, 2),
                    'thought_type': thought_type,
                    'pause_duration': round(time_gap, 2) if not end_of_words else 0,
                    'break_reason': break_reason
                }
            })
            
            if not end_of_words:
                current_start = words[i + 1]['timestamp']
                print(f"[SENTENCE] Starting new sentence at timestamp {current_start:.2f}s")
            current_words = []
        else:
            print(f"[SENTENCE] ➡️ Continuing sentence...")
            
    print(f"\n[SENTENCE] ========== Sentence Creation Complete ==========")
    print(f"[SENTENCE] Created {len(sentence_boundaries)} sentences:")
    for idx, sentence in enumerate(sentence_boundaries):
        print(f"[SENTENCE] - Sentence {idx+1}: '{sentence['text'][:50]}{'...' if len(sentence['text']) > 50 else ''}' ({sentence['word_count']} words, {sentence['metadata']['break_reason']})")
    
    return sentence_boundaries

# =============================================================================
# CALL SCRIPT CONFIGURATION
# =============================================================================

DEFAULT_CALL_SCRIPT = """
<semantic> Hello, thank you for calling Customer Service </semantic>
<semantic> My name is [Agent Name], how can I assist you today? </semantic>
<topic> I understand how frustrating that must be. Let me check available options for you </topic>
<semantic> Can you provide your flight details? </semantic>
<semantic> I see your flight was canceled due to [Reason] </semantic>
<semantic> The next available flights are at [Times] </semantic>
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
    print("[SCRIPT PARSER] Starting script parsing...")
    
    if not script_text or not script_text.strip():
        print("[SCRIPT PARSER] No script text provided, using default script")
        return DEFAULT_CALL_SCRIPT
    
    import re
    
    # Pattern to match tags: <type>content</type> (case insensitive)
    pattern = r'<(strict|semantic|topic)>(.*?)</\1>'
    matches = re.findall(pattern, script_text, re.IGNORECASE | re.DOTALL)
    
    if not matches:
        print("[SCRIPT PARSER] No valid tags found, cannot parse script")
        return []
    
    checkpoints = []
    
    for i, (adherence_type, prompt_text) in enumerate(matches):
        adherence_type_upper = adherence_type.upper()
        prompt_text_clean = prompt_text.strip()
        
        if not prompt_text_clean:
            continue
        
        # Assign weights based on adherence type and position
        # Higher weights for stricter types and important positions
        base_weights = {"STRICT": 25, "SEMANTIC": 20, "TOPIC": 15}
        position_bonus = max(5, 25 - (i * 2))  # First checkpoints get higher weights
        weight = base_weights.get(adherence_type_upper, 15) + min(position_bonus, 10)
        
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
            "adherence_type": adherence_type_upper,
            "is_mandatory": is_mandatory,
            "weight": weight
        }
        
        checkpoints.append(checkpoint)
        print(f"[SCRIPT PARSER] Parsed checkpoint {i+1}: {adherence_type_upper} - Weight: {weight}, Mandatory: {is_mandatory} - '{prompt_text_clean[:50]}...'")
    
    print(f"[SCRIPT PARSER] Successfully parsed {len(checkpoints)} checkpoints")
    return checkpoints

def load_script_from_file(file_path="audioscript.txt"):
    """Load script from file in the analysis folder"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), file_path)
        print(f"[SCRIPT LOADER] Attempting to load script from: {script_path}")
        
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as file:
                script_content = file.read()
                print(f"[SCRIPT LOADER] Successfully loaded script file ({len(script_content)} characters)")
                return script_content
        else:
            print(f"[SCRIPT LOADER] Script file not found: {script_path}")
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
        print("[SCRIPT] Using script from request parameter")
        checkpoints = parse_script_from_text(script_text)
        if checkpoints:
            return checkpoints
        else:
            print("[SCRIPT] Request script parsing failed, falling back to default")
    
    # Try to load from file as fallback
    file_script = load_script_from_file()
    if file_script:
        print("[SCRIPT] Using script from audioscript.txt file")
        checkpoints = parse_script_from_text(file_script)
        if checkpoints:
            return checkpoints
        else:
            print("[SCRIPT] File script parsing failed, falling back to default")
    
    # Use default built-in script
    print("[SCRIPT] Using default built-in script")
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

def check_script_adherence_adaptive_windows(agent_text, agent_sentences, script_checkpoints=None, script_text=None):
    """
    ADAPTIVE WINDOW APPROACH - Tests window sizes 1, 2, and 3 for each checkpoint
    and automatically selects the best performing window size for optimal results.
    
    Args:
        agent_text: The agent's spoken text
        agent_sentences: Pre-tokenized sentences from the agent's text
        script_checkpoints: Pre-parsed checkpoint list (optional)
        script_text: Raw script text in XML format (optional)
    """
    print(f"[ADAPTIVE] ========== Starting Adaptive Window Analysis ==========")
    
    if script_checkpoints is None:
        script_checkpoints = get_current_call_script(script_text)

    if not agent_sentences:
        return {"real_time_adherence_score": 0, "script_completion_percentage": 0, "checkpoint_results": []}

    print(f"[ADAPTIVE] [SENTENCES] Agent said {len(agent_sentences)} sentences")
    print(f"[ADAPTIVE] [AGENT_TEXT] Full agent text: '{agent_text[:200]}{'...' if len(agent_text) > 200 else ''}'")
    
    # Show all agent sentences for reference
    for idx, sentence in enumerate(agent_sentences):
        print(f"[ADAPTIVE] [SENTENCE_{idx+1}] '{sentence}'")

    # Pre-compute sentence embeddings
    agent_embeddings = None
    if SBERT_AVAILABLE and sbert_model:
        try:
            agent_embeddings = encode_sentences_with_memory_optimization(agent_sentences, batch_size=32)
            print(f"[ADAPTIVE] [EMBEDDINGS] Computed embeddings for {len(agent_sentences)} sentences.")
        except Exception as e:
            print(f"[ADAPTIVE] [ERROR] Failed to compute embeddings: {e}")

    checkpoint_results = []
    
    for i, checkpoint in enumerate(script_checkpoints):
        checkpoint_id = checkpoint.get("checkpoint_id", f"unknown_{i}")
        prompt_text = checkpoint.get("prompt_text", "")
        adherence_type = checkpoint.get("adherence_type", "SEMANTIC")
        weight = checkpoint.get("weight", 1)
        is_mandatory = checkpoint.get("is_mandatory", False)
        
        print(f"\n[ADAPTIVE] [CHECKPOINT {i+1}] Analyzing: {checkpoint_id}")
        print(f"[ADAPTIVE] [EXPECTED] Script Text: '{prompt_text}'")
        threshold_map = {'STRICT': 85, 'SEMANTIC': 60, 'TOPIC': 40}
        threshold = threshold_map.get(adherence_type, 60)
        print(f"[ADAPTIVE] [TYPE] Adherence Type: {adherence_type} (Threshold: {threshold}%)")
        
        clean_prompt_preserve = clean_text_for_matching(prompt_text, preserve_structure=True)
        clean_prompt_remove = clean_text_for_matching(prompt_text, preserve_structure=False)
        
        # Pre-compute prompt embeddings
        prompt_embedding_preserve, prompt_embedding_remove = None, None
        if SBERT_AVAILABLE and sbert_model:
            try:
                prompt_embedding_preserve = get_cached_prompt_embedding(checkpoint_id, prompt_text, True)
                prompt_embedding_remove = get_cached_prompt_embedding(checkpoint_id, prompt_text, False)
            except Exception as e:
                print(f"[ADAPTIVE] [ERROR] Failed to get cached embeddings: {e}")

        best_overall_score = 0
        best_window_size = 1
        best_match_details = "No match found"
        best_sentence_location = None
        
        # Test each window size (1, 2, 3)
        for window_size in [1, 2, 3]:
            best_score_for_this_window_size = 0
            best_match_info = None
            
            # Create all possible windows of this size
            for start_idx in range(len(agent_sentences)):
                end_idx = min(start_idx + window_size - 1, len(agent_sentences) - 1)
                
                # Test each sentence within the window
                for sent_idx in range(start_idx, end_idx + 1):
                    current_sentence = agent_sentences[sent_idx]
                    score_for_sentence = 0
                    method_used = ""

                    if adherence_type == "STRICT":
                        if clean_prompt_preserve in current_sentence:
                            score_for_sentence = 100
                            method_used = "exact_preserve"
                        elif clean_prompt_remove in current_sentence:
                            score_for_sentence = 100
                            method_used = "exact_remove"
                        else:
                            score_preserve = fuzz.ratio(clean_prompt_preserve, current_sentence)
                            score_remove = fuzz.ratio(clean_prompt_remove, current_sentence)
                            score_for_sentence = max(score_preserve, score_remove)
                            method_used = "fuzzy"
                    
                    elif adherence_type == "SEMANTIC" and SBERT_AVAILABLE and agent_embeddings is not None:
                        sim_preserve, sim_remove = 0, 0
                        if prompt_embedding_preserve is not None:
                            try:
                                sim_preserve = util.cos_sim(prompt_embedding_preserve, agent_embeddings[sent_idx]).item()
                            except:
                                pass
                        if prompt_embedding_remove is not None:
                            try:
                                sim_remove = util.cos_sim(prompt_embedding_remove, agent_embeddings[sent_idx]).item()
                            except:
                                pass
                        
                        score_for_sentence = round(max(sim_preserve, sim_remove) * 100, 2)
                        method_used = "sbert"
                    
                    elif adherence_type == "TOPIC":
                        score_preserve = fuzz.ratio(clean_prompt_preserve, current_sentence)
                        score_remove = fuzz.ratio(clean_prompt_remove, current_sentence)
                        score_for_sentence = max(score_preserve, score_remove)
                        method_used = "fuzzy"

                    # Show detailed comparison for significant scores or first few attempts
                    if score_for_sentence > 30 or (window_size == 1 and sent_idx < 3):
                        print(f"[ADAPTIVE] [COMPARING] Agent sentence {sent_idx + 1}: '{current_sentence}'")
                        print(f"[ADAPTIVE] [SCORE] {method_used.upper()} Score: {score_for_sentence:.2f}%")

                    # Track the best score for this window size
                    if score_for_sentence > best_score_for_this_window_size:
                        best_score_for_this_window_size = score_for_sentence
                        best_match_info = {
                            "sentence_idx": sent_idx + 1,
                            "window_start": start_idx + 1,
                            "window_end": end_idx + 1,
                            "method": method_used,
                            "sentence_text": current_sentence[:50] + "..." if len(current_sentence) > 50 else current_sentence
                        }
                
                if best_score_for_this_window_size >= 100:
                    break
            
            # Check if this window size gave us the best score so far
            if best_score_for_this_window_size > best_overall_score:
                best_overall_score = best_score_for_this_window_size
                best_window_size = window_size
                best_sentence_location = best_match_info
                best_match_details = f"Best match {best_overall_score:.2f}% found using window size {window_size} at sentence {best_match_info['sentence_idx']}"

        # Determine final pass/fail status
        threshold = {"STRICT": 85, "SEMANTIC": 60, "TOPIC": 40}.get(adherence_type, 60)
        status = "PASS" if best_overall_score >= threshold else "FAIL"

        # Show final comparison summary
        if best_sentence_location:
            print(f"[ADAPTIVE] [BEST_MATCH] Agent said: '{best_sentence_location['sentence_text']}'")
            print(f"[ADAPTIVE] [BEST_MATCH] Expected: '{prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}'")
        
        print(f"[ADAPTIVE] [CHECKPOINT {i+1}] Winner: Window Size {best_window_size} with {best_overall_score:.2f}% - {status}")
        print(f"[ADAPTIVE] [RESULT] {'✓ PASSED' if status == 'PASS' else '✗ FAILED'} - Threshold: {threshold}%")

        checkpoint_results.append({
            "checkpoint_id": checkpoint_id,
            "status": status,
            "score": best_overall_score,
            "weight": weight,
            "is_mandatory": is_mandatory,
            "match_details": best_match_details,
            "optimal_window_size": best_window_size,
            "best_sentence": best_sentence_location
        })

    # Calculate final score using real-time heuristic
    last_passed_index = -1
    passed_checkpoints = sum(1 for r in checkpoint_results if r["status"] == "PASS")
    
    for i, result in enumerate(checkpoint_results):
        if result["status"] == "PASS":
            last_passed_index = i
            
    script_completion_percentage = round((passed_checkpoints / len(checkpoint_results)) * 100, 2)
    
    if last_passed_index >= 0:
        relevant_checkpoints = checkpoint_results[:last_passed_index + 1]
        total_weight = sum(r["weight"] for r in relevant_checkpoints)
        weighted_score_sum = sum(r["score"] * r["weight"] for r in relevant_checkpoints)
        real_time_adherence_score = round(weighted_score_sum / total_weight, 2) if total_weight > 0 else 0
    else:
        real_time_adherence_score = 0

    # Analyze window size usage (convert keys to strings for MongoDB compatibility)
    window_size_usage = {}
    for result in checkpoint_results:
        ws = str(result["optimal_window_size"])  # Convert to string for MongoDB
        window_size_usage[ws] = window_size_usage.get(ws, 0) + 1
    
    print(f"\n[ADAPTIVE] [FINAL] Real-time adherence score: {real_time_adherence_score}%")
    print(f"[ADAPTIVE] [FINAL] Script completion: {script_completion_percentage}%")
    print(f"[ADAPTIVE] [FINAL] Window size usage: {window_size_usage}")
    
    return {
        "real_time_adherence_score": real_time_adherence_score,
        "script_completion_percentage": script_completion_percentage,
        "checkpoint_results": checkpoint_results,
        "method": "adaptive_windows",
        "window_size_usage": window_size_usage,
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
    """Convert audio to text using Groq Whisper API"""
    try:
        audio_copy = BytesIO()
        audio.seek(0)
        audio_copy.write(audio.read())
        audio_copy.seek(0)
        
        buffer_data = audio_copy.read()
        audio_copy.seek(0)
        
        if len(buffer_data) < 44:
            return "", 0, []
        
        # Voice Activity Detection
        vad_result = detect_voice_activity(audio_copy, min_speech_duration=0.3, min_silence_duration=0.1)
        
        if not vad_result['has_speech']:
            return "", vad_result['total_audio_duration'], []
        
        # Groq Whisper API transcription
        audio_copy.name = "audio.wav"
        
        print("[API REQUEST] Calling Groq Whisper API for transcription")
        response = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_copy,
            language="en",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        
        print("[API RESPONSE] Groq Whisper transcription completed")
        
        if hasattr(response, 'text'):
            transcription = response.text or ""
        else:
            transcription = response.get('text', '') if isinstance(response, dict) else ""
        
        duration = vad_result['total_audio_duration']
        
        # Process word-level timestamps
        word_data = []
        words = None
        if hasattr(response, 'words') and response.words:
            words = response.words
        elif isinstance(response, dict) and 'words' in response and response['words']:
            words = response['words']
        
        if words:
            for word in words:
                if hasattr(word, 'word') and hasattr(word, 'start') and hasattr(word, 'end'):
                    word_data.append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'timestamp': word.start
                    })
                elif isinstance(word, dict):
                    word_data.append({
                        'word': word.get('word', word.get('text', '')),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0),
                        'timestamp': word.get('start', 0)
                    })
        
        return transcription, duration, word_data
        
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
                    timestamp_granularities=["word"]
                )
                
                if hasattr(response, 'text'):
                    transcription = response.text or ""
                else:
                    transcription = response.get('text', '') if isinstance(response, dict) else ""
                
                print("[API RETRY] Groq API retry successful")
                return transcription, 0, []
                
            except Exception as retry_error:
                print(f"[API ERROR] Groq API retry failed: {retry_error}")
                return "", 0, []
        else:
            print(f"[API ERROR] Groq API error: {error_message}")
            return "", 0, []

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
    transcript = request.form.get('transcript')  # Get transcript from form data
    
    # If no transcript provided, try to load from audioscript.txt
    if not transcript:
        print("[UPDATE_CALL] No transcript provided, attempting to load from audioscript.txt")
        transcript = load_script_from_file("audioscript.txt")
        if transcript:
            print(f"[UPDATE_CALL] Successfully loaded transcript from file ({len(transcript)} characters)")
        else:
            print("[UPDATE_CALL] No transcript file found, will use default script")
    else:
        print(f"[UPDATE_CALL] Transcript provided in request ({len(transcript)} characters)")

    try:
        # Process audio files
        agent_text, duration, agent_words = speech_to_text(agent_audio)
        client_text, duration, client_words = speech_to_text(client_audio)

        # Check if both audio files are silent
        if not agent_text and not client_text:
            return jsonify({
                "status": "success",
                "message": "Both audio files are silent - no transcription or analysis performed",
                "agent_text": "",
                "client_text": "",
                "duration": duration,
                "silent_audio": True
            }), 200
        
        # Ensure empty strings for silent audio
        if not agent_text:
            agent_text = ""
        if not client_text:
            client_text = ""

        # Get existing call data from MongoDB
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)
        
        if not existing_call:
            return jsonify({"error": "No record found for the given CallID for updating"}), 404
        
        # Prepare script sentences
        script_checkpoints = get_current_call_script(transcript)
        script_sentences = [checkpoint['prompt_text'] for checkpoint in script_checkpoints]

        # Store audio and transcription data in MongoDB
        client_audio.seek(0)
        agent_audio.seek(0)
        
        # Handle transcription concatenation with timestamps
        if existing_call.get('duration') is None:
            # First chunk case
            combined_transcription = f"Agent: {agent_text}\nClient: {client_text}".strip()
            total_agent_text = agent_text
            total_client_text = client_text
            total_agent_words = agent_words
            total_client_words = client_words
        else:
            # Subsequent chunks - append to existing transcription
            existing_transcription = existing_call.get('transcription', {})
            if isinstance(existing_transcription, dict):
                existing_combined = existing_transcription.get('combined', '')
                existing_agent = existing_transcription.get('agent', '')
                existing_client = existing_transcription.get('client', '')
                existing_agent_words = existing_transcription.get('agent_words', [])
                existing_client_words = existing_transcription.get('client_words', [])
                
                combined_transcription = (existing_combined + "\nAgent: " + agent_text + "\nClient: " + client_text).strip()
                total_agent_text = (existing_agent + " " + agent_text).strip()
                total_client_text = (existing_client + " " + client_text).strip()
                total_agent_words = existing_agent_words + agent_words
                total_client_words = existing_client_words + client_words
            else:
                combined_transcription = f"Agent: {agent_text}\nClient: {client_text}".strip()
                total_agent_text = agent_text
                total_client_text = client_text
                total_agent_words = agent_words
                total_client_words = client_words
            
            Adherence = existing_call.get('adherence', {})

        # Create sentences for the full call so far (once per speaker)
        print(f"[UPDATE_CALL] Creating agent sentences from {len(total_agent_words)} total words...")
        agent_sentences = create_adherence_optimized_sentences(total_agent_words, script_sentences)
        print(f"[UPDATE_CALL] Created {len(agent_sentences)} agent sentences")
        print(f"[UPDATE_CALL] Creating client sentences from {len(total_client_words)} total words...")
        client_sentences = create_adherence_optimized_sentences(total_client_words, [])
        print(f"[UPDATE_CALL] Created {len(client_sentences)} client sentences")

        # Prepare timestamped dialogue for display (merge agent/client sentences)
        agent_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Agent', 'text': s['text']} for s in agent_sentences]
        client_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Client', 'text': s['text']} for s in client_sentences]
        all_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])

        # Prepare complete transcription with timestamps
        all_transcription = {
            "agent": total_agent_text,
            "client": total_client_text,
            "combined": combined_transcription,
            "timestamped_dialogue": all_timestamped_dialogue,
            "agent_words": total_agent_words,
            "client_words": total_client_words
        }

        stored_audio_result = call_ops.store_audio_and_update_call(
            call_id=call_id,
            client_audio=client_audio,
            agent_audio=agent_audio,
            is_final=False
        )

        # Perform analysis using adaptive windows approach with custom script
        all_agent_sentences = [s['text'] for s in agent_sentences]
        results = check_script_adherence_adaptive_windows(total_agent_text, all_agent_sentences, script_checkpoints=script_checkpoints)
        overall_adherence = results.get("real_time_adherence_score", 0)
        script_completion = results.get("script_completion_percentage", 0)
        
        print(f"[UPDATE_CALL] Adaptive windows adherence analysis complete:")
        print(f"[UPDATE_CALL] - Real-time adherence score: {overall_adherence}%")
        print(f"[UPDATE_CALL] - Script completion: {script_completion}%")
        print(f"[UPDATE_CALL] - Window size usage: {results.get('window_size_usage', {})}")
        
        if isinstance(Adherence, dict):
            Adherence['overall'] = overall_adherence
            Adherence['script_completion'] = script_completion
            Adherence['details'] = results.get("checkpoint_results", [])
            Adherence['window_size_usage'] = results.get("window_size_usage", {})

        response = get_response(total_client_text)
        CQS, emotions = calculate_cqs(response["predictions"])
        quality = get_quality(emotions)

        # Update call with analysis data
        call_ops.insert_partial_update(call_id, duration, CQS, Adherence, emotions, all_transcription, quality)

        print(f"[UPDATE_CALL] Successfully updated call {call_id} with audio and data")

        return jsonify({
            "duration": duration,
            "status": "success",
            "agent_text": agent_text,
            "client_text": client_text,
            "call_id": call_id,
            "combined_transcription": combined_transcription,
            "overall_adherence": overall_adherence,
            "script_completion": script_completion,
            "adherence_details": results.get("checkpoint_results", []),
            "analysis_method": results.get("method", "adaptive_windows"),
            "window_size_usage": results.get("window_size_usage", {}),
            "script_info": {
                "script_provided": transcript is not None,
                "script_source": "request" if request.form.get('transcript') else "file" if transcript else "default",
                "total_checkpoints": results.get("total_checkpoints", len(results.get("checkpoint_results", [])))
            },
            "CQS": CQS,
            "emotions": emotions,
            "quality": quality,
            "stored_audio": stored_audio_result.get("stored_audio", {})
        })
    
    except Exception as e:
        print(f"[UPDATE_CALL ERROR] {e}")
        return jsonify({"error": "Database error occurred"}), 500

@app.route('/final_update', methods=['POST'])
def final_update():
    print("[ROUTE] POST /final_update")
    
    client_audio = request.files.get('client_audio')
    agent_audio = request.files.get('agent_audio')
    call_id = request.form.get('call_id')
    transcript = request.form.get('transcript')  # Get transcript from form data
    
    # If no transcript provided, try to load from audioscript.txt
    if not transcript:
        print("[FINAL_UPDATE] No transcript provided, attempting to load from audioscript.txt")
        transcript = load_script_from_file("audioscript.txt")
        if transcript:
            print(f"[FINAL_UPDATE] Successfully loaded transcript from file ({len(transcript)} characters)")
        else:
            print("[FINAL_UPDATE] No transcript file found, will use default script")
    else:
        print(f"[FINAL_UPDATE] Transcript provided in request ({len(transcript)} characters)")

    try:
        # Process audio files
        agent_text, duration, agent_words = speech_to_text(agent_audio)
        client_text, duration, client_words = speech_to_text(client_audio)

        # Get existing call data from MongoDB
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)

        if not existing_call:
            return jsonify({"error": "No record found for the given CallID for updating"}), 404

        # Prepare script sentences
        script_checkpoints = get_current_call_script(transcript)
        script_sentences = [checkpoint['prompt_text'] for checkpoint in script_checkpoints]

        # Store final audio and data
        client_audio.seek(0)
        agent_audio.seek(0)
        
        # Handle transcription concatenation with existing data and timestamps
        existing_transcription = existing_call.get('transcription', {})
        if existing_transcription and isinstance(existing_transcription, dict):
            existing_combined = existing_transcription.get('combined', '')
            existing_agent = existing_transcription.get('agent', '')
            existing_client = existing_transcription.get('client', '')
            existing_agent_words = existing_transcription.get('agent_words', [])
            existing_client_words = existing_transcription.get('client_words', [])
            
            combined_transcription = (existing_combined + "\nAgent: " + agent_text + "\nClient: " + client_text).strip()
            final_agent_text = (existing_agent + " " + agent_text).strip()
            final_client_text = (existing_client + " " + client_text).strip()
            final_agent_words = existing_agent_words + agent_words
            final_client_words = existing_client_words + client_words
        else:
            # No existing transcription or invalid format
            combined_transcription = f"Agent: {agent_text}\nClient: {client_text}".strip()
            final_agent_text = agent_text
            final_client_text = client_text
            final_agent_words = agent_words
            final_client_words = client_words

        # Create sentences for the full call so far (once per speaker)
        print(f"[FINAL_UPDATE] Creating agent sentences from {len(final_agent_words)} total words...")
        agent_sentences = create_adherence_optimized_sentences(final_agent_words, script_sentences)
        print(f"[FINAL_UPDATE] Created {len(agent_sentences)} agent sentences")
        print(f"[FINAL_UPDATE] Creating client sentences from {len(final_client_words)} total words...")
        client_sentences = create_adherence_optimized_sentences(final_client_words, [])
        print(f"[FINAL_UPDATE] Created {len(client_sentences)} client sentences")

        # Prepare timestamped dialogue for display (merge agent/client sentences)
        agent_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Agent', 'text': s['text']} for s in agent_sentences]
        client_dialogues = [{'timestamp': s['start_time'], 'speaker': 'Client', 'text': s['text']} for s in client_sentences]
        final_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])

        # Prepare final transcription data with timestamps
        final_transcription_data = {
            "agent": final_agent_text,
            "client": final_client_text,
            "combined": combined_transcription,
            "timestamped_dialogue": final_timestamped_dialogue,
            "agent_words": final_agent_words,
            "client_words": final_client_words
        }
        
        stored_audio_result = call_ops.store_audio_and_update_call(
            call_id=call_id,
            client_audio=client_audio,
            agent_audio=agent_audio,
            is_final=True
        )

        # Perform final analysis using adaptive windows approach with custom script
        all_agent_sentences = [s['text'] for s in agent_sentences]
        results = check_script_adherence_adaptive_windows(final_agent_text, all_agent_sentences, script_checkpoints=script_checkpoints)
        overall_adherence = results.get("real_time_adherence_score", 0)
        script_completion = results.get("script_completion_percentage", 0)
        adherence_details = results.get("checkpoint_results", [])
        
        print(f"[FINAL_UPDATE] Final adaptive windows adherence analysis complete:")
        print(f"[FINAL_UPDATE] - Real-time adherence score: {overall_adherence}%")
        print(f"[FINAL_UPDATE] - Script completion: {script_completion}%")
        print(f"[FINAL_UPDATE] - Total passed checkpoints: {len([r for r in adherence_details if r.get('status') == 'PASS'])}")
        print(f"[FINAL_UPDATE] - Window size usage: {results.get('window_size_usage', {})}")

        response = get_response(final_client_text)
        CQS, emotions = calculate_cqs(response["predictions"])
        quality = get_quality(emotions)

        agent_quality = agent_scores(final_agent_text)
        summary = call_summary(final_agent_text, final_client_text)

        text = f"Client:\n{final_client_text}\nAgent:\n{final_agent_text}"
        obj = get_tags(text)
        tags = ', '.join(obj['Tags'])

        # Calculate total duration
        total_duration = existing_call.get('duration', 0) + duration
        
        # Prepare comprehensive adherence data for final storage
        final_adherence_data = {
            "overall": overall_adherence,
            "script_completion": script_completion,
            "details": adherence_details,
            "window_size_usage": results.get("window_size_usage", {}),
            "method": results.get("method", "adaptive_windows")
        }
        
        # Complete the call update with timestamped dialogue
        call_ops.complete_call_update(
            call_id=call_id,
            agent_text=final_agent_text,
            client_text=final_client_text,
            combined=combined_transcription,
            cqs=CQS,
            overall_adherence=final_adherence_data,
            agent_quality=agent_quality,
            summary=summary,
            emotions=emotions,
            duration=total_duration,
            quality=quality,
            tags=tags,
            timestamped_dialogue=final_timestamped_dialogue,
            agent_words=final_agent_words,
            client_words=final_client_words
        )

        print(f"[FINAL_UPDATE] Successfully completed call {call_id} with final audio and data")

        return jsonify({
            "status": "success",
            "message": "Call analysis completed and data stored successfully",
            "call_id": call_id,
            "stored_audio": stored_audio_result.get("stored_audio", {}),
            "total_duration": total_duration,
            "final_adherence": {
                "real_time_score": overall_adherence,
                "script_completion": script_completion,
                "checkpoint_details": adherence_details,
                "analysis_method": results.get("method", "adaptive_windows"),
                "window_size_usage": results.get("window_size_usage", {}),
                "total_checkpoints": results.get("total_checkpoints", len(adherence_details))
            },
            "script_info": {
                "script_provided": transcript is not None,
                "script_source": "request" if request.form.get('transcript') else "file" if transcript else "default",
                "total_checkpoints": results.get("total_checkpoints", len(adherence_details))
            }
        }), 200
    
    except Exception as e:
        print(f"[FINAL_UPDATE ERROR] {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

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