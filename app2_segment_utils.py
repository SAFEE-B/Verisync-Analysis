"""
Utility functions extracted from app2-segment.py for use by Celery tasks
"""
import sys
import os

import os
from datetime import datetime, timezone
from rapidfuzz import fuzz
import requests
from dotenv import load_dotenv
from groq import Groq
import json
from io import BytesIO
import time
import nltk
from sentence_transformers import SentenceTransformer, util
import threading
from functools import lru_cache
import traceback
import soundfile as sf
import numpy as np
import torchaudio
import torch

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
emotion_model_url = os.getenv('EMOTION_MODEL')

# Initialize Groq client
try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    print(f"[ERROR] Failed to initialize Groq client: {e}")

# Initialize SBERT model
try:
    model_options = [
        'paraphrase-MiniLM-L3-v2',
        'all-MiniLM-L12-v2',
        'all-MiniLM-L6-v2'
    ]
    
    sbert_model = None
    for model_name in model_options:
        try:
            sbert_model = SentenceTransformer(model_name)
            # Force SBERT to run on CPU
            sbert_model = sbert_model.to('cpu')
            print(f"[INFO] SBERT model forced to run on CPU")
            break
        except Exception as model_error:
            continue
    
    SBERT_AVAILABLE = sbert_model is not None
    
except Exception as e:
    print(f"[ERROR] Failed to initialize Sentence Transformer: {e}")
    SBERT_AVAILABLE = False
    sbert_model = None

# Global cache and locks
_prompt_embedding_cache = {}
_cache_lock = threading.Lock()
emotions_CACHE = []

# Constants
SILENT_AUDIO_THRESHOLD = 1e-12

def speech_to_text(audio):
    """Convert audio to text using Groq Whisper API with segment-based processing, with retry logic."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            audio_copy = BytesIO()
            audio.seek(0)
            audio_copy.write(audio.read())
            audio_copy.seek(0)
            buffer_data = audio_copy.read()
            audio_copy.seek(0)
            if len(buffer_data) < 44:
                return "", 0, [], []
            audio_copy.name = "audio.wav"
            print(f"[API REQUEST] Calling Groq Whisper API for segment-based transcription. Attempt {attempt}.")
            response = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_copy,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"]
            )
            print("[API RESPONSE] Groq Whisper segment transcription completed")
            transcription = response.text or "" if hasattr(response, 'text') else response.get('text', '')
            duration = 0  # Simplified for now
            segment_data = []
            segments = response.segments if hasattr(response, 'segments') else response.get('segments', [])
            if segments:
                for segment in segments:
                    if isinstance(segment, dict):
                        no_speech_prob = segment.get('no_speech_prob', 0.0)
                        segment_info = {
                            'text': segment.get('text', '').strip(),
                            'start': segment.get('start', 0),
                            'end': segment.get('end', 0),
                            'confidence': segment.get('avg_logprob', 0.0),
                            'no_speech_prob': no_speech_prob
                        }
                    else:
                        no_speech_prob = getattr(segment, 'no_speech_prob', 0.0)
                        segment_info = {
                            'text': segment.text.strip(),
                            'start': segment.start,
                            'end': segment.end,
                            'confidence': getattr(segment, 'avg_logprob', 0.0),
                            'no_speech_prob': no_speech_prob
                        }
                    if no_speech_prob <= SILENT_AUDIO_THRESHOLD:
                        segment_data.append(segment_info)
            word_data = []
            words = response.words if hasattr(response, 'words') else response.get('words', [])
            if words:
                for word in words:
                    word_data.append({
                        'word': word.get('word', '').strip(),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0)
                    })
            return transcription, duration, segment_data, word_data
        except Exception as e:
            print(f"[API ERROR] Groq API error on attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"[RETRY] Retrying Whisper transcription (attempt {attempt + 1} of {max_retries})...")
                time.sleep(1)
            else:
                print("[API ERROR] All retry attempts failed for Groq Whisper API.")
                return "", 0, [], []

def resegment_based_on_punctuation(words):
    """Resegments text based on punctuation using word-level timestamps"""
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

    return new_segments

def merge_transcription_segments(previous_segments, new_segments, overlap_start_time):
    """Merges new transcription segments with a previous set by replacing the overlapping portion"""
    if not previous_segments:
        return new_segments

    base_segments = [seg for seg in previous_segments if seg.get('start', 0) < overlap_start_time]
    final_segments = base_segments + new_segments

    return final_segments

def process_whisper_segments(segments):
    """Process Whisper segments into sentence-like structures for analysis"""
    if not segments:
        return []
    
    processed_segments = []
    
    for i, segment in enumerate(segments):
        if hasattr(segment, 'text') and hasattr(segment, 'start') and hasattr(segment, 'end'):
            text = segment.text.strip()
            start_time = segment.start
            end_time = segment.end
        elif isinstance(segment, dict):
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
        else:
            continue
        
        if not text:
            continue
        
        word_count = len(text.split())
        duration = end_time - start_time
        
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
    
    return processed_segments

# Script handling functions
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
    """Parse script from custom XML-like format"""
    if not script_text or not script_text.strip():
        return []
    
    import re
    
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
            sentences = [prompt_text_clean]
        
        base_weights = {"STRICT": 25, "SEMANTIC": 20, "TOPIC": 15}
        weight = base_weights.get(adherence_type_upper, 15)
        
        is_mandatory = (
            adherence_type_upper == "STRICT" or 
            i == 0 or
            i == len(matches) - 1
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

def get_current_call_script(script_text=None):
    """Get the current call script for adherence checking"""
    if script_text:
        checkpoints = parse_script_from_text(script_text)
        if checkpoints:
            return checkpoints
    
    checkpoints = parse_script_from_text(DEFAULT_CALL_SCRIPT)
    
    if not checkpoints:
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

def _precompute_prompt_embeddings(script_text=None):
    """Pre-compute embeddings for default script prompts to improve performance"""
    try:
        print("[CACHE] Pre-computing default prompt embeddings...")
        script_checkpoints = get_current_call_script(script_text)  # This will use default or file-based script
        
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
                                    embedding = sbert_model.encode(prompt_version, convert_to_tensor=True, show_progress_bar=False)
                                    _prompt_embedding_cache[cache_key] = embedding
                                
        print(f"[CACHE] Cached {len(_prompt_embedding_cache)} default prompt embeddings")
        return script_checkpoints
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
                embedding = sbert_model.encode(clean_prompt, convert_to_tensor=True, show_progress_bar=False)
                _prompt_embedding_cache[cache_key] = embedding
                return embedding
        
        return None
        
    except Exception as e:
        print(f"[CACHE] Error getting cached embedding for {checkpoint_id}: {e}")
        return None
    

def clear_prompt_embedding_cache():
    """Clear the precomputed prompt embedding cache (thread-safe)."""
    with _cache_lock:
        _prompt_embedding_cache.clear()


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
    script_checkpoints=_precompute_prompt_embeddings(script_text)
    
    

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
                            window_embedding = sbert_model.encode(window_text, convert_to_tensor=True, show_progress_bar=False)
                            
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
    clear_prompt_embedding_cache()
    return {
        "real_time_adherence_score": real_time_adherence_score,
        "script_completion_percentage": script_completion_percentage,
        "checkpoint_results": checkpoint_results,
        "method": "adaptive_windows_concatenated",
        "total_checkpoints": len(checkpoint_results)
    }

def get_response(text):
    """Get emotion analysis from external API"""
    print(f"[API REQUEST] Calling emotion model API")
    
    url = "https://f9a470925488.ngrok-free.app/emotion"
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload, timeout=2.5)
        
        if response.status_code == 200:
            print("[API RESPONSE] Emotion model API response received")
            return response.json()
        else:
            print("Sentiment server error:", response.text)
    except Exception as e:
        print("Sentiment server not responding:", str(e))
    return None



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




def calculate_cqs(predictions, script_completion=0.0):
    """Calculate Call Quality Score (CQS) using emotions + script adherence"""
    print('Calculating call quality...')
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
        return [0.0, None]
    current_emotion = dict(predictions[0])
    
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
        total_confidence = 1
    
    emotion_cqs = round((total_weighted_score / total_confidence) * 100, 2)
    script_score = round(script_completion * 100, 2) if script_completion <= 1 else round(script_completion, 2)
    final_cqs = round((emotion_cqs * 0.6) + (script_score * 0.4), 2)
    
    return [final_cqs, current_emotion]

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

def build_conversation_from_segments(agent_segments, client_segments):
    """Build conversation array from separate agent and client segments"""
    from db_config import build_conversation_from_segments as db_build_conversation
    return db_build_conversation(agent_segments, client_segments) 

def clear_prompt_embedding_cache():
    """Clear the precomputed prompt embedding cache (thread-safe)."""
    with _cache_lock:
        _prompt_embedding_cache.clear() 