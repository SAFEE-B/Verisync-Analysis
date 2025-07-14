import os
from datetime import datetime
from rapidfuzz import fuzz
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from groq import Groq
import json
from flask_cors import CORS  # Add this import
from io import BytesIO
import time
import nltk
from sentence_transformers import SentenceTransformer, util
import threading
from functools import lru_cache

# =============================================================================
# VERISYNC ANALYSIS BACKEND - CONSOLIDATED ADAPTIVE WINDOWS APPROACH
# =============================================================================
# 
# This application uses the Adaptive Windows method for script adherence analysis,
# which automatically tests window sizes 1, 2, and 3 for each checkpoint and 
# selects the optimal window size for the best accuracy.
#
# Key Features:
# - Adaptive Windows: Tests multiple window sizes automatically
# - SBERT Optimization: Cached embeddings + GPU support + batching
# - Real-time Analysis: Processes audio chunks during calls
# - Comprehensive Metrics: CQS, emotions, quality scores, adherence
#
# Main Endpoints:
# - POST /update_call: Process audio chunks with adaptive analysis
# - POST /final_update: Complete call analysis with final metrics
# - POST /compare_methods: Compare standard vs adaptive methods
# - GET /performance_stats: Monitor SBERT performance
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
    # Load Silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    
    # Extract utility functions
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    VAD_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] Failed to load Silero VAD model: {e}")
    VAD_AVAILABLE = False
    model = None

# Initialize Sentence Transformer model with optimizations
try:
    print("[INFO] Attempting to load Sentence Transformer model...")
    
    # Try smaller models first in case of network issues
    model_options = [
        'all-MiniLM-L6-v2',  # Original choice - good quality
        'paraphrase-MiniLM-L3-v2',  # Smaller alternative
        'all-MiniLM-L12-v2'  # Larger alternative
    ]
    
    sbert_model = None
    for model_name in model_options:
        try:
            print(f"[INFO] Trying to load model: {model_name}")
            sbert_model = SentenceTransformer(model_name)
            
            # Enable GPU if available
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



def get_cached_prompt_embedding(checkpoint_id, prompt_text, preserve_structure=True):
    """Get cached prompt embedding or compute and cache if not available"""
    try:
        clean_prompt = clean_text_for_matching(prompt_text, preserve_structure=preserve_structure)
        version_key = "preserve" if preserve_structure else "remove"
        cache_key = f"{checkpoint_id}_{version_key}"
        
        with _cache_lock:
            if cache_key in _prompt_embedding_cache:
                return _prompt_embedding_cache[cache_key]
            
            # Compute and cache if not available
            if clean_prompt.strip() and SBERT_AVAILABLE and sbert_model is not None:
                embedding = sbert_model.encode(clean_prompt, convert_to_tensor=True)
                _prompt_embedding_cache[cache_key] = embedding
                return embedding
        
        return None
        
    except Exception as e:
        print(f"[CACHE] Error getting cached embedding for {checkpoint_id}: {e}")
        return None

@lru_cache(maxsize=128)
def get_sentence_embeddings_cached(text_hash):
    """Cache sentence embeddings for repeated text analysis"""
    try:
        if SBERT_AVAILABLE and sbert_model is not None:
            # This function will be called with hashed text
            # The actual encoding happens in the calling function
            return True
        return False
    except Exception as e:
        print(f"[CACHE] Error in sentence embedding cache: {e}")
        return False

def encode_sentences_with_memory_optimization(sentences, batch_size=32):
    """
    Encode sentences in batches to manage memory usage during concurrent requests
    """
    try:
        if not SBERT_AVAILABLE or sbert_model is None:
            return None
            
        if len(sentences) <= batch_size:
            # Small batch, encode directly
            return sbert_model.encode(sentences, convert_to_tensor=True)
        
        # Large batch, process in chunks to manage memory
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = sbert_model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
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

# Initialize prompt embeddings cache after functions are defined
if SBERT_AVAILABLE and sbert_model is not None:
    try:
        _precompute_prompt_embeddings()
    except Exception as e:
        print(f"[ERROR] Failed to precompute prompt embeddings: {e}")

# =============================================================================
# CALL SCRIPT CONFIGURATION
# =============================================================================

# Fallback call script - can be updated later via frontend
DEFAULT_CALL_SCRIPT = [
    {
        "checkpoint_id": "greeting_and_company_intro",
        "prompt_text": "Hello, thank you for calling [Company Name].",
        "adherence_type": "SEMANTIC",
        "is_mandatory": True,
        "weight": 15
    },
    {
        "checkpoint_id": "agent_self_introduction",
        "prompt_text": "My name is [Agent Name], how can I assist you today?",
        "adherence_type": "SEMANTIC",
        "is_mandatory": True,
        "weight": 15
    },
    {
        "checkpoint_id": "commitment_to_help",
        "prompt_text": "I understand your concern. Let me help you with that.",
        "adherence_type": "TOPIC",
        "is_mandatory": False,
        "weight": 20
    },
    {
        "checkpoint_id": "account_verification",
        "prompt_text": "Can I please get your account number or phone number to pull up your account?",
        "adherence_type": "SEMANTIC",
        "is_mandatory": True,
        "weight": 25
    },
    {
        "checkpoint_id": "offer_additional_assistance",
        "prompt_text": "Is there anything else I can help you with today?",
        "adherence_type": "STRICT",
        "is_mandatory": False,
        "weight": 10
    },
    {
        "checkpoint_id": "closing",
        "prompt_text": "Thank you for calling [Company Name]. Have a great day!",
        "adherence_type": "SEMANTIC",
        "is_mandatory": True,
        "weight": 15
    }
]

def get_current_call_script():
    """
    Get the current call script for adherence checking.
    For now, returns the fallback script. Later this can be modified to:
    - Load from database
    - Load from frontend configuration
    - Load from environment variables
    """
    return DEFAULT_CALL_SCRIPT

def fallback_similarity(text1, text2):
    """
    Fallback similarity function using rapidfuzz when SBERT is not available.
    Returns a score between 0 and 1.
    """
    return fuzz.ratio(text1.lower(), text2.lower()) / 100.0

def clean_text_for_matching(text, preserve_structure=False):
    """
    Clean text for better semantic matching.
    If preserve_structure=True, replace placeholders with generic terms instead of removing them.
    """
    if preserve_structure:
        # Replace placeholders with generic terms to preserve sentence structure
        cleaned = text.replace('[Company Name]', 'company')
        cleaned = cleaned.replace('[Agent Name]', 'agent')
        cleaned = cleaned.replace('[Customer Name]', 'customer')
        cleaned = cleaned.replace('[', '').replace(']', '')  # Remove any remaining brackets
    else:
        # Original cleaning method - remove placeholders entirely
        cleaned = text.replace('[Company Name]', '').replace('[Agent Name]', '')
        cleaned = cleaned.replace('[Customer Name]', '').replace('[', '').replace(']', '')
    
    # Clean up extra spaces and normalize
    cleaned = ' '.join(cleaned.split()).lower().strip()
    return cleaned

def fuzzy_semantic_match(prompt, sentences, threshold=60):
    """
    Enhanced semantic matching that tries multiple approaches.
    Returns the best match details.
    """
    best_score = 0
    best_sentence = ""
    best_index = 0
    match_method = "none"
    
    # Method 1: Direct sentence matching
    for i, sentence in enumerate(sentences):
        # Try both cleaning methods
        clean_sent_preserve = clean_text_for_matching(sentence, preserve_structure=True)
        clean_sent_remove = clean_text_for_matching(sentence, preserve_structure=False)
        
        # Use rapidfuzz for fallback comparison
        score1 = fuzz.ratio(prompt, clean_sent_preserve) 
        score2 = fuzz.ratio(prompt, clean_sent_remove)
        score3 = fuzz.partial_ratio(prompt, clean_sent_preserve)
        score4 = fuzz.partial_ratio(prompt, clean_sent_remove)
        
        max_score = max(score1, score2, score3, score4)
        
        if max_score > best_score:
            best_score = max_score
            best_sentence = sentence
            best_index = i
            match_method = f"fuzzy_{'preserve' if max_score in [score1, score3] else 'remove'}"
    
    return {
        "score": best_score,
        "sentence": best_sentence,
        "index": best_index,
        "method": match_method,
        "passed": best_score >= threshold
    }



def check_script_adherence(agent_text, script_checkpoints=None):
    """
    Check how well the agent followed the call script using a dynamic checkpoint-based approach.
    This function implements the Dynamic Real-Time Adherence Model.
    """
    if script_checkpoints is None:
        script_checkpoints = get_current_call_script()
    
    if not script_checkpoints or not agent_text:
        return {
            "real_time_adherence_score": 0,
            "script_completion_percentage": 0,
            "checkpoint_results": [],
            "debug_info": {
                "sbert_available": SBERT_AVAILABLE,
                "checkpoints_count": len(script_checkpoints) if script_checkpoints else 0,
                "agent_text_length": len(agent_text) if agent_text else 0,
                "error": "Missing required data"
            }
        }
    
    # Clean agent text by removing placeholders for better matching
    clean_agent_text = agent_text.lower()
    
    checkpoint_results = []
    
    # Split agent's transcript into sentences for semantic analysis
    try:
        agent_sentences = nltk.sent_tokenize(clean_agent_text)
    except Exception as e:
        agent_sentences = clean_agent_text.split('.')
    
    # Phase A: Initial Evaluation (evaluate all checkpoints against current transcript)
    for i, checkpoint in enumerate(script_checkpoints):
        checkpoint_id = checkpoint.get("checkpoint_id", f"unknown_{i}")
        prompt_text = checkpoint.get("prompt_text", "")
        adherence_type = checkpoint.get("adherence_type", "SEMANTIC")
        weight = checkpoint.get("weight", 1)
        is_mandatory = checkpoint.get("is_mandatory", False)

        # Try multiple cleaning approaches for better matching
        clean_prompt_preserve = clean_text_for_matching(prompt_text, preserve_structure=True)
        clean_prompt_remove = clean_text_for_matching(prompt_text, preserve_structure=False)

        score = 0
        status = "FAIL"
        match_details = "Not found"

        if adherence_type == "STRICT":
            # First try exact matching
            if clean_prompt_preserve in clean_agent_text or clean_prompt_remove in clean_agent_text:
                score = 100
                status = "PASS"
                match_details = "Exact phrase found"
            else:
                # Use fuzzy matching for STRICT with high threshold (allows minor variations)
                strict_threshold = 85  # High threshold for strict matching
                fuzzy_result_preserve = fuzzy_semantic_match(clean_prompt_preserve, agent_sentences, strict_threshold)
                fuzzy_result_remove = fuzzy_semantic_match(clean_prompt_remove, agent_sentences, strict_threshold)
                
                best_fuzzy_score = max(fuzzy_result_preserve["score"], fuzzy_result_remove["score"])
                
                if best_fuzzy_score >= strict_threshold:
                    score = best_fuzzy_score
                    status = "PASS"
                    match_details = f"Fuzzy match found ({best_fuzzy_score}%)"
                else:
                    score = best_fuzzy_score
                    match_details = f"No sufficient match found (best: {best_fuzzy_score}%)"

        elif adherence_type == "SEMANTIC":
            # Use SBERT if available, otherwise fallback to fuzzy
            if SBERT_AVAILABLE:
                try:
                    # Get embeddings
                    prompt_embedding = sbert_model.encode([prompt_text])[0]
                    agent_embeddings = sbert_model.encode(agent_sentences)
                    
                    # Calculate cosine similarity with each sentence
                    similarities = cosine_similarity([prompt_embedding], agent_embeddings)[0]
                    best_score = max(similarities) * 100
                    
                    if best_score >= 60:  # Semantic threshold
                        score = best_score
                        status = "PASS"
                        match_details = f"SBERT semantic match: {score}%"
                    else:
                        score = best_score
                        match_details = f"Low semantic similarity: {score}%"
                except Exception as e:
                    score = 0
                    match_details = f"SBERT error: {e}"
            else:
                # Fallback to fuzzy matching
                semantic_threshold = 70
                fuzzy_result = fuzzy_semantic_match(clean_prompt_preserve, agent_sentences, semantic_threshold)
                score = fuzzy_result["score"]
                if score >= semantic_threshold:
                    status = "PASS"
                    match_details = f"Fuzzy semantic match: {score}%"
                else:
                    match_details = f"Low fuzzy match: {score}%"

        elif adherence_type == "TOPIC":
            # Topic matching using SBERT or fallback
            topic_threshold = 40  # Lower threshold for topic matching
            if SBERT_AVAILABLE:
                try:
                    # Get embeddings
                    prompt_embedding = sbert_model.encode([prompt_text])[0]
                    agent_embeddings = sbert_model.encode(agent_sentences)
                    
                    # Calculate cosine similarity with each sentence
                    similarities = cosine_similarity([prompt_embedding], agent_embeddings)[0]
                    best_score = max(similarities) * 100
                    
                    if best_score >= topic_threshold:
                        status = "PASS"
                        match_details = f"SBERT topic similarity: {score}%"
                    else:
                        match_details = f"Low topic similarity: {score}%"
                        
                except Exception as e:
                    score = 0
                    match_details = f"SBERT topic error: {e}"
            else:
                # Simple fallback if SBERT not available
                if clean_prompt_preserve in clean_agent_text or clean_prompt_remove in clean_agent_text:
                    score = 60  # Moderate confidence for topic fallback
                    status = "PASS"
                    match_details = "Topic keywords found (SBERT fallback)"
                else:
                    score = 0
                    match_details = "No topic match found (SBERT fallback)"

        checkpoint_results.append({
            "checkpoint_id": checkpoint_id,
            "adherence_type": adherence_type,
            "status": status,
            "score": score,
            "weight": weight,
            "is_mandatory": is_mandatory,
            "match_details": match_details
        })

    # Phase B: Apply the "Highest Achieved Checkpoint" Heuristic
    last_passed_index = -1
    passed_checkpoints = 0
    
    # Find the last checkpoint that passed (scanning from beginning to end)
    for i, result in enumerate(checkpoint_results):
        if result["status"] == "PASS":
            last_passed_index = i
            passed_checkpoints += 1
    
    # Calculate Script Completion Percentage
    script_completion_percentage = round((passed_checkpoints / len(checkpoint_results)) * 100, 2) if checkpoint_results else 0
    
    # Calculate Real-Time Adherence Score (only consider checkpoints up to last passed)
    if last_passed_index >= 0:
        # Consider checkpoints from 0 to last_passed_index (inclusive)
        relevant_checkpoints = checkpoint_results[:last_passed_index + 1]
        
        total_weight = 0
        weighted_score_sum = 0
        
        for result in relevant_checkpoints:
            weight = result["weight"]
            score = result["score"]
            total_weight += weight
            weighted_score_sum += score * weight
        
        real_time_adherence_score = round(weighted_score_sum / total_weight, 2) if total_weight > 0 else 0
    else:
        # No checkpoints passed yet
        real_time_adherence_score = 0
    
    return {
        "real_time_adherence_score": real_time_adherence_score,
        "script_completion_percentage": script_completion_percentage,
        "checkpoint_results": checkpoint_results,
        "debug_info": {
            "total_checkpoints": len(checkpoint_results),
            "passed_checkpoints": passed_checkpoints,
            "last_passed_index": last_passed_index,
            "agent_sentences_count": len(agent_sentences),
            "sbert_available": SBERT_AVAILABLE,
            "script_used": "dynamic_v2_enhanced"
        }
    }

# =============================================================================
# END CALL SCRIPT CONFIGURATION
# =============================================================================

def get_quality(emotions):
    # Define emotions as positive or negative contributors
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

def speech_to_text(audio):
    try:
        # Create a temporary copy of the audio to ensure we're working with fresh data
        audio_copy = BytesIO()
        audio.seek(0)  # Reset position
        audio_copy.write(audio.read())  # Copy all data
        audio_copy.seek(0)  # Reset position of copy
        
        # Print some diagnostics
        buffer_data = audio_copy.read()
        audio_copy.seek(0)  # Reset again after reading
        
        # Verify audio file (basic check)
        if len(buffer_data) < 44:
            return "", 0, []
        
        # STEP 1: Voice Activity Detection
        vad_result = detect_voice_activity(audio_copy, min_speech_duration=0.3, min_silence_duration=0.1)
        
        # Handle silent audio
        if not vad_result['has_speech']:
            return "", vad_result['total_audio_duration'], []
        
        # STEP 2: Proceed with Groq transcription for audio with speech
        # Prepare the audio file for Groq Whisper API
        # The API expects a file-like object with a name attribute
        audio_copy.name = "audio.wav"  # Add name attribute required by Groq API
        
        # Call Groq Whisper API with timestamp_granularities for word-level timestamps
        print("[API REQUEST] Calling Groq Whisper API for transcription")
        response = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",  # Groq's fastest Whisper model
            file=audio_copy,
            language="en",  # Force English-only transcription
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        
        print("[API RESPONSE] Groq Whisper transcription completed")
        
        # Extract transcription text - handle both object and dict formats
        if hasattr(response, 'text'):
            transcription = response.text or ""
        else:
            transcription = response.get('text', '') if isinstance(response, dict) else ""
        
        # Use VAD duration if available, otherwise extract from Groq response
        duration = vad_result['total_audio_duration']
        line = []
        
        # Try to get words from response - Groq may not support word-level timestamps
        words = None
        if hasattr(response, 'words') and response.words:
            words = response.words
        elif isinstance(response, dict) and 'words' in response and response['words']:
            words = response['words']
        
        if words and transcription:  # Only process words if we have transcription
            try:
                # Group words into sentences/segments
                current_segment = ""
                current_start = 0
                sentence_break_chars = ['.', '!', '?']
                
                for i, word in enumerate(words):
                    if i == 0:
                        if isinstance(word, dict):
                            current_start = word.get('start', 0)
                        else:
                            current_start = getattr(word, 'start', 0)
                    
                    # Get word text - handle both dict and object formats
                    if isinstance(word, dict):
                        word_text = word.get('word', word.get('text', ''))
                    else:
                        word_text = getattr(word, 'word', getattr(word, 'text', ''))
                    
                    current_segment += word_text + " "
                    
                    # Check if this word ends a sentence or if we've reached a reasonable segment length
                    if (any(char in word_text for char in sentence_break_chars) or 
                        len(current_segment.split()) >= 15 or 
                        i == len(words) - 1):
                        
                        segment_text = current_segment.strip()
                        line.append([current_start, segment_text])
                        current_segment = ""
                        # Set next segment start time
                        if i + 1 < len(words):
                            next_word = words[i + 1]
                            if isinstance(next_word, dict):
                                current_start = next_word.get('start', current_start)
                            else:
                                current_start = getattr(next_word, 'start', current_start)
                                
            except Exception as word_error:
                # Fall back to single segment
                line = [[0, transcription]] if transcription else []
        
        # If no word-level timestamps or no transcription, create appropriate segments
        if not line:
            if transcription:
                line.append([0, transcription])
        
        return transcription, duration, line
        
    except Exception as e:
        # Handle Groq API errors - Groq uses similar error structure to OpenAI
        error_message = str(e)
        
        if "rate_limit" in error_message.lower() or "429" in error_message:
            print(f"[API ERROR] Groq API Rate Limit: {error_message}")
            # Wait and retry once for rate limiting
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
                
                # Extract transcription text - handle both object and dict formats
                if hasattr(response, 'text'):
                    transcription = response.text or ""
                else:
                    transcription = response.get('text', '') if isinstance(response, dict) else ""
                
                print("[API RETRY] Groq API retry successful")
                
                # Return basic result after retry
                return transcription, 0, [[0, transcription]] if transcription else []
                
            except Exception as retry_error:
                print(f"[API ERROR] Groq API retry failed: {retry_error}")
                return "", 0, []
        else:
            print(f"[API ERROR] Groq API error: {error_message}")
            return "", 0, []

def get_response(text):
    print(f"[API REQUEST] Calling emotion model API")
    
    url = os.getenv('EMOTION_MODEL')
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "input": text
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"[API RESPONSE] Emotion API call successful")
            return result
        else:
            print(f"[API ERROR] Emotion API returned status {response.status_code}")
            # Return neutral response on API error
            return {
                "predictions": [
                    {"label": "neutral", "score": 1.0}
                ]
            }
            
    except Exception as e:
        print(f"[API ERROR] Exception calling emotion API: {e}")
        # Return neutral response on exception
        return {
            "predictions": [
                {"label": "neutral", "score": 1.0}
            ]
        }

def calculate_cqs(predictions):
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
        # else: skip unknown emotions, no print
    final_cqs = round(cqs, 2)
    result = [final_cqs, emotions]
    return result

def agent_scores(transcription):
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
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            response_format={"type": "json_object"}
        )

        # Ensure the response is valid JSON
        response_content = completion.choices[0].message.content
        result = json.loads(response_content)
        print("[API RESPONSE] Groq AI agent evaluation completed")
        
        return result
            
    except Exception as e:
        print(f"[API ERROR] Exception calling Groq AI: {e}")
        # Return default N/A scores on exception
        result = {}
        for i in range(1, 10):
            result[f'q{i}'] = {"yes": 0, "no": 0, "na": 100}
        return result

def call_summary(agent_text, client_text):
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
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
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

# Removed db_insertion function - now using call_ops.create_call_entry and call_ops.complete_call_update directly

def get_tags(conversation):
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
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            response_format={"type": "json_object"}
        )

        # Ensure the response is valid JSON
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

# Removed insert_realtime function - now using call_ops.insert_partial_update directly

# Removed notify_main_backend function - now handled by call_ops._notify_main_backend

def detect_voice_activity(audio_file, min_speech_duration=0.5, min_silence_duration=0.1):
    """
    Detect voice activity in audio using Silero VAD
    
    Args:
        audio_file: Audio file (BytesIO object or file path)
        min_speech_duration: Minimum duration of speech to consider as voice activity (seconds)
        min_silence_duration: Minimum silence duration between speech segments (seconds)
    
    Returns:
        dict: {
            'has_speech': bool,
            'speech_segments': list of [start, end] timestamps,
            'total_speech_duration': float,
            'total_audio_duration': float,
            'speech_ratio': float (0-1)
        }
    """
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
        # Reset file pointer if it's a BytesIO object
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)
            
        # Read audio data
        if isinstance(audio_file, BytesIO):
            # Save BytesIO to temporary file for torchaudio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio_file.seek(0)
                temp_file.write(audio_file.read())
                temp_path = temp_file.name
                audio_file.seek(0)  # Reset for later use
            
            # Load audio using torchaudio
            wav, sr = torchaudio.load(temp_path)
            
            # Clean up temp file
            import os
            os.unlink(temp_path)
        else:
            # Direct file path
            wav, sr = torchaudio.load(audio_file)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (Silero VAD expects 16kHz)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000
        
        # Squeeze to 1D tensor
        wav = wav.squeeze()
        
        # Calculate total audio duration
        total_duration = len(wav) / sr
        
        # Get speech timestamps using Silero VAD
        speech_timestamps = get_speech_timestamps(
            wav, 
            model,
            sampling_rate=sr,
            min_speech_duration_ms=int(min_speech_duration * 1000),
            min_silence_duration_ms=int(min_silence_duration * 1000),
            window_size_samples=512,
            speech_pad_ms=30
        )
        
        # Calculate speech statistics
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
        # Return conservative result (assume speech) on error
        return {
            'has_speech': True,
            'speech_segments': [],
            'total_speech_duration': 0,
            'total_audio_duration': 0,
            'speech_ratio': 1.0,
            'vad_available': True,
            'error': str(e)
        }

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

load_dotenv()

# Load and validate environment variables
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


@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Analysis backend is running!"}), 200

@app.route('/performance_stats', methods=['GET'])
def get_performance_stats():
    """Get current SBERT performance statistics"""
    print("[ROUTE] GET /performance_stats")
    try:
        stats = {
            "sbert_available": SBERT_AVAILABLE,
            "gpu_available": torch.cuda.is_available() if SBERT_AVAILABLE else False,
            "cached_prompt_embeddings": len(_prompt_embedding_cache),
            "model_info": {
                "device": str(sbert_model.device) if SBERT_AVAILABLE and sbert_model else "N/A",
                "max_seq_length": sbert_model.max_seq_length if SBERT_AVAILABLE and sbert_model else "N/A"
            },
            "memory_info": {
                "torch_memory_allocated": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                "torch_memory_cached": torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
            },
            "recommendations": []
        }
        
        # Add performance recommendations
        if not torch.cuda.is_available():
            stats["recommendations"].append("Consider enabling GPU acceleration for 3-5x performance improvement")
        
        if len(_prompt_embedding_cache) == 0:
            stats["recommendations"].append("Prompt embeddings cache is empty - restart server to initialize")
            
        if len(_prompt_embedding_cache) < 10:
            stats["recommendations"].append("Consider pre-caching more script variations")
            
        return jsonify({
            "status": "success",
            "performance_stats": stats
        }), 200
        
    except Exception as e:
        print(f"[ROUTE ERROR] /performance_stats: {e}")
        return jsonify({"error": "Failed to get performance stats"}), 500

@app.route('/sbert_benchmark', methods=['POST'])
def benchmark_sbert():
    """Benchmark SBERT encoding performance"""
    print("[ROUTE] POST /sbert_benchmark")
    try:
        data = request.get_json()
        test_sentences = data.get('test_sentences', [
            "Hello, thank you for calling our company.",
            "My name is John, how can I assist you today?", 
            "I understand your concern. Let me help you with that.",
            "Can I please get your account number?",
            "Is there anything else I can help you with?",
            "Thank you for calling. Have a great day!"
        ])
        iterations = data.get('iterations', 5)
        
        if not SBERT_AVAILABLE or sbert_model is None:
            return jsonify({
                "error": "SBERT not available",
                "status": "failed"
            }), 400
        
        # Benchmark standard encoding
        import time
        start_time = time.time()
        for _ in range(iterations):
            embeddings = sbert_model.encode(test_sentences, convert_to_tensor=True)
        standard_time = (time.time() - start_time) / iterations
        
        # Benchmark optimized encoding
        start_time = time.time()
        for _ in range(iterations):
            embeddings = encode_sentences_with_memory_optimization(test_sentences, batch_size=32)
        optimized_time = (time.time() - start_time) / iterations
        
        # Calculate throughput
        sentences_per_second_standard = len(test_sentences) / standard_time
        sentences_per_second_optimized = len(test_sentences) / optimized_time
        
        return jsonify({
            "status": "success",
            "benchmark_results": {
                "test_sentences_count": len(test_sentences),
                "iterations": iterations,
                "standard_encoding": {
                    "avg_time_seconds": round(standard_time, 4),
                    "sentences_per_second": round(sentences_per_second_standard, 2)
                },
                "optimized_encoding": {
                    "avg_time_seconds": round(optimized_time, 4),
                    "sentences_per_second": round(sentences_per_second_optimized, 2)
                },
                "performance_improvement": f"{round((standard_time / optimized_time), 2)}x faster" if optimized_time > 0 else "N/A",
                "device": str(sbert_model.device),
                "gpu_available": torch.cuda.is_available()
            }
        }), 200
        
    except Exception as e:
        print(f"[ROUTE ERROR] /sbert_benchmark: {e}")
        return jsonify({"error": str(e)}), 500


# Removed insert_partial function - now using call_ops.insert_partial_update directly


@app.route('/update_call', methods=['POST'])
def update_call():
    print("[ROUTE] POST /update_call")
    
    client_audio = request.files.get('client_audio')
    agent_audio = request.files.get('agent_audio')
    call_id = request.form.get('call_id')
    transcript_file = request.files.get('transcript')

    # Read the transcript file and create a list of lines
    if transcript_file:
        transcript_lines = transcript_file.read().decode('utf-8').splitlines()
    else:
        transcript_lines = []

    try:
        # Process audio files
        agent_text, duration, agent_dialogues = speech_to_text(agent_audio)
        client_text, duration, client_dialogues = speech_to_text(client_audio)

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
        
        # Prepare current chunk data with timestamps preserved
        combined_with_timestamps = []
        combined_with_timestamps += [[time, "Agent", text] for time, text in agent_dialogues]
        combined_with_timestamps += [[time, "Client", text] for time, text in client_dialogues]
        combined_with_timestamps.sort(key=lambda x: x[0])

        # Create timestamped dialogue structure for database storage
        current_timestamped_dialogue = [
            {"timestamp": time, "speaker": speaker, "text": text} 
            for time, speaker, text in combined_with_timestamps
        ]
        
        # Also create traditional combined transcription for backward compatibility
        final_dialogue = [f"{speaker}: {text}" for _, speaker, text in combined_with_timestamps]
        current_combined_transcription = "\n".join(final_dialogue)

        # Store audio and transcription data in MongoDB
        client_audio.seek(0)
        agent_audio.seek(0)
        
        # Handle transcription concatenation with timestamps
        if existing_call.get('duration') is None:
            # First chunk case
            combined_transcription = current_combined_transcription
            total_agent_text = agent_text
            total_client_text = client_text
            all_timestamped_dialogue = current_timestamped_dialogue
            
            # Initialize adherence structure
            Adherence = {}
            for i in range(1, 10):
                Adherence[f'q{i}'] = {"yes": 0, "no": 0, "na": 0}
        else:
            # Subsequent chunks - append to existing transcription
            existing_transcription = existing_call.get('transcription', {})
            if isinstance(existing_transcription, dict):
                existing_combined = existing_transcription.get('combined', '')
                existing_agent = existing_transcription.get('agent', '')
                existing_client = existing_transcription.get('client', '')
                existing_timestamped = existing_transcription.get('timestamped_dialogue', [])
                
                combined_transcription = (existing_combined + "\n" + current_combined_transcription).strip()
                total_agent_text = (existing_agent + " " + agent_text).strip()
                total_client_text = (existing_client + " " + client_text).strip()
                all_timestamped_dialogue = existing_timestamped + current_timestamped_dialogue
            else:
                combined_transcription = current_combined_transcription
                total_agent_text = agent_text
                total_client_text = client_text
                all_timestamped_dialogue = current_timestamped_dialogue
            
            Adherence = existing_call.get('adherence', {})

        # Prepare complete transcription with timestamps
        all_transcription = {
            "agent": total_agent_text,
            "client": total_client_text,
            "combined": combined_transcription,
            "timestamped_dialogue": all_timestamped_dialogue
        }

        stored_audio_result = call_ops.store_audio_and_update_call(
            call_id=call_id,
            client_audio=client_audio,
            agent_audio=agent_audio,
            transcription_data=all_transcription,
            analysis_data={"duration": duration},
            is_final=False
        )

        # Perform analysis using adaptive windows approach (tests window sizes 1, 2, 3)
        # Use uploaded transcript_lines if available, otherwise use default script
        script_to_use = transcript_lines if transcript_lines else None
        results = check_script_adherence_adaptive_windows(total_agent_text, script_to_use)
        overall_adherence = results.get("real_time_adherence_score", 0)
        script_completion = results.get("script_completion_percentage", 0)
        
        print(f"[UPDATE_CALL] Adaptive windows adherence analysis complete:")
        print(f"[UPDATE_CALL] - Real-time adherence score: {overall_adherence}%")
        print(f"[UPDATE_CALL] - Script completion: {script_completion}%")
        print(f"[UPDATE_CALL] - Window size usage: {results.get('window_size_usage', {})}")
        print(f"[UPDATE_CALL] - Method: {results.get('method', 'adaptive_windows')}")
        
        if isinstance(Adherence, dict):
            Adherence['overall'] = overall_adherence
            Adherence['script_completion'] = script_completion
            Adherence['details'] = results.get("checkpoint_results", [])
            Adherence['debug_info'] = results.get("debug_info", {})

        response = get_response(total_client_text)
        CQS, emotions = calculate_cqs(response["predictions"])
        quality = get_quality(emotions)

        # Update call with analysis data (audio already stored above)
        call_ops.insert_partial_update(call_id, duration, CQS, Adherence, emotions, all_transcription, quality)

        print(f"[UPDATE_CALL] Successfully updated call {call_id} with audio and data")

        return jsonify({
            "duration": duration,
            "status": "success",
            "agent_text": agent_text,
            "client_text": client_text,
            "transcript_lines": transcript_lines,
            "call_id": call_id,
            "combined_transcription": current_combined_transcription,
            "overall_adherence": overall_adherence,
            "script_completion": script_completion,
            "adherence_details": results.get("checkpoint_results", []),
            "analysis_method": results.get("method", "adaptive_windows"),
            "window_size_usage": results.get("window_size_usage", {}),
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
    transcript_file = request.files.get('transcript')

    # Read the transcript file and create a list of lines
    if transcript_file:
        transcript_lines = transcript_file.read().decode('utf-8').splitlines()
    else:
        transcript_lines = []

    try:
        # Process audio files
        agent_text, duration, agent_dialogues = speech_to_text(agent_audio)
        client_text, duration, client_dialogues = speech_to_text(client_audio)

        # Get existing call data from MongoDB
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)

        if not existing_call:
            return jsonify({"error": "No record found for the given CallID for updating"}), 404

        # Prepare current chunk data with timestamps preserved
        combined_with_timestamps = []
        combined_with_timestamps += [[time, "Agent", text] for time, text in agent_dialogues]
        combined_with_timestamps += [[time, "Client", text] for time, text in client_dialogues]
        combined_with_timestamps.sort(key=lambda x: x[0])

        # Create timestamped dialogue structure for database storage
        current_timestamped_dialogue = [
            {"timestamp": time, "speaker": speaker, "text": text} 
            for time, speaker, text in combined_with_timestamps
        ]
        
        # Also create traditional combined transcription for backward compatibility
        final_dialogue = [f"{speaker}: {text}" for _, speaker, text in combined_with_timestamps]
        current_combined_transcription = "\n".join(final_dialogue)

        # Handle transcription concatenation with existing data and timestamps
        existing_transcription = existing_call.get('transcription', {})
        
        if existing_transcription and isinstance(existing_transcription, dict):
            existing_combined = existing_transcription.get('combined', '')
            existing_agent = existing_transcription.get('agent', '')
            existing_client = existing_transcription.get('client', '')
            existing_timestamped = existing_transcription.get('timestamped_dialogue', [])
            
            # Combine with new transcriptions
            combined_transcription = (existing_combined + "\n" + current_combined_transcription).strip()
            final_agent_text = (existing_agent + " " + agent_text).strip()
            final_client_text = (existing_client + " " + client_text).strip()
            final_timestamped_dialogue = existing_timestamped + current_timestamped_dialogue
        else:
            # No existing transcription or invalid format
            combined_transcription = current_combined_transcription
            final_agent_text = agent_text
            final_client_text = client_text
            final_timestamped_dialogue = current_timestamped_dialogue

        # Store final audio and data
        client_audio.seek(0)
        agent_audio.seek(0)
        
        # Prepare final transcription data with timestamps
        final_transcription_data = {
            "agent": final_agent_text,
            "client": final_client_text,
            "combined": combined_transcription,
            "timestamped_dialogue": final_timestamped_dialogue
        }
        
        stored_audio_result = call_ops.store_audio_and_update_call(
            call_id=call_id,
            client_audio=client_audio,
            agent_audio=agent_audio,
            transcription_data=final_transcription_data,
            analysis_data={"duration": duration},
            is_final=True
        )

        # Perform final analysis using adaptive windows approach (tests window sizes 1, 2, 3)
        # Use uploaded transcript_lines if available, otherwise use default script
        script_to_use = transcript_lines if transcript_lines else None
        results = check_script_adherence_adaptive_windows(final_agent_text, script_to_use)
        overall_adherence = results.get("real_time_adherence_score", 0)
        script_completion = results.get("script_completion_percentage", 0)
        adherence_details = results.get("checkpoint_results", [])
        
        print(f"[FINAL_UPDATE] Final adaptive windows adherence analysis complete:")
        print(f"[FINAL_UPDATE] - Real-time adherence score: {overall_adherence}%")
        print(f"[FINAL_UPDATE] - Script completion: {script_completion}%")
        print(f"[FINAL_UPDATE] - Total passed checkpoints: {len([r for r in adherence_details if r.get('status') == 'PASS'])}")
        print(f"[FINAL_UPDATE] - Window size usage: {results.get('window_size_usage', {})}")
        print(f"[FINAL_UPDATE] - Method: {results.get('method', 'adaptive_windows')}")

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
            "debug_info": results.get("debug_info", {})
        }
        
        # Complete the call update with timestamped dialogue - use call_ops directly
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
            timestamped_dialogue=final_timestamped_dialogue
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
            }
        }), 200
    
    except Exception as e:
        print(f"[FINAL_UPDATE ERROR] {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500



@app.route('/compare_methods', methods=['POST'])
def compare_methods():
    """
    Compare standard and adaptive windows adherence methods side-by-side
    """
    try:
        data = request.get_json()
        transcript_text = data.get('transcript_text', '')
        
        if not transcript_text:
            return jsonify({'error': 'Missing transcript_text'}), 400
        
        # Run both methods
        standard_result = check_script_adherence(transcript_text)
        adaptive_result = check_script_adherence_adaptive_windows(transcript_text)
        
        return jsonify({
            'status': 'success',
            'comparison': {
                'standard_method': {
                    'real_time_score': standard_result['real_time_adherence_score'],
                    'completion_percentage': standard_result['script_completion_percentage'],
                    'method': 'standard',
                    'details': standard_result
                },
                'adaptive_windows_method': {
                    'real_time_score': adaptive_result['real_time_adherence_score'],
                    'completion_percentage': adaptive_result['script_completion_percentage'],
                    'method': 'adaptive_windows',
                    'window_size_usage': adaptive_result.get('window_size_usage', {}),
                    'details': adaptive_result
                }
            }
        })
        
    except Exception as e:
        print(f"[COMPARE_METHODS] Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_call', methods=['GET'])
def get_call():
    """Endpoint to fetch a call document based on call_id."""
    print("[ROUTE] GET /get_call")
    call_id = request.args.get('call_id')  # Get call_id from query parameters

    if not call_id:
        return jsonify({"error": "call_id is required"}), 400

    try:
        call_ops = get_call_operations()
        result = call_ops.get_call_with_audio(call_id)

        if not result:
            return jsonify({"error": "No record found for the given CallID"}), 404

        return jsonify({"status": "success", "data": result}), 200

    except Exception as e:
        print(f"[DB ERROR] {e}")
        return jsonify({"error": "Database error occurred"}), 500

@app.route('/get_call_audio', methods=['GET'])
def get_call_audio_file():
    """Endpoint to retrieve the complete audio file for a call."""
    print("[ROUTE] GET /get_call_audio")
    call_id = request.args.get('call_id')
    audio_type = request.args.get('audio_type')  # 'agent' or 'client'

    if not call_id:
        return jsonify({"error": "call_id is required"}), 400

    if not audio_type or audio_type not in ['agent', 'client']:
        return jsonify({"error": "audio_type must be 'agent' or 'client'"}), 400

    try:
        db = get_db()
        audio_data = db.get_call_audio(call_id, audio_type)

        if not audio_data:
            return jsonify({"error": f"Audio file not found for {audio_type}"}), 404

        # Return audio file with appropriate headers
        from flask import Response
        filename = f"{call_id}_{audio_type}.wav"
        return Response(
            audio_data,
            mimetype='audio/wav',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Length': str(len(audio_data))
            }
        )

    except Exception as e:
        print(f"[AUDIO ERROR] Failed to retrieve audio for call {call_id}, type {audio_type}: {e}")
        return jsonify({"error": "Failed to retrieve audio file"}), 500

@app.route('/get_call_audio_info', methods=['GET'])
def get_call_audio_info():
    """Endpoint to get audio file information for a call."""
    print("[ROUTE] GET /get_call_audio_info")
    call_id = request.args.get('call_id')

    if not call_id:
        return jsonify({"error": "call_id is required"}), 400

    try:
        db = get_db()
        audio_info = db.get_call_audio_info(call_id)

        return jsonify({
            "status": "success",
            "call_id": call_id,
            "audio_info": audio_info
        }), 200

    except Exception as e:
        print(f"[AUDIO ERROR] Failed to get audio info for call {call_id}: {e}")
        return jsonify({"error": "Failed to retrieve audio information"}), 500

@app.route('/get_call_script', methods=['GET'])
def get_call_script_endpoint():
    """Get the current call script"""
    print("[ROUTE] GET /get_call_script")
    try:
        script = get_current_call_script()
        return jsonify({
            "status": "success",
            "script": script,
            "script_type": "fallback",
            "total_lines": len(script)
        }), 200
    except Exception as e:
        print(f"[ROUTE ERROR] /get_call_script: {e}")
        return jsonify({"error": "Failed to get call script"}), 500

@app.route('/update_call_script', methods=['POST'])
def update_call_script_endpoint():
    """Update the call script (placeholder for future frontend integration)"""
    print("[ROUTE] POST /update_call_script")
    try:
        data = request.get_json()
        new_script = data.get('script', [])
        
        if not isinstance(new_script, list):
            return jsonify({"error": "Script must be a list of strings"}), 400
        
        # For now, just validate the script format
        # Later this can be updated to save to database/config
        
        # Basic validation
        if len(new_script) == 0:
            return jsonify({"error": "Script cannot be empty"}), 400
        
        for line in new_script:
            if not isinstance(line, str):
                return jsonify({"error": "All script lines must be strings"}), 400
        
        return jsonify({
            "status": "success",
            "message": "Script format validated. Update functionality will be implemented later.",
            "received_lines": len(new_script),
            "note": "Currently using fallback script. Database integration pending."
        }), 200
        
    except Exception as e:
        print(f"[ROUTE ERROR] /update_call_script: {e}")
        return jsonify({"error": "Failed to update call script"}), 500

@app.route('/get_timestamped_dialogue', methods=['GET'])
def get_timestamped_dialogue():
    """Get timestamped dialogue for a specific call"""
    print("[ROUTE] GET /get_timestamped_dialogue")
    call_id = request.args.get('call_id')

    if not call_id:
        return jsonify({"error": "call_id is required"}), 400

    try:
        call_ops = get_call_operations()
        call_data = call_ops.get_call(call_id)

        if not call_data:
            return jsonify({"error": "No record found for the given CallID"}), 404

        transcription = call_data.get('transcription', {})
        timestamped_dialogue = transcription.get('timestamped_dialogue', [])

        return jsonify({
            "status": "success",
            "call_id": call_id,
            "timestamped_dialogue": timestamped_dialogue,
            "total_sentences": len(timestamped_dialogue),
            "has_timestamps": len(timestamped_dialogue) > 0
        }), 200

    except Exception as e:
        print(f"[ROUTE ERROR] /get_timestamped_dialogue: {e}")
        return jsonify({"error": "Failed to retrieve timestamped dialogue"}), 500



def check_script_adherence_adaptive_windows(agent_text, script_checkpoints=None):
    """
    ADAPTIVE WINDOW APPROACH - Tests window sizes 1, 2, and 3 for each checkpoint
    and automatically selects the best performing window size for optimal results.
    
    This approach combines the benefits of:
    - Individual sentence matching (window size 1)
    - 2-sentence context windows (window size 2) 
    - 3-sentence context windows (window size 3)
    
    For each checkpoint, it tests all three window sizes and keeps the best score.
    """
    print(f"[ADAPTIVE] ========== Starting Adaptive Window Analysis ==========")
    
    # --- Step 1: Initial setup ---
    if script_checkpoints is None:
        script_checkpoints = get_current_call_script()

    # Tokenize sentences
    clean_agent_text = agent_text.lower()
    try:
        agent_sentences = nltk.sent_tokenize(clean_agent_text)
    except:
        # Fallback if NLTK is not available
        agent_sentences = [s.strip() for s in clean_agent_text.split('.') if s.strip()]
    
    if not agent_sentences:
        return {"real_time_adherence_score": 0, "script_completion_percentage": 0, "checkpoint_results": []}

    print(f"[ADAPTIVE] [SENTENCES] Agent said {len(agent_sentences)} sentences:")
    for i, sentence in enumerate(agent_sentences):
        print(f"[ADAPTIVE] [SENTENCE {i+1}] {sentence}")

    # Pre-compute sentence embeddings (crucial for performance)
    agent_embeddings = None
    if SBERT_AVAILABLE and sbert_model:
        try:
            agent_embeddings = sbert_model.encode(agent_sentences, convert_to_tensor=True)
            print(f"[ADAPTIVE] [EMBEDDINGS] Computed embeddings for {len(agent_sentences)} sentences.")
        except Exception as e:
            print(f"[ADAPTIVE] [ERROR] Failed to compute embeddings: {e}")

    checkpoint_results = []
    
    # --- Step 2: Process each checkpoint with adaptive window sizing ---
    for i, checkpoint in enumerate(script_checkpoints):
        checkpoint_id = checkpoint.get("checkpoint_id", f"unknown_{i}")
        prompt_text = checkpoint.get("prompt_text", "")
        adherence_type = checkpoint.get("adherence_type", "SEMANTIC")
        weight = checkpoint.get("weight", 1)
        is_mandatory = checkpoint.get("is_mandatory", False)
        
        print(f"\n[ADAPTIVE] [CHECKPOINT {i+1}] Analyzing: {checkpoint_id}")
        print(f"[ADAPTIVE] [PROMPT] {prompt_text}")
        print(f"[ADAPTIVE] [TYPE] {adherence_type}")
        
        clean_prompt_preserve = clean_text_for_matching(prompt_text, preserve_structure=True)
        clean_prompt_remove = clean_text_for_matching(prompt_text, preserve_structure=False)
        
        # Pre-compute prompt embeddings
        prompt_embedding_preserve, prompt_embedding_remove = None, None
        if SBERT_AVAILABLE and sbert_model:
            try:
                if clean_prompt_preserve:
                    prompt_embedding_preserve = sbert_model.encode(clean_prompt_preserve, convert_to_tensor=True)
                if clean_prompt_remove:
                    prompt_embedding_remove = sbert_model.encode(clean_prompt_remove, convert_to_tensor=True)
            except Exception as e:
                print(f"[ADAPTIVE] [ERROR] Failed to compute prompt embeddings: {e}")

        best_overall_score = 0
        best_window_size = 1
        best_match_details = "No match found"
        best_sentence_location = None
        
        # --- Step 3: Test each window size (1, 2, 3) ---
        for window_size in [1, 2, 3]:
            print(f"[ADAPTIVE] [WINDOW SIZE {window_size}] Testing...")
            
            best_score_for_this_window_size = 0
            best_match_info = None
            
            # Create all possible windows of this size
            for start_idx in range(len(agent_sentences)):
                end_idx = min(start_idx + window_size - 1, len(agent_sentences) - 1)
                
                # For each window, test each sentence within it
                for sent_idx in range(start_idx, end_idx + 1):
                    current_sentence = agent_sentences[sent_idx]
                    score_for_sentence = 0
                    method_used = ""

                    if adherence_type == "STRICT":
                        # Exact match check
                        if clean_prompt_preserve in current_sentence:
                            score_for_sentence = 100
                            method_used = "exact_preserve"
                        elif clean_prompt_remove in current_sentence:
                            score_for_sentence = 100
                            method_used = "exact_remove"
                        else:
                            # Fuzzy match fallback
                            score_preserve = fuzz.ratio(clean_prompt_preserve, current_sentence)
                            score_remove = fuzz.ratio(clean_prompt_remove, current_sentence)
                            score_for_sentence = max(score_preserve, score_remove)
                            method_used = "fuzzy"
                    
                    elif adherence_type == "SEMANTIC" and SBERT_AVAILABLE and agent_embeddings is not None:
                        # Compare prompt embeddings to the current sentence's embedding
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
                        # For TOPIC, use fuzzy matching
                        score_preserve = fuzz.ratio(clean_prompt_preserve, current_sentence)
                        score_remove = fuzz.ratio(clean_prompt_remove, current_sentence)
                        score_for_sentence = max(score_preserve, score_remove)
                        method_used = "fuzzy"

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
                
                # Early termination if we find a perfect match
                if best_score_for_this_window_size >= 100:
                    break
            
            print(f"[ADAPTIVE] [WINDOW SIZE {window_size}] Best score: {best_score_for_this_window_size:.2f}%")
            
            # Check if this window size gave us the best score so far
            if best_score_for_this_window_size > best_overall_score:
                best_overall_score = best_score_for_this_window_size
                best_window_size = window_size
                best_sentence_location = best_match_info
                best_match_details = f"Best match {best_overall_score:.2f}% found using window size {window_size} at sentence {best_match_info['sentence_idx']} (window {best_match_info['window_start']}-{best_match_info['window_end']})"

        # --- Step 4: Determine final pass/fail status ---
        threshold = {"STRICT": 85, "SEMANTIC": 60, "TOPIC": 40}.get(adherence_type, 60)
        status = "PASS" if best_overall_score >= threshold else "FAIL"

        print(f"[ADAPTIVE] [CHECKPOINT {i+1}] 🏆 WINNER: Window Size {best_window_size} with {best_overall_score:.2f}%")
        print(f"[ADAPTIVE] [CHECKPOINT {i+1}] Final result: {status} ({best_overall_score:.2f}% >= {threshold}%)")
        print(f"[ADAPTIVE] [BEST MATCH] {best_match_details}")
        if best_sentence_location:
            print(f"[ADAPTIVE] [SENTENCE] \"{best_sentence_location['sentence_text']}\" ({best_sentence_location['method']})")

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

    # --- Step 5: Calculate final score using real-time heuristic ---
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

    # Analyze window size usage
    window_size_usage = {}
    for result in checkpoint_results:
        ws = result["optimal_window_size"]
        window_size_usage[ws] = window_size_usage.get(ws, 0) + 1
    
    print(f"\n[ADAPTIVE] [FINAL] Real-time adherence score: {real_time_adherence_score}%")
    print(f"[ADAPTIVE] [FINAL] Script completion: {script_completion_percentage}%")
    print(f"[ADAPTIVE] [FINAL] Passed checkpoints: {passed_checkpoints}/{len(checkpoint_results)}")
    print(f"[ADAPTIVE] [WINDOW ANALYSIS] Window size usage: {window_size_usage}")
    
    return {
        "real_time_adherence_score": real_time_adherence_score,
        "script_completion_percentage": script_completion_percentage,
        "checkpoint_results": checkpoint_results,
        "method": "adaptive_windows",
        "window_size_usage": window_size_usage,
        "total_checkpoints": len(checkpoint_results)
    }

if __name__ == '__main__':
    # Configure Flask for production deployment (Render) or local development
    # Render provides PORT environment variable
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Always use 0.0.0.0 for deployment - don't use HOST env var as it may contain
    # incompatible hostnames from other platforms (like Railway)
    host = '0.0.0.0'
    
    print(f"[SERVER] Starting Flask server at http://{host}:{port}")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug
        )
    except Exception as e:
        print(f"[SERVER ERROR] Failed to start server: {e}")
    finally:
        print("[SERVER] Server stopped")
