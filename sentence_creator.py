import os
import json
import time
from datetime import datetime
from io import BytesIO
import tempfile
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from rapidfuzz import fuzz
import nltk

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=groq_api_key)

# Initialize Sentence Transformer
try:
    print("[INFO] Loading Sentence Transformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    if torch.cuda.is_available():
        sbert_model = sbert_model.to('cuda')
        print("[INFO] SBERT model moved to GPU")
    else:
        print("[INFO] SBERT model running on CPU")
    SBERT_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] Failed to load Sentence Transformer: {e}")
    SBERT_AVAILABLE = False
    sbert_model = None

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    print("[INFO] NLTK punkt model ready")
except Exception as e:
    print(f"[WARNING] Failed to download NLTK punkt: {e}")

def transcribe_audio_with_groq(audio_file_path):
    """
    Transcribe audio file using Groq Whisper API with word-level timestamps
    """
    print(f"[TRANSCRIPTION] Starting transcription of: {audio_file_path}")
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            # Create BytesIO object for API
            audio_buffer = BytesIO(audio_file.read())
            audio_buffer.name = "audio.wav"
            
            print("[TRANSCRIPTION] Calling Groq Whisper API...")
            response = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_buffer,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            
            print("[TRANSCRIPTION] Groq Whisper transcription completed")
            
            # Extract text and word timestamps
            transcription = response.text or ""
            words = response.words if hasattr(response, 'words') else []
            
            if not words:
                print("[WARNING] No word-level timestamps received")
                return transcription, []
            
            # Process word timestamps
            word_data = []
            for word in words:
                if hasattr(word, 'word') and hasattr(word, 'start') and hasattr(word, 'end'):
                    word_data.append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'timestamp': word.start  # Use start time as primary timestamp
                    })
                elif isinstance(word, dict):
                    word_data.append({
                        'word': word.get('word', word.get('text', '')),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0),
                        'timestamp': word.get('start', 0)
                    })
            
            print(f"[TRANSCRIPTION] Extracted {len(word_data)} words with timestamps")
            return transcription, word_data
            
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return "", []

def get_semantic_match(current_text, script_sentences, model):
    """
    Compare current text against script sentences using semantic similarity
    """
    if not SBERT_AVAILABLE or model is None:
        return 0.0
    
    try:
        # Encode current text
        current_embedding = model.encode(current_text.lower())
        
        # Encode all script sentences
        script_embeddings = model.encode([s.lower() for s in script_sentences])
        
        # Calculate similarities
        similarities = [
            util.cos_sim(current_embedding, script_emb).item()
            for script_emb in script_embeddings
        ]
        
        return max(similarities)
    except Exception as e:
        print(f"[ERROR] Semantic matching failed: {e}")
        return 0.0

def detect_complete_thought(text):
    """
    Detect if text contains markers indicating a complete thought
    """
    text = text.lower()
    
    # Check for sentence-final punctuation
    if any(punct in text for punct in ['.', '?', '!']):
        return True, 'punctuation'
        
    # Check minimum word count for potential complete thought
    if len(text.split()) >= 5:
        return True, 'length'
        
    return False, None

def create_adherence_optimized_sentences(words, script_sentences=None, min_confidence=0.7):
    """
    Create sentences optimized for script adherence checking by combining
    multiple signals: semantic similarity, dialog acts, and pause detection
    """
    print(f"[SENTENCE_CREATION] Creating sentences from {len(words)} words")
    
    if not words:
        return []
    
    # Default script sentences if none provided
    if script_sentences is None:
        script_sentences = [
            "hello thanks for calling",
            "how can i help you today",
            "would you like to confirm",
            "thank you for choosing"
        ]
    
    sentence_boundaries = []
    current_start = words[0]['timestamp']
    current_words = []
    
    for i in range(len(words) - 1):
        current_words.append(words[i])
        current_text = ' '.join(w['word'] for w in current_words)
        
        # Calculate pause duration
        time_gap = words[i + 1]['timestamp'] - words[i]['timestamp']
        
        # Get semantic similarity with script sentences
        semantic_score = get_semantic_match(current_text, script_sentences, sbert_model)
        
        # Check for complete thought
        is_complete, thought_type = detect_complete_thought(current_text)
        
        # Decision factors
        long_pause = time_gap > 0.7
        good_semantic_match = semantic_score > min_confidence
        complete_thought = is_complete
        
        # Decision logic
        should_break = (
            (long_pause and len(current_words) > 3) or
            (good_semantic_match and complete_thought) or
            len(current_words) > 20  # safety max length
        )
        
        if should_break:
            sentence_boundaries.append({
                'start_time': current_start,
                'end_time': words[i]['timestamp'],
                'text': current_text,
                'word_count': len(current_words),
                'metadata': {
                    'semantic_score': round(semantic_score * 100, 2),
                    'thought_type': thought_type,
                    'pause_duration': round(time_gap, 2),
                    'break_reason': 'pause' if long_pause else 'semantic' if good_semantic_match else 'length'
                }
            })
            current_start = words[i + 1]['timestamp']
            current_words = []
    
    # Handle the last sentence if there are remaining words
    if current_words:
        sentence_boundaries.append({
            'start_time': current_start,
            'end_time': words[-1]['timestamp'],
            'text': ' '.join(w['word'] for w in current_words),
            'word_count': len(current_words),
            'metadata': {
                'semantic_score': 0,
                'thought_type': 'final',
                'pause_duration': 0,
                'break_reason': 'end_of_audio'
            }
        })
    
    print(f"[SENTENCE_CREATION] Created {len(sentence_boundaries)} sentences")
    return sentence_boundaries

def analyze_sentences(sentences, script_sentences=None):
    """
    Analyze the created sentences for adherence and quality
    """
    if not sentences:
        return {}
    
    analysis = {
        'total_sentences': len(sentences),
        'average_sentence_length': 0,
        'sentence_details': [],
        'break_reasons': {},
        'thought_types': {},
        'semantic_scores': []
    }
    
    total_words = 0
    for sentence in sentences:
        total_words += sentence['word_count']
        analysis['sentence_details'].append({
            'text': sentence['text'][:100] + "..." if len(sentence['text']) > 100 else sentence['text'],
            'duration': round(sentence['end_time'] - sentence['start_time'], 2),
            'word_count': sentence['word_count'],
            'break_reason': sentence['metadata']['break_reason'],
            'thought_type': sentence['metadata']['thought_type'],
            'semantic_score': sentence['metadata']['semantic_score']
        })
        
        # Count break reasons
        reason = sentence['metadata']['break_reason']
        analysis['break_reasons'][reason] = analysis['break_reasons'].get(reason, 0) + 1
        
        # Count thought types
        thought_type = sentence['metadata']['thought_type']
        analysis['thought_types'][thought_type] = analysis['thought_types'].get(thought_type, 0) + 1
        
        # Collect semantic scores
        if sentence['metadata']['semantic_score'] > 0:
            analysis['semantic_scores'].append(sentence['metadata']['semantic_score'])
    
    analysis['average_sentence_length'] = round(total_words / len(sentences), 2)
    if analysis['semantic_scores']:
        analysis['average_semantic_score'] = round(sum(analysis['semantic_scores']) / len(analysis['semantic_scores']), 2)
    
    return analysis

def main():
    """
    Main function to process audio file and create sentences
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sentence_creator.py <audio_file_path> [script_file_path]")
        print("Example: python sentence_creator.py audio.wav script.txt")
        return
    
    audio_file_path = sys.argv[1]
    script_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate audio file
    if not os.path.exists(audio_file_path):
        print(f"[ERROR] Audio file not found: {audio_file_path}")
        return
    
    print(f"[MAIN] Processing audio file: {audio_file_path}")
    
    # Load script if provided
    script_sentences = None
    if script_file_path and os.path.exists(script_file_path):
        try:
            with open(script_file_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
                # Simple script parsing - split by lines and clean
                script_sentences = [line.strip().lower() for line in script_content.split('\n') if line.strip()]
            print(f"[MAIN] Loaded {len(script_sentences)} script sentences from {script_file_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load script file: {e}")
    
    # Step 1: Transcribe audio
    print("\n" + "="*50)
    print("STEP 1: TRANSCRIPTION")
    print("="*50)
    
    transcription, words = transcribe_audio_with_groq(audio_file_path)
    
    if not transcription:
        print("[ERROR] No transcription obtained")
        return
    
    print(f"[TRANSCRIPTION] Full transcription: {transcription[:200]}{'...' if len(transcription) > 200 else ''}")
    print(f"[TRANSCRIPTION] Total words with timestamps: {len(words)}")
    
    # Step 2: Create sentences
    print("\n" + "="*50)
    print("STEP 2: SENTENCE CREATION")
    print("="*50)
    
    sentences = create_adherence_optimized_sentences(words, script_sentences)
    
    # Step 3: Analyze sentences
    print("\n" + "="*50)
    print("STEP 3: SENTENCE ANALYSIS")
    print("="*50)
    
    analysis = analyze_sentences(sentences, script_sentences)
    
    # Step 4: Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    print(f"Total sentences created: {analysis['total_sentences']}")
    print(f"Average sentence length: {analysis['average_sentence_length']} words")
    print(f"Average semantic score: {analysis.get('average_semantic_score', 'N/A')}%")
    
    print(f"\nBreak reasons:")
    for reason, count in analysis['break_reasons'].items():
        print(f"  {reason}: {count}")
    
    print(f"\nThought types:")
    for thought_type, count in analysis['thought_types'].items():
        print(f"  {thought_type}: {count}")
    
    print(f"\nDetailed sentences:")
    for i, detail in enumerate(analysis['sentence_details'], 1):
        print(f"\nSentence {i}:")
        print(f"  Text: {detail['text']}")
        print(f"  Duration: {detail['duration']}s")
        print(f"  Words: {detail['word_count']}")
        print(f"  Break reason: {detail['break_reason']}")
        print(f"  Thought type: {detail['thought_type']}")
        print(f"  Semantic score: {detail['semantic_score']}%")
    
    # Step 5: Save results to file
    output_file = f"sentence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results = {
        'audio_file': audio_file_path,
        'script_file': script_file_path,
        'transcription': transcription,
        'words': words,
        'sentences': sentences,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[MAIN] Results saved to: {output_file}")
    
    # Step 6: Create MongoDB-ready format
    print("\n" + "="*50)
    print("MONGODB FORMAT")
    print("="*50)
    
    mongodb_format = {
        'conversation_id': f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'words': words,
        'sentence_boundaries': [
            {
                'start_time': s['start_time'],
                'end_time': s['end_time'],
                'is_final': True
            }
            for s in sentences
        ],
        'metadata': {
            'audio_file': audio_file_path,
            'total_duration': words[-1]['timestamp'] if words else 0,
            'total_words': len(words),
            'total_sentences': len(sentences),
            'analysis_timestamp': datetime.now().isoformat()
        }
    }
    
    mongodb_file = f"mongodb_format_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(mongodb_file, 'w', encoding='utf-8') as f:
        json.dump(mongodb_format, f, indent=2, ensure_ascii=False)
    
    print(f"[MAIN] MongoDB format saved to: {mongodb_file}")
    print(f"[MAIN] You can use this format to store in your MongoDB database")

if __name__ == "__main__":
    main() 