import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def transcribe_with_segments(audio_file_path):
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print('[ERROR] GROQ_API_KEY not found in environment variables')
        sys.exit(1)
    client = Groq(api_key=api_key)

    with open(audio_file_path, 'rb') as audio_file:
        print(f'[INFO] Transcribing {audio_file_path} using Whisper (segments granularity)...')
        response = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_file,
            language="en",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        return response

def print_segments(response):
    if hasattr(response, 'segments'):
        segments = response.segments
    elif isinstance(response, dict) and 'segments' in response:
        segments = response['segments']
    else:
        print('[ERROR] No segments found in response!')
        return

    print(f"\n[RESULT] {len(segments)} segments found:")
    for i, seg in enumerate(segments, 1):
        # Each segment should have start, end, and text
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        print(f"Segment {i}: {start:.2f}s - {end:.2f}s | {text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_whisper_segments.py <audio_file>")
        sys.exit(1)
    audio_file_path = sys.argv[1]
    if not os.path.exists(audio_file_path):
        print(f"[ERROR] File not found: {audio_file_path}")
        sys.exit(1)
    response = transcribe_with_segments(audio_file_path)
    print_segments(response) 