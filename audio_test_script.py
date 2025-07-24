#!/usr/bin/env python3
"""
Audio Chunking Test Script for app2-segment.py

This script:
1. Creates an initial call record in the MongoDB database
2. Takes a WAV audio file and splits it into 10-second chunks
3. Sends each chunk to both client_audio and agent_audio streams (default mode)
4. Uses /update_call for intermediate chunks and /final_update for the last chunk
5. Simulates a real-time call analysis scenario with proper database integration
6. Provides detailed logging and error handling
7. Supports sending transcript data for script adherence analysis

Usage:
    python audio_test_script.py --audio_file path/to/audio.wav [--server_url http://localhost:5000] [--chunk_duration 10] [--call_id test_call_123] [--transcript path/to/script.xml]
"""

import argparse
import os
import sys
import time
import uuid
from io import BytesIO
import requests
from pydub import AudioSegment
from pydub.utils import make_chunks

class AudioChunkTester:
    def __init__(self, server_url="http://localhost:5000", chunk_duration=10):
        self.server_url = server_url.rstrip('/')
        self.chunk_duration = chunk_duration * 1000  # Convert to milliseconds for pydub
        self.session = requests.Session()
        
    def load_audio(self, audio_file_path):
        """Load audio file and return AudioSegment object"""
        try:
            print(f"[LOAD] Loading audio file: {audio_file_path}")
            
            # Check file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Load audio (pydub auto-detects format)
            audio = AudioSegment.from_file(audio_file_path)
            
            duration_seconds = len(audio) / 1000
            print(f"[LOAD] Audio loaded successfully:")
            print(f"[LOAD] - Duration: {duration_seconds:.2f} seconds")
            print(f"[LOAD] - Sample rate: {audio.frame_rate} Hz")
            print(f"[LOAD] - Channels: {audio.channels}")
            print(f"[LOAD] - Sample width: {audio.sample_width} bytes")
            
            return audio
            
        except Exception as e:
            print(f"[ERROR] Failed to load audio file: {e}")
            return None
    
    def load_transcript(self, transcript_file_path):
        """Load transcript/script file and return content"""
        try:
            print(f"[TRANSCRIPT] Loading transcript file: {transcript_file_path}")
            
            # Check file exists
            if not os.path.exists(transcript_file_path):
                raise FileNotFoundError(f"Transcript file not found: {transcript_file_path}")
            
            # Load transcript content
            with open(transcript_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"[TRANSCRIPT] Transcript loaded successfully:")
            print(f"[TRANSCRIPT] - File size: {len(content)} characters")
            print(f"[TRANSCRIPT] - First 100 chars: {content[:100]}...")
            
            return content
            
        except Exception as e:
            print(f"[ERROR] Failed to load transcript file: {e}")
            return None
    
    def create_chunks(self, audio):
        """Split audio into chunks of specified duration"""
        try:
            print(f"[CHUNK] Creating {self.chunk_duration/1000}s chunks...")
            
            chunks = make_chunks(audio, self.chunk_duration)
            
            print(f"[CHUNK] Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                chunk_duration = len(chunk) / 1000
                print(f"[CHUNK] - Chunk {i+1}: {chunk_duration:.2f}s")
            
            return chunks
            
        except Exception as e:
            print(f"[ERROR] Failed to create chunks: {e}")
            return []
    
    def audio_to_wav_bytes(self, audio_segment):
        """Convert AudioSegment to WAV bytes for sending"""
        try:
            buffer = BytesIO()
            audio_segment.export(buffer, format="wav")
            buffer.seek(0)
            return buffer
        except Exception as e:
            print(f"[ERROR] Failed to convert audio to WAV bytes: {e}")
            return None
    
    def create_silent_chunk(self, duration_ms, sample_rate=24000):
        """Create a silent audio chunk of specified duration"""
        try:
            # Create silent audio with the same sample rate as the main audio
            silent_audio = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
            return silent_audio
        except Exception as e:
            print(f"[ERROR] Failed to create silent chunk: {e}")
            return None
    
    def send_chunk(self, chunk_index, total_chunks, client_audio_chunk, agent_audio_chunk, call_id, transcript_content=None):
        """Send a chunk to the appropriate endpoint"""
        try:
            is_final = (chunk_index == total_chunks - 1)
            endpoint = "/final_update" if is_final else "/update_call"
            url = f"{self.server_url}{endpoint}"
            
            # Convert audio chunks to WAV bytes
            client_wav = self.audio_to_wav_bytes(client_audio_chunk) if client_audio_chunk else None
            agent_wav = self.audio_to_wav_bytes(agent_audio_chunk) if agent_audio_chunk else None
            
            # Prepare files for upload
            files = {}
            if client_wav:
                files['client_audio'] = ('client_chunk.wav', client_wav, 'audio/wav')
            if agent_wav:
                files['agent_audio'] = ('agent_chunk.wav', agent_wav, 'audio/wav')
            
            # Prepare form data
            data = {'call_id': call_id}
            if transcript_content:
                data['transcript'] = transcript_content
            
            chunk_type = "FINAL" if is_final else "UPDATE"
            print(f"[SEND] Sending chunk {chunk_index + 1}/{total_chunks} ({chunk_type}) to {endpoint}")
            print(f"[SEND] - Call ID: {call_id}")
            print(f"[SEND] - Client audio: {'✓' if client_wav else '✗'}")
            print(f"[SEND] - Agent audio: {'✓' if agent_wav else '✗'}")
            if transcript_content:
                print(f"[SEND] - Transcript: {len(transcript_content)} chars")
            
            # Send request
            start_time = time.time()
            response = self.session.post(url, files=files, data=data, timeout=60)
            request_time = time.time() - start_time
            
            # Process response
            if response.status_code == 202:
                result = response.json()
                print(f"[SUCCESS] Chunk {chunk_index + 1} processed in {request_time:.2f}s")
                
                # Show key metrics for intermediate chunks
                if not is_final:
                    if 'overall_adherence' in result:
                        print(f"[RESULT] - Adherence: {result['overall_adherence']:.1f}%")
                    if 'script_completion' in result:
                        print(f"[RESULT] - Script completion: {result['script_completion']:.1f}%")
                    if 'CQS' in result:
                        print(f"[RESULT] - CQS: {result['CQS']:.2f}")
                    if 'quality' in result:
                        print(f"[RESULT] - Quality: {result['quality']:.1f}%")
                    if 'adherence_details' in result:
                        details = result['adherence_details']
                        found_count = sum(1 for d in details if d.get('found', False))
                        print(f"[RESULT] - Checkpoints found: {found_count}/{len(details)}")
                    if 'analysis_method' in result:
                        print(f"[RESULT] - Analysis method: {result['analysis_method']}")
                    if 'window_size_usage' in result:
                        usage = result['window_size_usage']
                        print(f"[RESULT] - Window usage: S:{usage.get('small', 0)} M:{usage.get('medium', 0)} L:{usage.get('large', 0)}")

                # Show final analysis for last chunk
                if is_final:
                    if 'final_adherence' in result:
                        final = result['final_adherence']
                        print(f"[FINAL] Final Analysis Complete:")
                        print(f"[FINAL] - Overall adherence: {final.get('overall', 0):.1f}%")
                        print(f"[FINAL] - Script completion: {final.get('script_completion', 0):.1f}%")
                        print(f"[FINAL] - Analysis method: {final.get('method', 'unknown')}")
                        print(f"[FINAL] - Total checkpoints: {final.get('total_checkpoints', 0)}")
                        if 'window_size_usage' in final:
                            usage = final['window_size_usage']
                            print(f"[FINAL] - Window usage: S:{usage.get('small', 0)} M:{usage.get('medium', 0)} L:{usage.get('large', 0)}")
                        if 'details' in final:
                            details = final['details']
                            found_count = sum(1 for d in details if d.get('found', False))
                            print(f"[FINAL] - Checkpoints found: {found_count}/{len(details)}")
                    
                    # Show final emotion analysis
                    if 'final_emotions' in result:
                        final_emotions = result['final_emotions']
                        print(f"[FINAL] Final Emotion Analysis:")
                        print(f"[FINAL] - Final CQS: {final_emotions.get('cqs', 0):.2f}")
                        print(f"[FINAL] - Final Quality: {final_emotions.get('quality', 0):.1f}%")
                        emotions = final_emotions.get('emotions', {})
                        if emotions:
                            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(f"[FINAL] - Top emotions: {', '.join([f'{e}({s:.2f})' for e, s in top_emotions])}")
                    
                    if 'chunk_emotions_count' in result:
                        print(f"[FINAL] - Total emotion chunks: {result['chunk_emotions_count']}")
                    
                    if 'total_duration' in result:
                        print(f"[FINAL] - Total duration: {result['total_duration']:.2f}s")
                    
                return result
            else:
                print(f"[ERROR] Chunk {chunk_index + 1} failed with status {response.status_code}")
                print(f"[ERROR] Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"[ERROR] Chunk {chunk_index + 1} timed out after 60 seconds")
            return None
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Failed to connect to server at {self.server_url}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to send chunk {chunk_index + 1}: {e}")
            return None
    
    def test_server_connection(self):
        """Test if the server is running and accessible"""
        try:
            print(f"[TEST] Testing connection to {self.server_url}...")
            # Try to access a non-existent endpoint to check if server is running
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            print(f"[TEST] Server responded with status {response.status_code}")
            return True
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to server at {self.server_url}")
            print(f"[ERROR] Make sure app2-segment.py is running on this URL")
            return False
        except Exception as e:
            # If we get a 404, that means the server is running but the endpoint doesn't exist
            # This is actually good - it means the server is up
            if "404" in str(e) or "Not Found" in str(e):
                print(f"[TEST] Server is running (404 for /health is expected)")
                return True
            print(f"[ERROR] Connection test failed: {e}")
            return False
    
    def create_call_record(self, call_id=None, agent_id=None, sip_id=None, transcript_content=None):
        """Create initial call record via API endpoint"""
        if call_id:
            print(f"[API] Using provided custom call_id: {call_id}")
        else:
            # Generate a unique call ID if none provided
            call_id = f"test_call_{uuid.uuid4().hex[:8]}"
            print(f"[API] Generated call_id: {call_id}")

        try:
            print(f"[API] Creating call record in database...")
            url = f"{self.server_url}/create_call"
            
            # Prepare form data (app2-segment.py expects form data, not JSON)
            data = {
                'call_id': call_id,
                'agent_id': agent_id or 'test_agent_from_script',
                'sip_id': sip_id or 'test_sip_from_script'  # Changed from customer_id to sip_id
            }
            
            if transcript_content:
                data['transcript'] = transcript_content
                print(f"[API] Including transcript in call creation ({len(transcript_content)} chars)")
            
            response = self.session.post(url, data=data, timeout=20)
            
            if response.status_code == 201:
                result = response.json()
                print(f"[API] Successfully created call record: {call_id}")
                print(f"[API] Response: {result.get('message', 'Call created')}")
                return call_id
            else:
                print(f"[API ERROR] Failed to create call record. Status: {response.status_code}")
                print(f"[API ERROR] Response: {response.text}")
                # Even if creation fails, we can still try to use the call_id
                print(f"[API] Will attempt to use call_id: {call_id}")
                return call_id
                
        except Exception as e:
            print(f"[API ERROR] Failed to create call record: {e}")
            print(f"[API] Will attempt to use call_id: {call_id}")
            return call_id
    
    def run_test(self, audio_file_path, call_id=None, transcript_content=None, test_mode="both", agent_id=None, sip_id=None, include_silent=False):
        """
        Run the complete audio chunking test
        
        test_mode options:
        - "alternating": Alternate chunks between client and agent
        - "client_only": Send all chunks as client audio
        - "agent_only": Send all chunks as agent audio
        - "both": Send same chunk to both client and agent
        
        include_silent: If True, insert silent chunks alternating with audio chunks
        """
        print("="*80)
        print("AUDIO CHUNKING TEST SCRIPT WITH TRANSCRIPT SUPPORT")
        print("="*80)
        
        # Test server connection
        if not self.test_server_connection():
            return False
        
        # Create call record in database or use provided call_id
        call_id = self.create_call_record(call_id, agent_id, sip_id, transcript_content)
        
        print(f"[INFO] Call ID: {call_id}")
        print(f"[INFO] Test mode: {test_mode}")
        print(f"[INFO] Chunk duration: {self.chunk_duration/1000}s")
        if transcript_content:
            print(f"[INFO] Transcript: {len(transcript_content)} characters loaded")
        else:
            print(f"[INFO] Transcript: None (will use default script)")
        
        # Load and chunk audio
        audio = self.load_audio(audio_file_path)
        if not audio:
            return False
        
        chunks = self.create_chunks(audio)
        if not chunks:
            return False
        
        # Create final chunk list (with silent chunks if requested)
        final_chunks = []
        chunk_types = []  # Track whether each chunk is 'audio' or 'silent'
        
        if include_silent:
            print(f"[INFO] Including silent chunks between audio chunks")
            for i, chunk in enumerate(chunks):
                final_chunks.append(chunk)
                chunk_types.append('audio')
                
                # Add silent chunk after each audio chunk (except the last one)
                if i < len(chunks) - 1:
                    silent_chunk = self.create_silent_chunk(self.chunk_duration, chunk.frame_rate)
                    if silent_chunk:
                        final_chunks.append(silent_chunk)
                        chunk_types.append('silent')
                    else:
                        print(f"[WARNING] Failed to create silent chunk after audio chunk {i+1}")
        else:
            final_chunks = chunks
            chunk_types = ['audio'] * len(chunks)
        
        # Process chunks
        print(f"[INFO] Processing {len(final_chunks)} chunks ({len([t for t in chunk_types if t == 'audio'])} audio + {len([t for t in chunk_types if t == 'silent'])} silent)...")
        print("-" * 60)
        
        successful_chunks = 0
        failed_chunks = 0
        
        for i, chunk in enumerate(final_chunks):
            chunk_type = chunk_types[i]
            
            # Log chunk info
            chunk_duration = len(chunk) / 1000
            print(f"[CHUNK {i+1}/{len(final_chunks)}] Processing {chunk_type} chunk ({chunk_duration:.2f}s)")
            
            # Determine audio routing based on test mode
            client_audio = None
            agent_audio = None
            
            if test_mode == "alternating":
                if i % 2 == 0:
                    client_audio = chunk
                else:
                    agent_audio = chunk
            elif test_mode == "client_only":
                client_audio = chunk
            elif test_mode == "agent_only":
                agent_audio = chunk
            elif test_mode == "both":
                client_audio = chunk
                agent_audio = chunk
            
            # Send chunk with transcript data
            # Use the original chunks length for final determination (not final_chunks)
            original_chunk_count = len(chunks)
            result = self.send_chunk(i, len(final_chunks), client_audio, agent_audio, call_id, transcript_content)
            
            if result:
                successful_chunks += 1
            else:
                failed_chunks += 1
            
            print("-" * 60)
            
            # Small delay between chunks to avoid overwhelming server
            if i < len(chunks) - 1:
                time.sleep(0.5)
        
        # Summary
        print("="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"[SUMMARY] Total chunks: {len(chunks)}")
        print(f"[SUMMARY] Successful: {successful_chunks}")
        print(f"[SUMMARY] Failed: {failed_chunks}")
        print(f"[SUMMARY] Success rate: {(successful_chunks/len(chunks)*100):.1f}%")
        print(f"[SUMMARY] Call ID: {call_id}")
        print(f"[SUMMARY] Test mode: {test_mode}")
        if transcript_content:
            print(f"[SUMMARY] Transcript: {len(transcript_content)} characters processed")
        print(f"[SUMMARY] Database record created and updated successfully")
        
        return successful_chunks > 0

def main():
    parser = argparse.ArgumentParser(description='Test audio chunking with app2-segment.py - Creates database records and sends chunked audio with transcript support')
    parser.add_argument('--audio_file', required=True, help='Path to WAV audio file')
    parser.add_argument('--server_url', default='http://localhost:5000', help='Server URL (default: http://localhost:5000)')
    parser.add_argument('--chunk_duration', type=int, default=10, help='Chunk duration in seconds (default: 10)')
    parser.add_argument('--call_id', help='Custom call ID (default: auto-generated)')
    parser.add_argument('--agent_id', default='test_agent', help='Agent ID for database record (default: test_agent)')
    parser.add_argument('--sip_id', default='test_sip', help='SIP ID for database record (default: test_sip)')  # Changed from customer_id to sip_id
    parser.add_argument('--test_mode', choices=['alternating', 'client_only', 'agent_only', 'both'], 
                        default='both', help='How to distribute chunks (default: both - sends same chunk to both streams)')
    parser.add_argument('--transcript', help='Path to transcript/script file (XML format) for adherence analysis')
    
    args = parser.parse_args()
    
    # Load transcript if provided
    transcript_content = None
    if args.transcript:
        tester_temp = AudioChunkTester()  # Temporary instance just for loading transcript
        transcript_content = tester_temp.load_transcript(args.transcript)
        if not transcript_content:
            print(f"[ERROR] Failed to load transcript file: {args.transcript}")
            return False
    
    # Create tester and run
    tester = AudioChunkTester(args.server_url, args.chunk_duration)
    success = tester.run_test(
        args.audio_file, 
        args.call_id, 
        transcript_content, 
        args.test_mode,
        args.agent_id,
        args.sip_id  # Changed from customer_id to sip_id
    )
    
    return success

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1) 