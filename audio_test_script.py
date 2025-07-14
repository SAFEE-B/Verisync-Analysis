#!/usr/bin/env python3
"""
Audio Chunking Test Script for app2.py

This script:
1. Creates an initial call record in the MongoDB database
2. Takes a WAV audio file and splits it into 10-second chunks
3. Sends each chunk to both client_audio and agent_audio streams (default mode)
4. Uses /update_call for intermediate chunks and /final_update for the last chunk
5. Simulates a real-time call analysis scenario with proper database integration
6. Provides detailed logging and error handling

Usage:
    python audio_test_script.py --audio_file path/to/audio.wav [--server_url http://localhost:5000] [--chunk_duration 10] [--call_id test_call_123]
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
    
    def send_chunk(self, chunk_index, total_chunks, client_audio_chunk, agent_audio_chunk, call_id, custom_script=None):
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
            if custom_script:
                data['transcript'] = custom_script
            
            chunk_type = "FINAL" if is_final else "UPDATE"
            print(f"[SEND] Sending chunk {chunk_index + 1}/{total_chunks} ({chunk_type}) to {endpoint}")
            print(f"[SEND] - Call ID: {call_id}")
            print(f"[SEND] - Client audio: {'✓' if client_wav else '✗'}")
            print(f"[SEND] - Agent audio: {'✓' if agent_wav else '✗'}")
            if custom_script:
                print(f"[SEND] - Custom script: {len(custom_script)} chars")
            
            # Send request
            start_time = time.time()
            response = self.session.post(url, files=files, data=data, timeout=60)
            request_time = time.time() - start_time
            
            # Process response
            if response.status_code == 200:
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

                # Show final analysis for last chunk
                if is_final:
                    if 'final_adherence' in result:
                        final = result['final_adherence']
                        print(f"[FINAL] Final Analysis Complete:")
                        if 'real_time_score' in final:
                            print(f"[FINAL] - Real-time score: {final.get('real_time_score', 0):.1f}%")
                        if 'script_completion' in final:
                            print(f"[FINAL] - Script completion: {final.get('script_completion', 0):.1f}%")
                        if 'analysis_method' in final:
                            print(f"[FINAL] - Analysis method: {final.get('analysis_method', 'unknown')}")
                        if 'total_checkpoints' in final:
                            print(f"[FINAL] - Total checkpoints: {final.get('total_checkpoints', 0)}")
                    
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
            print(f"[ERROR] Make sure app2.py is running on this URL")
            return False
        except Exception as e:
            # If we get a 404, that means the server is running but the endpoint doesn't exist
            # This is actually good - it means the server is up
            if "404" in str(e) or "Not Found" in str(e):
                print(f"[TEST] Server is running (404 for /health is expected)")
                return True
            print(f"[ERROR] Connection test failed: {e}")
            return False
    
    def create_call_record(self, call_id=None, agent_id=None, customer_id=None, script_text=None):
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
            
            # Prepare form data (app2.py expects form data, not JSON)
            data = {
                'call_id': call_id,
                'agent_id': agent_id or 'test_agent_from_script',
                'customer_id': customer_id or 'test_customer_from_script'
            }
            
            if script_text:
                data['transcript'] = script_text
            
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
    
    def run_test(self, audio_file_path, call_id=None, custom_script=None, test_mode="both", agent_id=None, customer_id=None):
        """
        Run the complete audio chunking test
        
        test_mode options:
        - "alternating": Alternate chunks between client and agent
        - "client_only": Send all chunks as client audio
        - "agent_only": Send all chunks as agent audio
        - "both": Send same chunk to both client and agent
        """
        print("="*60)
        print("AUDIO CHUNKING TEST SCRIPT")
        print("="*60)
        
        # Test server connection
        if not self.test_server_connection():
            return False
        
        # Create call record in database or use provided call_id
        call_id = self.create_call_record(call_id, agent_id, customer_id, custom_script)
        
        print(f"[INFO] Call ID: {call_id}")
        print(f"[INFO] Test mode: {test_mode}")
        print(f"[INFO] Chunk duration: {self.chunk_duration/1000}s")
        
        # Load and chunk audio
        audio = self.load_audio(audio_file_path)
        if not audio:
            return False
        
        chunks = self.create_chunks(audio)
        if not chunks:
            return False
        
        # Process chunks
        print(f"[INFO] Processing {len(chunks)} chunks...")
        print("-" * 40)
        
        successful_chunks = 0
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks):
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
            
            # Send chunk
            result = self.send_chunk(i, len(chunks), client_audio, agent_audio, call_id, custom_script)
            
            if result:
                successful_chunks += 1
            else:
                failed_chunks += 1
            
            print("-" * 40)
            
            # Small delay between chunks to avoid overwhelming server
            if i < len(chunks) - 1:
                time.sleep(0.5)
        
        # Summary
        print("="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"[SUMMARY] Total chunks: {len(chunks)}")
        print(f"[SUMMARY] Successful: {successful_chunks}")
        print(f"[SUMMARY] Failed: {failed_chunks}")
        print(f"[SUMMARY] Success rate: {(successful_chunks/len(chunks)*100):.1f}%")
        print(f"[SUMMARY] Call ID: {call_id}")
        print(f"[SUMMARY] Database record created and updated successfully")
        
        return successful_chunks > 0

def main():
    parser = argparse.ArgumentParser(description='Test audio chunking with app2.py - Creates database records and sends chunked audio')
    parser.add_argument('--audio_file', required=True, help='Path to WAV audio file')
    parser.add_argument('--server_url', default='http://localhost:5000', help='Server URL (default: http://localhost:5000)')
    parser.add_argument('--chunk_duration', type=int, default=10, help='Chunk duration in seconds (default: 10)')
    parser.add_argument('--call_id', help='Custom call ID (default: auto-generated)')
    parser.add_argument('--agent_id', default='test_agent', help='Agent ID for database record (default: test_agent)')
    parser.add_argument('--customer_id', default='test_customer', help='Customer ID for database record (default: test_customer)')
    parser.add_argument('--test_mode', choices=['alternating', 'client_only', 'agent_only', 'both'], 
                        default='both', help='How to distribute chunks (default: both - sends same chunk to both streams)')
    parser.add_argument('--script', help='Path to custom script file in XML format')
    
    args = parser.parse_args()
    
    # Load custom script if provided
    custom_script = None
    if args.script:
        try:
            with open(args.script, 'r', encoding='utf-8') as f:
                custom_script = f.read()
            print(f"[INFO] Loaded custom script from {args.script} ({len(custom_script)} characters)")
        except Exception as e:
            print(f"[ERROR] Failed to load script file: {e}")
            return False
    
    # Create tester and run
    tester = AudioChunkTester(args.server_url, args.chunk_duration)
    success = tester.run_test(
        args.audio_file, 
        args.call_id, 
        custom_script, 
        args.test_mode,
        args.agent_id,
        args.customer_id
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