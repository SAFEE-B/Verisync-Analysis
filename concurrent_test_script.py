#!/usr/bin/env python3
"""
Concurrent Call Testing Script for app2-segment.py

This script tests the system's ability to handle multiple simultaneous calls by:
1. Creating multiple call records concurrently
2. Processing audio chunks from different calls simultaneously
3. Testing database thread safety and resource management
4. Validating concurrent transcription and analysis
5. Measuring performance under load

Usage:
    python concurrent_test_script.py --audio_file audiotest.wav --num_calls 5 --transcript test_transcript.xml
"""

import argparse
import asyncio
import os
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import requests
from pydub import AudioSegment
from pydub.utils import make_chunks
import json
from datetime import datetime

class ConcurrentCallTester:
    def __init__(self, server_url="http://localhost:5000", chunk_duration=10, max_workers=10):
        self.server_url = server_url.rstrip('/')
        self.chunk_duration = chunk_duration * 1000  # Convert to milliseconds
        self.max_workers = max_workers
        self.session_pool = []
        self.results = {}
        self.lock = threading.Lock()
        
        # Create session pool for concurrent requests
        for _ in range(max_workers):
            session = requests.Session()
            session.timeout = 60
            self.session_pool.append(session)
    
    def get_session(self):
        """Get a session from the pool (thread-safe)"""
        with self.lock:
            if self.session_pool:
                return self.session_pool.pop()
            else:
                # Create new session if pool is empty
                session = requests.Session()
                session.timeout = 60
                return session
    
    def return_session(self, session):
        """Return a session to the pool (thread-safe)"""
        with self.lock:
            self.session_pool.append(session)
    
    def load_audio(self, audio_file_path):
        """Load audio file and return AudioSegment object"""
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            audio = AudioSegment.from_file(audio_file_path)
            return audio
        except Exception as e:
            print(f"[ERROR] Failed to load audio file: {e}")
            return None
    
    def load_transcript(self, transcript_file_path):
        """Load transcript file content"""
        try:
            if not transcript_file_path or not os.path.exists(transcript_file_path):
                return None
            
            with open(transcript_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[ERROR] Failed to load transcript: {e}")
            return None
    
    def create_chunks(self, audio):
        """Split audio into chunks"""
        try:
            chunks = make_chunks(audio, self.chunk_duration)
            return chunks
        except Exception as e:
            print(f"[ERROR] Failed to create chunks: {e}")
            return []
    
    def audio_to_wav_bytes(self, audio_segment):
        """Convert AudioSegment to WAV bytes"""
        try:
            buffer = BytesIO()
            audio_segment.export(buffer, format="wav")
            buffer.seek(0)
            return buffer
        except Exception as e:
            print(f"[ERROR] Failed to convert audio: {e}")
            return None
    
    def create_call_record(self, call_id, agent_id, customer_id, transcript_content):
        """Create a call record via API"""
        session = self.get_session()
        try:
            url = f"{self.server_url}/create_call"
            data = {
                'call_id': call_id,
                'agent_id': agent_id,
                'customer_id': customer_id
            }
            
            if transcript_content:
                data['transcript'] = transcript_content
            
            response = session.post(url, data=data, timeout=20)
            
            if response.status_code == 201:
                return True, response.json()
            else:
                return False, f"Status: {response.status_code}, Response: {response.text}"
                
        except Exception as e:
            return False, str(e)
        finally:
            self.return_session(session)
    
    def send_audio_chunk(self, call_id, chunk_index, total_chunks, client_audio, agent_audio, transcript_content):
        """Send a single audio chunk"""
        session = self.get_session()
        try:
            is_final = (chunk_index == total_chunks - 1)
            endpoint = "/final_update" if is_final else "/update_call"
            url = f"{self.server_url}{endpoint}"
            
            # Convert audio to WAV bytes
            files = {}
            if client_audio:
                client_wav = self.audio_to_wav_bytes(client_audio)
                if client_wav:
                    files['client_audio'] = ('client_chunk.wav', client_wav, 'audio/wav')
            
            if agent_audio:
                agent_wav = self.audio_to_wav_bytes(agent_audio)
                if agent_wav:
                    files['agent_audio'] = ('agent_chunk.wav', agent_wav, 'audio/wav')
            
            data = {'call_id': call_id}
            if transcript_content:
                data['transcript'] = transcript_content
            
            start_time = time.time()
            response = session.post(url, files=files, data=data, timeout=60)
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result['request_time'] = request_time
                result['endpoint'] = endpoint
                result['chunk_index'] = chunk_index
                return True, result
            else:
                return False, f"Status: {response.status_code}, Response: {response.text}"
                
        except Exception as e:
            return False, str(e)
        finally:
            self.return_session(session)
    
    def process_single_call(self, call_config):
        """Process a single call with all its chunks"""
        call_id = call_config['call_id']
        agent_id = call_config['agent_id']
        customer_id = call_config['customer_id']
        audio_chunks = call_config['audio_chunks']
        transcript_content = call_config['transcript_content']
        test_mode = call_config.get('test_mode', 'both')
        
        print(f"[CALL-{call_id}] Starting call processing with {len(audio_chunks)} chunks")
        
        call_results = {
            'call_id': call_id,
            'start_time': time.time(),
            'chunks_processed': 0,
            'chunks_failed': 0,
            'chunk_results': [],
            'total_processing_time': 0,
            'status': 'processing'
        }
        
        try:
            # Create call record
            success, result = self.create_call_record(call_id, agent_id, customer_id, transcript_content)
            if not success:
                call_results['status'] = 'failed'
                call_results['error'] = f"Failed to create call record: {result}"
                return call_results
            
            # Process chunks sequentially for this call (but multiple calls run concurrently)
            for i, chunk in enumerate(audio_chunks):
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
                success, result = self.send_audio_chunk(
                    call_id, i, len(audio_chunks), client_audio, agent_audio, transcript_content
                )
                
                if success:
                    call_results['chunks_processed'] += 1
                    call_results['chunk_results'].append(result)
                    call_results['total_processing_time'] += result.get('request_time', 0)
                    
                    # Log progress
                    chunk_type = "FINAL" if i == len(audio_chunks) - 1 else "UPDATE"
                    adherence = result.get('overall_adherence', result.get('final_adherence', {}).get('overall', 0))
                    print(f"[CALL-{call_id}] Chunk {i+1}/{len(audio_chunks)} ({chunk_type}) - Adherence: {adherence:.1f}% - Time: {result.get('request_time', 0):.2f}s")
                else:
                    call_results['chunks_failed'] += 1
                    print(f"[CALL-{call_id}] Chunk {i+1} FAILED: {result}")
                
                # Small delay between chunks to avoid overwhelming the server
                if i < len(audio_chunks) - 1:
                    time.sleep(0.2)
            
            call_results['end_time'] = time.time()
            call_results['total_duration'] = call_results['end_time'] - call_results['start_time']
            call_results['status'] = 'completed' if call_results['chunks_failed'] == 0 else 'partial'
            
            print(f"[CALL-{call_id}] COMPLETED - {call_results['chunks_processed']}/{len(audio_chunks)} chunks successful")
            
        except Exception as e:
            call_results['status'] = 'failed'
            call_results['error'] = str(e)
            call_results['end_time'] = time.time()
            call_results['total_duration'] = call_results['end_time'] - call_results['start_time']
            print(f"[CALL-{call_id}] FAILED: {e}")
        
        return call_results
    
    def test_server_connection(self):
        """Test if the server is accessible"""
        try:
            session = self.get_session()
            response = session.get(f"{self.server_url}/health", timeout=10)
            self.return_session(session)
            return True
        except:
            try:
                session = self.get_session()
                response = session.get(f"{self.server_url}/", timeout=10)
                self.return_session(session)
                return True
            except:
                return False
    
    def run_concurrent_test(self, audio_file_path, num_calls, transcript_file_path=None, test_mode="both"):
        """Run concurrent call testing"""
        print("="*100)
        print(f"CONCURRENT CALL TESTING - {num_calls} SIMULTANEOUS CALLS")
        print("="*100)
        
        # Test server connection
        if not self.test_server_connection():
            print(f"[ERROR] Cannot connect to server at {self.server_url}")
            return False
        
        print(f"[INFO] Server connection successful")
        print(f"[INFO] Number of concurrent calls: {num_calls}")
        print(f"[INFO] Chunk duration: {self.chunk_duration/1000}s")
        print(f"[INFO] Test mode: {test_mode}")
        print(f"[INFO] Max workers: {self.max_workers}")
        
        # Load audio and transcript
        audio = self.load_audio(audio_file_path)
        if not audio:
            return False
        
        transcript_content = self.load_transcript(transcript_file_path)
        if transcript_content:
            print(f"[INFO] Transcript loaded: {len(transcript_content)} characters")
        else:
            print(f"[INFO] No transcript provided")
        
        # Create chunks
        chunks = self.create_chunks(audio)
        if not chunks:
            return False
        
        print(f"[INFO] Audio chunks created: {len(chunks)} chunks")
        
        # Prepare call configurations
        call_configs = []
        for i in range(num_calls):
            call_id = f"concurrent_test_{uuid.uuid4().hex[:8]}"
            call_config = {
                'call_id': call_id,
                'agent_id': f'agent_{i+1:03d}',
                'customer_id': f'customer_{i+1:03d}',
                'audio_chunks': chunks,
                'transcript_content': transcript_content,
                'test_mode': test_mode
            }
            call_configs.append(call_config)
        
        print(f"[INFO] Starting concurrent processing...")
        print("-" * 100)
        
        # Run concurrent calls
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all call processing tasks
            future_to_call = {
                executor.submit(self.process_single_call, config): config['call_id']
                for config in call_configs
            }
            
            # Collect results as they complete
            completed_calls = 0
            failed_calls = 0
            total_chunks_processed = 0
            total_processing_time = 0
            
            for future in as_completed(future_to_call):
                call_id = future_to_call[future]
                try:
                    result = future.result()
                    self.results[call_id] = result
                    
                    if result['status'] == 'completed':
                        completed_calls += 1
                    else:
                        failed_calls += 1
                    
                    total_chunks_processed += result['chunks_processed']
                    total_processing_time += result['total_processing_time']
                    
                    print(f"[COMPLETE] {call_id} - Status: {result['status']} - Chunks: {result['chunks_processed']}/{len(chunks)} - Duration: {result['total_duration']:.2f}s")
                    
                except Exception as e:
                    failed_calls += 1
                    print(f"[FAILED] {call_id} - Exception: {e}")
        
        end_time = time.time()
        total_test_duration = end_time - start_time
        
        # Generate comprehensive report
        print("="*100)
        print("CONCURRENT TEST RESULTS")
        print("="*100)
        print(f"[SUMMARY] Total calls: {num_calls}")
        print(f"[SUMMARY] Completed successfully: {completed_calls}")
        print(f"[SUMMARY] Failed: {failed_calls}")
        print(f"[SUMMARY] Success rate: {(completed_calls/num_calls*100):.1f}%")
        print(f"[SUMMARY] Total test duration: {total_test_duration:.2f}s")
        print(f"[SUMMARY] Average call duration: {total_test_duration/num_calls:.2f}s")
        print(f"[SUMMARY] Total chunks processed: {total_chunks_processed}")
        print(f"[SUMMARY] Total processing time: {total_processing_time:.2f}s")
        print(f"[SUMMARY] Average chunk processing time: {total_processing_time/total_chunks_processed:.2f}s" if total_chunks_processed > 0 else "[SUMMARY] No chunks processed")
        
        # Performance metrics
        chunks_per_second = total_chunks_processed / total_test_duration if total_test_duration > 0 else 0
        calls_per_second = num_calls / total_test_duration if total_test_duration > 0 else 0
        
        print(f"[PERFORMANCE] Chunks per second: {chunks_per_second:.2f}")
        print(f"[PERFORMANCE] Calls per second: {calls_per_second:.2f}")
        print(f"[PERFORMANCE] Concurrent efficiency: {(total_processing_time/total_test_duration):.2f}x")
        
        # Individual call results
        print("\n" + "="*100)
        print("INDIVIDUAL CALL RESULTS")
        print("="*100)
        
        for call_id, result in self.results.items():
            print(f"[{call_id}] Status: {result['status']}")
            print(f"  - Duration: {result['total_duration']:.2f}s")
            print(f"  - Chunks: {result['chunks_processed']}/{len(chunks)}")
            print(f"  - Processing time: {result['total_processing_time']:.2f}s")
            
            if result['chunk_results']:
                final_result = result['chunk_results'][-1]  # Last chunk (final_update)
                if 'final_adherence' in final_result:
                    adherence = final_result['final_adherence']
                    print(f"  - Final adherence: {adherence.get('overall', 0):.1f}%")
                    print(f"  - Script completion: {adherence.get('script_completion', 0):.1f}%")
                elif 'overall_adherence' in final_result:
                    print(f"  - Final adherence: {final_result['overall_adherence']:.1f}%")
                
                # Show emotion analysis
                if 'final_emotions' in final_result:
                    final_emotions = final_result['final_emotions']
                    print(f"  - Final CQS: {final_emotions.get('cqs', 0):.2f}")
                    print(f"  - Final Quality: {final_emotions.get('quality', 0):.1f}%")
                
                if 'chunk_emotions_count' in final_result:
                    print(f"  - Emotion chunks: {final_result['chunk_emotions_count']}")
            
            if result['status'] == 'failed' and 'error' in result:
                print(f"  - Error: {result['error']}")
            print()
        
        return completed_calls == num_calls

def main():
    parser = argparse.ArgumentParser(description='Test concurrent call processing with app2-segment.py')
    parser.add_argument('--audio_file', required=True, help='Path to audio file for testing')
    parser.add_argument('--num_calls', type=int, default=3, help='Number of concurrent calls to simulate (default: 3)')
    parser.add_argument('--server_url', default='http://localhost:5000', help='Server URL (default: http://localhost:5000)')
    parser.add_argument('--chunk_duration', type=int, default=10, help='Chunk duration in seconds (default: 10)')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum worker threads (default: 10)')
    parser.add_argument('--transcript', help='Path to transcript file for adherence testing')
    parser.add_argument('--test_mode', choices=['alternating', 'client_only', 'agent_only', 'both'], 
                        default='both', help='Audio distribution mode (default: both)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.num_calls < 1:
        print("[ERROR] Number of calls must be at least 1")
        return False
    
    if args.num_calls > 20:
        print("[WARNING] Testing with more than 20 concurrent calls may overwhelm the server")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create tester and run
    tester = ConcurrentCallTester(
        server_url=args.server_url,
        chunk_duration=args.chunk_duration,
        max_workers=args.max_workers
    )
    
    success = tester.run_concurrent_test(
        audio_file_path=args.audio_file,
        num_calls=args.num_calls,
        transcript_file_path=args.transcript,
        test_mode=args.test_mode
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
        import traceback
        traceback.print_exc()
        sys.exit(1) 