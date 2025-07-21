#!/usr/bin/env python3
"""
Parallel Audio Chunking Test Script

This script launches multiple simultaneous audio test calls using the logic from audio_test_script.py.

Usage:
    python audio_test_parallel.py --audio_file path/to/audio.wav --num_calls 10 [--server_url http://localhost:5000] [--chunk_duration 10] [--test_mode both]

It will report per-call and aggregate stats at the end.
"""
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import bson
from audio_test_script import AudioChunkTester


def run_single_call(call_index, args, custom_script=None):
    """Run a single simulated call with a unique call_id."""
    call_id = f"parallel_test_{call_index}_{uuid.uuid4().hex[:6]}"
    tester = AudioChunkTester(args.server_url, args.chunk_duration)
    success = tester.run_test(
        args.audio_file,
        call_id=call_id,
        transcript_content=custom_script,
        test_mode=args.test_mode,
        agent_id=str(bson.ObjectId()),
        sip_id=f"sip_{call_index}",
        include_silent=args.include_silent
    )
    return {
        'call_id': call_id,
        'success': success
    }

def main():
    parser = argparse.ArgumentParser(description='Run multiple parallel audio chunking test calls.')
    parser.add_argument('--audio_file', required=True, help='Path to WAV audio file')
    parser.add_argument('--num_calls', type=int, default=1, help='Number of parallel calls to simulate (default: 5)')
    parser.add_argument('--server_url', default='http://localhost:5000', help='Server URL (default: http://localhost:5000)')
    parser.add_argument('--chunk_duration', type=int, default=10, help='Chunk duration in seconds (default: 10)')
    parser.add_argument('--test_mode', choices=['alternating', 'client_only', 'agent_only', 'both'], 
                        default='both', help='How to distribute chunks (default: both)')
    parser.add_argument('--script', help='Path to custom script file in XML format')
    parser.add_argument('--max_workers', type=int, default=10, help='Max parallel threads (default: 10)')
    parser.add_argument('--include_silent', action='store_true', 
                        help='Include silent 10-second chunks alternating with audio chunks')
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

    print(f"[PARALLEL TEST] Launching {args.num_calls} parallel calls...")
    if args.include_silent:
        print(f"[PARALLEL TEST] Including silent chunks between audio chunks")
    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_single_call, i, args, custom_script) for i in range(args.num_calls)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[CALL RESULT] Call {result['call_id']}: {'SUCCESS' if result['success'] else 'FAIL'}")

    total_time = time.time() - start_time
    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes
    print("="*60)
    print(f"[SUMMARY] Total calls: {len(results)}")
    print(f"[SUMMARY] Successes: {successes}")
    print(f"[SUMMARY] Failures: {failures}")
    print(f"[SUMMARY] Total elapsed time: {total_time:.2f}s")
    print("="*60)
    return successes == len(results)

if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        exit(1) 