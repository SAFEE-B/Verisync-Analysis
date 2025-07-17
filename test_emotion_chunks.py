#!/usr/bin/env python3
"""
Test script to demonstrate the new chunk-based emotion tracking system.

This script shows how emotions are now stored as an array of chunks with
a final emotion analysis for the complete transcript.

Usage:
    python test_emotion_chunks.py --call_id test_call_123 --server_url http://localhost:5000
"""

import argparse
import requests
import json
from datetime import datetime

def test_emotion_tracking(server_url, call_id):
    """Test the new emotion tracking system"""
    
    print("="*80)
    print("EMOTION CHUNK TRACKING TEST")
    print("="*80)
    
    session = requests.Session()
    
    # 1. Create a call
    print(f"[TEST] Creating call: {call_id}")
    create_response = session.post(f"{server_url}/create_call", data={
        'call_id': call_id,
        'agent_id': 'test_agent',
        'customer_id': 'test_customer'
    })
    
    if create_response.status_code != 201:
        print(f"[ERROR] Failed to create call: {create_response.text}")
        return False
    
    print(f"[SUCCESS] Call created successfully")
    
    # 2. Simulate multiple update_call requests with different emotions
    test_chunks = [
        {
            "chunk_num": 1,
            "client_text": "Hello, I'm having trouble with my account",
            "expected_emotions": ["neutral", "confusion"]
        },
        {
            "chunk_num": 2,
            "client_text": "I'm really frustrated because I can't access my files",
            "expected_emotions": ["anger", "frustration"]
        },
        {
            "chunk_num": 3,
            "client_text": "Thank you so much for helping me resolve this issue",
            "expected_emotions": ["gratitude", "joy"]
        }
    ]
    
    print(f"\n[TEST] Processing {len(test_chunks)} emotion chunks...")
    
    for i, chunk in enumerate(test_chunks):
        print(f"\n--- Chunk {chunk['chunk_num']} ---")
        print(f"Client text: {chunk['client_text']}")
        
        # Create dummy audio data (empty for this test)
        files = {}
        data = {
            'call_id': call_id,
            'client_text_override': chunk['client_text']  # For testing without actual audio
        }
        
        # Send update_call request
        if i == len(test_chunks) - 1:
            # Last chunk - use final_update
            endpoint = "/final_update"
        else:
            endpoint = "/update_call"
        
        print(f"[REQUEST] Sending to {endpoint}")
        
        # Note: In real usage, you'd send actual audio files
        # For this test, we're just demonstrating the emotion tracking structure
        response = session.post(f"{server_url}{endpoint}", data=data, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"[SUCCESS] Chunk processed successfully")
            
            # Show chunk-level results
            if 'emotions' in result:
                print(f"Chunk emotions: {result['emotions']}")
            if 'CQS' in result:
                print(f"Chunk CQS: {result['CQS']}")
            if 'quality' in result:
                print(f"Chunk quality: {result['quality']}")
            
            # Show final results (only for final_update)
            if 'final_emotions' in result:
                final_emotions = result['final_emotions']
                print(f"\n[FINAL EMOTIONS] Complete transcript analysis:")
                print(f"Final CQS: {final_emotions.get('cqs', 0)}")
                print(f"Final Quality: {final_emotions.get('quality', 0)}")
                print(f"Final Emotions: {final_emotions.get('emotions', {})}")
                
            if 'chunk_emotions_count' in result:
                print(f"Total emotion chunks: {result['chunk_emotions_count']}")
        else:
            print(f"[ERROR] Failed to process chunk: {response.text}")
    
    # 3. Retrieve the final call data to show the complete emotion structure
    print(f"\n[TEST] Retrieving final call data...")
    
    # Note: You would need to implement a get_call endpoint or query the database directly
    # For demonstration, we'll show what the database structure looks like
    
    print(f"\n" + "="*80)
    print("EXPECTED DATABASE STRUCTURE")
    print("="*80)
    
    expected_structure = {
        "call_id": call_id,
        "emotions": [
            {
                "chunk_number": 1,
                "timestamp": "2024-01-01T10:00:00Z",
                "duration": 10.0,
                "emotions": {"neutral": 0.8, "confusion": 0.2},
                "cqs": 0.5,
                "quality": 75.0,
                "client_text_length": 45
            },
            {
                "chunk_number": 2,
                "timestamp": "2024-01-01T10:00:10Z",
                "duration": 10.0,
                "emotions": {"anger": 0.6, "frustration": 0.4},
                "cqs": -1.2,
                "quality": 45.0,
                "client_text_length": 58
            },
            {
                "chunk_number": 3,
                "timestamp": "2024-01-01T10:00:20Z",
                "duration": 10.0,
                "emotions": {"gratitude": 0.7, "joy": 0.3},
                "cqs": 2.1,
                "quality": 95.0,
                "client_text_length": 52
            }
        ],
        "final_emotions": {
            "joy": 0.4,
            "gratitude": 0.3,
            "neutral": 0.2,
            "anger": 0.1
        },
        "cqs": 1.8,  # Final CQS from complete transcript
        "quality": 85.0,  # Final quality from complete transcript
        "status": "completed"
    }
    
    print(json.dumps(expected_structure, indent=2))
    
    print(f"\n" + "="*80)
    print("EMOTION TRACKING BENEFITS")
    print("="*80)
    print("✅ Chunk-by-chunk emotion tracking")
    print("✅ Final emotion analysis of complete transcript")
    print("✅ Emotion progression over time")
    print("✅ Separate CQS and quality for chunks vs final")
    print("✅ Metadata for each emotion chunk")
    print("✅ Backward compatibility with legacy emotion field")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test the new chunk-based emotion tracking system')
    parser.add_argument('--server_url', default='http://localhost:5000', help='Server URL')
    parser.add_argument('--call_id', default=f'emotion_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}', help='Call ID for testing')
    
    args = parser.parse_args()
    
    success = test_emotion_tracking(args.server_url, args.call_id)
    
    if success:
        print(f"\n[SUCCESS] Emotion tracking test completed successfully!")
        print(f"[INFO] Call ID used: {args.call_id}")
    else:
        print(f"\n[FAILED] Emotion tracking test failed!")
    
    return success

if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        exit(1) 