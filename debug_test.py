#!/usr/bin/env python3
"""
Debug script to test the specific failing methods directly
"""

import sys
import time
from io import BytesIO
sys.path.append('.')

from call_operations import CallOperations
from db_config import get_db

def test_store_audio_chunk_and_process():
    """Test the store_audio_chunk_and_process method directly"""
    try:
        print("🔧 Testing store_audio_chunk_and_process method...")
        
        # Initialize database connection
        db = get_db()
        call_ops = CallOperations(db)
        
        # Create test call
        test_call_id = f"debug_test_{int(time.time())}"
        result = call_ops.create_call(test_call_id, 'agent1', 'customer1', 'test script')
        print(f"✅ Create call result: {result}")
        
        # Create minimal audio data
        audio_data = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00' + b'\x00\x00' * 1000
        
        client_audio = BytesIO(audio_data)
        agent_audio = BytesIO(audio_data)
        
        # Test store_audio_chunk_and_process
        print("🔄 Testing store_audio_chunk_and_process...")
        start_time = time.time()
        
        processing_data = call_ops.store_audio_chunk_and_process(test_call_id, client_audio, agent_audio)
        
        elapsed = time.time() - start_time
        print(f"✅ store_audio_chunk_and_process completed in {elapsed:.2f}s")
        print(f"📊 Processing data keys: {list(processing_data.keys())}")
        
        # Test insert_partial_update
        print("🔄 Testing insert_partial_update...")
        start_time = time.time()
        
        transcription = {'agent': 'Hello', 'client': 'Hi there'}
        call_ops.insert_partial_update(test_call_id, 10, 0.5, {}, {'neutral': 0.8}, transcription, 85)
        
        elapsed = time.time() - start_time
        print(f"✅ insert_partial_update completed in {elapsed:.2f}s")
        
        print("\n🎉 All methods tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_store_audio_chunk_and_process()
    sys.exit(0 if success else 1) 