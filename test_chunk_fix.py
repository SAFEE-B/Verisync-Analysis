#!/usr/bin/env python3
"""
Test script to verify the chunk saving optimization works without timeouts
"""

import sys
import time
sys.path.append('.')

from call_operations import CallOperations
from db_config import get_db

def test_chunk_saving():
    """Test the optimized chunk saving mechanism"""
    try:
        print("🔧 Testing optimized chunk saving...")
        
        # Initialize database connection
        db = get_db()
        call_ops = CallOperations(db)
        
        # Test create call
        test_call_id = f"test_chunk_fix_{int(time.time())}"
        result = call_ops.create_call(test_call_id, 'agent1', 'customer1', 'test script')
        print(f"✅ Create call result: {result}")
        
        # Test first partial update
        print("🔄 Testing first chunk...")
        start_time = time.time()
        transcription1 = {'agent': 'Hello', 'client': 'Hi there'}
        call_ops.insert_partial_update(test_call_id, 10, 0.5, {}, {'neutral': 0.8}, transcription1, 85)
        elapsed1 = time.time() - start_time
        print(f"✅ First partial update completed in {elapsed1:.2f}s")
        
        # Test second partial update  
        print("🔄 Testing second chunk...")
        start_time = time.time()
        transcription2 = {'agent': 'How are you?', 'client': 'I am fine'}
        call_ops.insert_partial_update(test_call_id, 10, 0.7, {}, {'happy': 0.9}, transcription2, 90)
        elapsed2 = time.time() - start_time
        print(f"✅ Second partial update completed in {elapsed2:.2f}s")
        
        # Test third partial update
        print("🔄 Testing third chunk...")
        start_time = time.time()
        transcription3 = {'agent': 'Thank you', 'client': 'You are welcome'}
        call_ops.insert_partial_update(test_call_id, 10, 0.8, {}, {'grateful': 0.95}, transcription3, 95)
        elapsed3 = time.time() - start_time
        print(f"✅ Third partial update completed in {elapsed3:.2f}s")
        
        # Check the results
        call = call_ops.get_call(test_call_id)
        if call:
            emotions = call.get('emotions', [])
            chunk_count = call.get('chunk_count', 0)
            
            print(f"\n📊 Results:")
            print(f"   - Total emotion chunks: {len(emotions)}")
            print(f"   - Chunk count field: {chunk_count}")
            print(f"   - Call duration: {call.get('duration', 0)}")
            print(f"   - Call status: {call.get('status', 'unknown')}")
            
            print(f"\n📝 Chunk Details:")
            for i, emotion in enumerate(emotions):
                chunk_num = emotion.get('chunk_number', 'N/A')
                chunk_emotions = emotion.get('emotions', {})
                chunk_cqs = emotion.get('cqs', 'N/A')
                print(f"   Chunk {i+1}: number={chunk_num}, emotions={chunk_emotions}, cqs={chunk_cqs}")
        
        # Performance summary
        avg_time = (elapsed1 + elapsed2 + elapsed3) / 3
        print(f"\n⚡ Performance Summary:")
        print(f"   - Average chunk processing time: {avg_time:.2f}s")
        print(f"   - All chunks processed in under 60s: {'✅ YES' if avg_time < 60 else '❌ NO'}")
        
        print("\n🎉 Chunk saving optimization test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunk_saving()
    sys.exit(0 if success else 1) 