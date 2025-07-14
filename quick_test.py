#!/usr/bin/env python3
"""
Quick test script to verify the audio_test_script.py works correctly
"""

import os
import sys
import subprocess
import tempfile
from pydub import AudioSegment
from pydub.generators import Sine

def create_test_audio(duration_seconds=30, filename="test_audio.wav"):
    """Create a simple test audio file"""
    print(f"[TEST] Creating test audio file: {filename}")
    
    # Create a simple sine wave tone
    sample_rate = 16000
    frequency = 440  # A4 note
    
    # Generate sine wave
    sine_wave = Sine(frequency, sample_rate=sample_rate)
    audio = sine_wave.to_audio_segment(duration=duration_seconds * 1000)
    
    # Export as WAV
    audio.export(filename, format="wav")
    
    print(f"[TEST] Created {filename} ({duration_seconds}s, {sample_rate}Hz)")
    return filename

def test_server_connection(server_url="http://localhost:5000"):
    """Test if the server is running"""
    import requests
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return True
    except:
        try:
            # Try a different approach - just check if server responds
            response = requests.get(f"{server_url}/", timeout=5)
            return True
        except:
            return False

def run_audio_test(audio_file, call_id="quick_test_call"):
    """Run the audio test script"""
    cmd = [
        sys.executable, "audio_test_script.py",
        "--audio_file", audio_file,
        "--call_id", call_id,
        "--test_mode", "both",
        "--chunk_duration", "5"
    ]
    
    print(f"[TEST] Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print("[TEST] STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("[TEST] STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("[TEST] Command timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"[TEST] Error running command: {e}")
        return False

def main():
    print("="*50)
    print("QUICK TEST FOR AUDIO TESTING SCRIPT")
    print("="*50)
    
    # Check if server is running
    print("[TEST] Checking if app2.py server is running...")
    if not test_server_connection():
        print("[ERROR] Server is not running. Please start app2.py first:")
        print("  python app2.py")
        return False
    
    print("[TEST] Server is running ✓")
    
    # Create test audio file
    test_audio_file = create_test_audio(20)  # 20 second test audio
    
    try:
        # Run the audio test
        print("\n[TEST] Running audio test...")
        success = run_audio_test(test_audio_file)
        
        if success:
            print("\n[SUCCESS] Audio test completed successfully!")
            print("[INFO] Check your MongoDB database for the test call record")
            return True
        else:
            print("\n[FAILURE] Audio test failed")
            return False
            
    finally:
        # Clean up test file
        if os.path.exists(test_audio_file):
            os.remove(test_audio_file)
            print(f"[CLEANUP] Removed test file: {test_audio_file}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 