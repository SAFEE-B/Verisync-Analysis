#!/usr/bin/env python3
"""
Create Test Audio File

This script creates a simple test audio file with different tones
that can be used to test the audio_test_script.py

Usage:
    python create_test_audio.py [--duration 30] [--output test_audio.wav]
"""

import argparse
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

def create_test_audio(duration_seconds=30, output_file="test_audio.wav"):
    """
    Create a test audio file with different tones and silence
    
    Structure:
    - 0-5s: 440 Hz tone (A4)
    - 5-10s: 523 Hz tone (C5) 
    - 10-15s: Silence
    - 15-20s: 659 Hz tone (E5)
    - 20-25s: 784 Hz tone (G5)
    - 25-30s: Mixed tones
    """
    
    print(f"[CREATE] Creating {duration_seconds}s test audio file: {output_file}")
    
    # Create silence as base
    audio = AudioSegment.silent(duration=duration_seconds * 1000)
    
    # Calculate segment duration
    segment_duration = duration_seconds * 1000 // 6  # 6 segments
    
    try:
        # Segment 1: 440 Hz (A4)
        tone1 = Sine(440).to_audio_segment(duration=segment_duration)
        audio = audio.overlay(tone1, position=0)
        print(f"[CREATE] Added 440Hz tone at 0-{segment_duration/1000:.1f}s")
        
        # Segment 2: 523 Hz (C5)
        tone2 = Sine(523).to_audio_segment(duration=segment_duration)
        audio = audio.overlay(tone2, position=segment_duration)
        print(f"[CREATE] Added 523Hz tone at {segment_duration/1000:.1f}-{2*segment_duration/1000:.1f}s")
        
        # Segment 3: Silence (already there)
        print(f"[CREATE] Silence at {2*segment_duration/1000:.1f}-{3*segment_duration/1000:.1f}s")
        
        # Segment 4: 659 Hz (E5)
        tone3 = Sine(659).to_audio_segment(duration=segment_duration)
        audio = audio.overlay(tone3, position=3*segment_duration)
        print(f"[CREATE] Added 659Hz tone at {3*segment_duration/1000:.1f}-{4*segment_duration/1000:.1f}s")
        
        # Segment 5: 784 Hz (G5)
        tone4 = Sine(784).to_audio_segment(duration=segment_duration)
        audio = audio.overlay(tone4, position=4*segment_duration)
        print(f"[CREATE] Added 784Hz tone at {4*segment_duration/1000:.1f}-{5*segment_duration/1000:.1f}s")
        
        # Segment 6: Mixed tones (chord)
        remaining_duration = duration_seconds * 1000 - 5 * segment_duration
        if remaining_duration > 0:
            # Create a chord (C major: C-E-G)
            chord_c = Sine(523).to_audio_segment(duration=remaining_duration) * 0.3  # C5
            chord_e = Sine(659).to_audio_segment(duration=remaining_duration) * 0.3  # E5
            chord_g = Sine(784).to_audio_segment(duration=remaining_duration) * 0.3  # G5
            chord = chord_c.overlay(chord_e).overlay(chord_g)
            audio = audio.overlay(chord, position=5*segment_duration)
            print(f"[CREATE] Added chord at {5*segment_duration/1000:.1f}-{duration_seconds:.1f}s")
        
        # Normalize volume
        audio = audio.normalize()
        
        # Export to file
        audio.export(output_file, format="wav")
        
        print(f"[SUCCESS] Test audio file created: {output_file}")
        print(f"[INFO] Duration: {len(audio)/1000:.2f} seconds")
        print(f"[INFO] Sample rate: {audio.frame_rate} Hz")
        print(f"[INFO] Channels: {audio.channels}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create test audio: {e}")
        return False

def create_conversation_audio(duration_seconds=45, output_file="conversation_test.wav"):
    """
    Create a more realistic conversation-like audio file
    with alternating speaker simulation
    """
    
    print(f"[CREATE] Creating {duration_seconds}s conversation audio file: {output_file}")
    
    # Create silence as base
    audio = AudioSegment.silent(duration=duration_seconds * 1000)
    
    # Define "speaker" frequencies
    speaker_a_freq = 350  # Lower pitch (male voice simulation)
    speaker_b_freq = 450  # Higher pitch (female voice simulation)
    
    try:
        # Simulate conversation turns
        turns = [
            (0, 3, speaker_a_freq),      # Speaker A: 0-3s
            (3.5, 6, speaker_b_freq),    # Speaker B: 3.5-6s
            (6.5, 10, speaker_a_freq),   # Speaker A: 6.5-10s
            (10.5, 13, speaker_b_freq),  # Speaker B: 10.5-13s
            (13.5, 18, speaker_a_freq),  # Speaker A: 13.5-18s
            (18.5, 21, speaker_b_freq),  # Speaker B: 18.5-21s
            (21.5, 26, speaker_a_freq),  # Speaker A: 21.5-26s
            (26.5, 29, speaker_b_freq),  # Speaker B: 26.5-29s
            (29.5, 34, speaker_a_freq),  # Speaker A: 29.5-34s
            (34.5, 37, speaker_b_freq),  # Speaker B: 34.5-37s
            (37.5, 42, speaker_a_freq),  # Speaker A: 37.5-42s
            (42.5, 45, speaker_b_freq),  # Speaker B: 42.5-45s
        ]
        
        for start_time, end_time, frequency in turns:
            if end_time > duration_seconds:
                end_time = duration_seconds
            
            duration_ms = (end_time - start_time) * 1000
            if duration_ms > 0:
                # Create a tone with slight frequency modulation to sound more natural
                tone = Sine(frequency).to_audio_segment(duration=duration_ms)
                
                # Add some volume variation
                tone = tone * 0.7  # Reduce volume a bit
                
                # Add to audio
                audio = audio.overlay(tone, position=start_time * 1000)
                
                speaker = "A" if frequency == speaker_a_freq else "B"
                print(f"[CREATE] Speaker {speaker} ({frequency}Hz): {start_time:.1f}-{end_time:.1f}s")
        
        # Normalize volume
        audio = audio.normalize()
        
        # Export to file
        audio.export(output_file, format="wav")
        
        print(f"[SUCCESS] Conversation audio file created: {output_file}")
        print(f"[INFO] Duration: {len(audio)/1000:.2f} seconds")
        print(f"[INFO] This simulates a conversation with alternating speakers")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create conversation audio: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create test audio files for testing')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds (default: 30)')
    parser.add_argument('--output', default='test_audio.wav', help='Output filename (default: test_audio.wav)')
    parser.add_argument('--type', choices=['tones', 'conversation'], default='tones', 
                        help='Type of test audio (default: tones)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("TEST AUDIO GENERATOR")
    print("="*50)
    
    try:
        if args.type == 'tones':
            success = create_test_audio(args.duration, args.output)
        else:
            success = create_conversation_audio(args.duration, args.output)
        
        if success:
            print("\n" + "="*50)
            print("READY TO TEST")
            print("="*50)
            print(f"You can now test with:")
            print(f"python audio_test_script.py --audio_file {args.output}")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Audio generation interrupted by user")
        exit(1) 