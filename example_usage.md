# Audio Test Script Usage Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r test_requirements.txt
   ```

2. **Make sure app2.py is running:**
   ```bash
   python app2.py
   ```
   The server should be running on `http://localhost:5000`

## Basic Usage

### Simple Test (Both Streams Mode)
```bash
python audio_test_script.py --audio_file your_audio.wav
```

This will:
- Split your audio into 10-second chunks
- Send each chunk to both client audio and agent audio
- Use `/update_call` for intermediate chunks
- Use `/final_update` for the last chunk

### Custom Chunk Duration
```bash
python audio_test_script.py --audio_file your_audio.wav --chunk_duration 5
```

### Different Test Modes

**Client Only:** Send all chunks as client audio
```bash
python audio_test_script.py --audio_file your_audio.wav --test_mode client_only
```

**Agent Only:** Send all chunks as agent audio
```bash
python audio_test_script.py --audio_file your_audio.wav --test_mode agent_only
```

**Both:** Send same chunk to both client and agent
```bash
python audio_test_script.py --audio_file your_audio.wav --test_mode both
```

### With Custom Script
```bash
python audio_test_script.py --audio_file your_audio.wav --script audioscript.txt
```

### Custom Server URL
```bash
python audio_test_script.py --audio_file your_audio.wav --server_url http://192.168.1.100:5000
```

### Custom Call ID
```bash
python audio_test_script.py --audio_file your_audio.wav --call_id my_test_call_001
```

## Complete Example
```bash
python audio_test_script.py \
  --audio_file conversation.wav \
  --server_url http://localhost:5000 \
  --chunk_duration 8 \
  --test_mode alternating \
  --call_id production_test_123 \
  --script custom_script.txt
```

## Output Format

The script provides detailed logging:

```
============================================================
AUDIO CHUNKING TEST SCRIPT
============================================================
[TEST] Testing connection to http://localhost:5000...
[TEST] Server responded with status 404
[INFO] Call ID: test_call_a1b2c3d4
[INFO] Test mode: alternating
[INFO] Chunk duration: 10.0s
[LOAD] Loading audio file: conversation.wav
[LOAD] Audio loaded successfully:
[LOAD] - Duration: 45.23 seconds
[LOAD] - Sample rate: 44100 Hz
[LOAD] - Channels: 1
[LOAD] - Sample width: 2 bytes
[CHUNK] Creating 10.0s chunks...
[CHUNK] Created 5 chunks
[CHUNK] - Chunk 1: 10.00s
[CHUNK] - Chunk 2: 10.00s
[CHUNK] - Chunk 3: 10.00s
[CHUNK] - Chunk 4: 10.00s
[CHUNK] - Chunk 5: 5.23s
[INFO] Processing 5 chunks...
----------------------------------------
[SEND] Sending chunk 1/5 (UPDATE) to /update_call
[SEND] - Call ID: test_call_a1b2c3d4
[SEND] - Client audio: ✓
[SEND] - Agent audio: ✗
[SUCCESS] Chunk 1 processed in 2.34s
[RESULT] - Adherence: 75.5%
[RESULT] - Script completion: 33.3%
[RESULT] - CQS: 0.85
[RESULT] - Duration: 10.00s
...
[FINAL] Final Analysis Complete:
[FINAL] - Real-time score: 82.1%
[FINAL] - Script completion: 91.7%
[FINAL] - Analysis method: adaptive_windows
============================================================
TEST SUMMARY
============================================================
[SUMMARY] Total chunks: 5
[SUMMARY] Successful: 5
[SUMMARY] Failed: 0
[SUMMARY] Success rate: 100.0%
[SUMMARY] Call ID: test_call_a1b2c3d4
```

## Error Handling

The script handles various error scenarios:
- Missing audio files
- Server connection failures
- Invalid audio formats
- API timeouts
- Server errors

## Supported Audio Formats

Thanks to pydub, the script supports various audio formats:
- WAV
- MP3
- M4A
- FLAC
- AAC
- And many others

Just make sure you have the appropriate codecs installed on your system. 