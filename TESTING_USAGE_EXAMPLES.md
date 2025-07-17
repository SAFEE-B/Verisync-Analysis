# Audio Test Script Usage Examples

## Overview
The updated `audio_test_script.py` now supports sending transcript data for script adherence analysis. This enables real-time testing of the complete call analysis pipeline including script compliance checking.

## Basic Usage

### 1. Simple Audio Test (No Transcript)
```bash
python audio_test_script.py --audio_file audiotest.wav
```

### 2. Audio Test with Transcript
```bash
python audio_test_script.py --audio_file audiotest.wav --transcript test_transcript.xml
```

### 3. Custom Server and Settings
```bash
python audio_test_script.py \
    --audio_file audiotest.wav \
    --transcript test_transcript.xml \
    --server_url http://localhost:5000 \
    --chunk_duration 15 \
    --call_id my_test_call_001
```

## Test Modes

### 1. Both Streams (Default)
Sends same audio chunk to both client and agent streams:
```bash
python audio_test_script.py --audio_file audiotest.wav --test_mode both
```

### 2. Agent Only
Sends audio only to agent stream (useful for testing agent script adherence):
```bash
python audio_test_script.py --audio_file audiotest.wav --test_mode agent_only
```

### 3. Client Only
Sends audio only to client stream (useful for testing client emotion analysis):
```bash
python audio_test_script.py --audio_file audiotest.wav --test_mode client_only
```

### 4. Alternating
Alternates chunks between client and agent streams:
```bash
python audio_test_script.py --audio_file audiotest.wav --test_mode alternating
```

## Complete Example with All Parameters

```bash
python audio_test_script.py \
    --audio_file audiotest.wav \
    --transcript test_transcript.xml \
    --server_url http://localhost:5000 \
    --chunk_duration 10 \
    --call_id test_call_with_transcript \
    --agent_id agent_sarah_001 \
    --customer_id customer_john_002 \
    --test_mode both
```

## Expected Output

### During Execution
```
================================================================================
AUDIO CHUNKING TEST SCRIPT WITH TRANSCRIPT SUPPORT
================================================================================
[TEST] Testing connection to http://localhost:5000...
[TEST] Server is running (404 for /health is expected)
[API] Generated call_id: test_call_a1b2c3d4
[API] Creating call record in database...
[API] Including transcript in call creation (2847 chars)
[API] Successfully created call record: test_call_a1b2c3d4
[TRANSCRIPT] Loading transcript file: test_transcript.xml
[TRANSCRIPT] Transcript loaded successfully:
[TRANSCRIPT] - File size: 2847 characters
[TRANSCRIPT] - First 100 chars: <?xml version="1.0" encoding="UTF-8"?>
<script>
    <checkpoint id="q1">
        <prompt_text>Thank...
[INFO] Call ID: test_call_a1b2c3d4
[INFO] Test mode: both
[INFO] Chunk duration: 10.0s
[INFO] Transcript: 2847 characters loaded
[LOAD] Loading audio file: audiotest.wav
[LOAD] Audio loaded successfully:
[LOAD] - Duration: 45.23 seconds
[LOAD] - Sample rate: 16000 Hz
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
------------------------------------------------------------
[SEND] Sending chunk 1/5 (UPDATE) to /update_call
[SEND] - Call ID: test_call_a1b2c3d4
[SEND] - Client audio: ✓
[SEND] - Agent audio: ✓
[SEND] - Transcript: 2847 chars
[SUCCESS] Chunk 1 processed in 2.34s
[RESULT] - Adherence: 25.0%
[RESULT] - Script completion: 11.1%
[RESULT] - CQS: 2.8
[RESULT] - Quality: 85.2%
[RESULT] - Checkpoints found: 1/9
[RESULT] - Analysis method: adaptive_windows_segments
[RESULT] - Window usage: S:5 M:3 L:1
------------------------------------------------------------
[SEND] Sending chunk 2/5 (UPDATE) to /update_call
[SEND] - Call ID: test_call_a1b2c3d4
[SEND] - Client audio: ✓
[SEND] - Agent audio: ✓
[SEND] - Transcript: 2847 chars
[SUCCESS] Chunk 2 processed in 2.12s
[RESULT] - Adherence: 44.4%
[RESULT] - Script completion: 33.3%
[RESULT] - CQS: 3.1
[RESULT] - Quality: 87.5%
[RESULT] - Checkpoints found: 3/9
[RESULT] - Analysis method: adaptive_windows_segments
[RESULT] - Window usage: S:8 M:5 L:2
------------------------------------------------------------
...
[SEND] Sending chunk 5/5 (FINAL) to /final_update
[SEND] - Call ID: test_call_a1b2c3d4
[SEND] - Client audio: ✓
[SEND] - Agent audio: ✓
[SEND] - Transcript: 2847 chars
[SUCCESS] Chunk 5 processed in 3.45s
[FINAL] Final Analysis Complete:
[FINAL] - Overall adherence: 77.8%
[FINAL] - Script completion: 88.9%
[FINAL] - Analysis method: adaptive_windows_segments
[FINAL] - Total checkpoints: 9
[FINAL] - Window usage: S:15 M:12 L:6
[FINAL] - Checkpoints found: 7/9
[FINAL] - Total duration: 45.23s
------------------------------------------------------------
================================================================================
TEST SUMMARY
================================================================================
[SUMMARY] Total chunks: 5
[SUMMARY] Successful: 5
[SUMMARY] Failed: 0
[SUMMARY] Success rate: 100.0%
[SUMMARY] Call ID: test_call_a1b2c3d4
[SUMMARY] Test mode: both
[SUMMARY] Transcript: 2847 characters processed
[SUMMARY] Database record created and updated successfully
```

## API Payload Details

### `/create_call` Payload
```
call_id: "test_call_a1b2c3d4"
agent_id: "agent_sarah_001"
customer_id: "customer_john_002"
transcript: "<?xml version=\"1.0\"...full XML content..."
```

### `/update_call` Payload
```
call_id: "test_call_a1b2c3d4"
transcript: "<?xml version=\"1.0\"...full XML content..."
client_audio: [WAV file data]
agent_audio: [WAV file data]
```

### `/final_update` Payload
```
call_id: "test_call_a1b2c3d4"
transcript: "<?xml version=\"1.0\"...full XML content..."
client_audio: [WAV file data]
agent_audio: [WAV file data]
```

## Key Features

1. **Transcript Support**: Automatically loads and sends transcript data with each request
2. **Real-time Adherence**: Shows adherence scores and checkpoint progress during processing
3. **Comprehensive Logging**: Detailed output showing all processing steps
4. **Flexible Test Modes**: Multiple ways to distribute audio between client/agent streams
5. **Error Handling**: Graceful handling of connection issues and file errors
6. **Database Integration**: Creates proper database records and updates them progressively

## Troubleshooting

### Server Connection Issues
- Ensure `app2-segment.py` is running on the specified URL
- Check that the server is accessible (404 for `/health` is expected)

### Transcript Loading Issues
- Verify the transcript file exists and is readable
- Check XML formatting is valid
- Ensure file encoding is UTF-8

### Audio Processing Issues
- Verify audio file format is supported (WAV, MP3, etc.)
- Check file permissions and accessibility
- Ensure audio file is not corrupted

## Files Created by This Example
- `test_transcript.xml`: Sample XML transcript for testing
- `TESTING_USAGE_EXAMPLES.md`: This usage guide
- Updated `audio_test_script.py`: Enhanced test script with transcript support 