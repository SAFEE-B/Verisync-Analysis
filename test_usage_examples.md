# Audio Testing Script Usage Guide

This guide explains how to use the updated `audio_test_script.py` to test your `app2.py` call analysis system.

## Prerequisites

1. **Install Dependencies**: Make sure you have the required packages:
   ```bash
   pip install pydub requests
   ```

2. **Start the Server**: Run your `app2.py` server:
   ```bash
   python app2.py
   ```

3. **Prepare Audio File**: You need a WAV audio file for testing.

## Basic Usage

### Simple Test with Default Settings
```bash
python audio_test_script.py --audio_file path/to/your/audio.wav
```

This will:
- Generate a unique call ID automatically
- Split the audio into 10-second chunks
- Send the same chunk to both client and agent audio streams
- Use the default script from `audioscript.txt` or built-in script
- Create a complete database record with all analysis data

### Test with Custom Call ID
```bash
python audio_test_script.py --audio_file audio.wav --call_id my_test_call_123
```

### Test with Custom Script
```bash
python audio_test_script.py --audio_file audio.wav --script my_custom_script.xml
```

### Test Different Audio Routing Modes

**Both streams (default)**: Same audio sent to both client and agent
```bash
python audio_test_script.py --audio_file audio.wav --test_mode both
```

**Alternating**: Alternate chunks between client and agent
```bash
python audio_test_script.py --audio_file audio.wav --test_mode alternating
```

**Client only**: All chunks sent as client audio only
```bash
python audio_test_script.py --audio_file audio.wav --test_mode client_only
```

**Agent only**: All chunks sent as agent audio only
```bash
python audio_test_script.py --audio_file audio.wav --test_mode agent_only
```

### Custom Server URL
```bash
python audio_test_script.py --audio_file audio.wav --server_url http://localhost:5001
```

### Custom Chunk Duration
```bash
python audio_test_script.py --audio_file audio.wav --chunk_duration 15
```

### Complete Example with All Options
```bash
python audio_test_script.py \
  --audio_file test_call.wav \
  --call_id production_test_001 \
  --agent_id agent_john_doe \
  --customer_id customer_12345 \
  --script custom_script.xml \
  --test_mode both \
  --chunk_duration 8 \
  --server_url http://localhost:5000
```

## What Gets Stored in Database

The script will create and update a complete call record in your MongoDB database with:

### Initial Call Record (`/create_call`)
- `call_id`: Unique identifier for the call
- `agent_id`: Agent identifier
- `customer_id`: Customer identifier  
- `transcript`: Script text for adherence checking
- `created_at`: Timestamp

### Ongoing Updates (`/update_call`)
- Audio files (stored via GridFS)
- Real-time transcriptions
- Adherence scores and analysis
- Emotion detection results
- Quality metrics

### Final Analysis (`/final_update`)
- Complete call summary
- Final adherence scores
- Agent performance evaluation
- Conversation tags
- Total duration and quality metrics

## Expected Output

The script provides detailed logging showing:

```
============================================================
AUDIO CHUNKING TEST SCRIPT
============================================================
[TEST] Testing connection to http://localhost:5000...
[TEST] Server is running (404 for /health is expected)
[API] Generated call_id: test_call_a1b2c3d4
[API] Creating call record in database...
[API] Successfully created call record: test_call_a1b2c3d4
[INFO] Call ID: test_call_a1b2c3d4
[INFO] Test mode: both
[INFO] Chunk duration: 10.0s
[LOAD] Loading audio file: test_audio.wav
[LOAD] Audio loaded successfully:
[LOAD] - Duration: 45.23 seconds
[LOAD] - Sample rate: 16000 Hz
[LOAD] - Channels: 1
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
[SEND] - Agent audio: ✓
[SUCCESS] Chunk 1 processed in 2.34s
[RESULT] - Adherence: 75.2%
[RESULT] - Script completion: 60.0%
[RESULT] - CQS: 2.45
[RESULT] - Quality: 85.3%
----------------------------------------
...
[FINAL] Final Analysis Complete:
[FINAL] - Real-time score: 82.1%
[FINAL] - Script completion: 80.0%
[FINAL] - Analysis method: adaptive_windows
[FINAL] - Total checkpoints: 12
[FINAL] - Total duration: 45.23s
============================================================
TEST SUMMARY
============================================================
[SUMMARY] Total chunks: 5
[SUMMARY] Successful: 5
[SUMMARY] Failed: 0
[SUMMARY] Success rate: 100.0%
[SUMMARY] Call ID: test_call_a1b2c3d4
[SUMMARY] Database record created and updated successfully
```

## Troubleshooting

### Common Issues

1. **Server Connection Failed**
   - Make sure `app2.py` is running
   - Check the server URL
   - Verify the port is correct

2. **Audio File Not Found**
   - Check the file path
   - Ensure the file exists and is readable
   - Supported formats: WAV, MP3, M4A, etc. (pydub handles conversion)

3. **Database Connection Issues**
   - Check MongoDB connection in `app2.py`
   - Verify environment variables are set
   - Check database permissions

4. **Script Parsing Errors**
   - Ensure custom script follows XML format
   - Check for proper tag structure: `<strict>`, `<semantic>`, `<topic>`

### Debug Mode

For more detailed logging, you can modify the script to add debug prints or check the server logs in `app2.py`.

## Database Verification

After running the test, you can verify the database record was created by checking your MongoDB:

```javascript
// Connect to your MongoDB and check the calls collection
db.calls.findOne({call_id: "test_call_a1b2c3d4"})
```

This should show a complete call record with all the analysis data, audio references, and metrics. 