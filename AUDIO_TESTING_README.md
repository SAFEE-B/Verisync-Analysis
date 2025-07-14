# Audio Testing Suite for app2.py

This testing suite allows you to simulate real-time call analysis by chunking audio files and sending them to your app2.py endpoints.

## 📁 Files Overview

- **`audio_test_script.py`** - Main testing script that chunks audio and sends to API
- **`create_test_audio.py`** - Generates synthetic audio files for testing
- **`test_requirements.txt`** - Dependencies for the testing scripts
- **`example_usage.md`** - Detailed usage examples and documentation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r test_requirements.txt
```

### 2. Start Your Server
```bash
python app2.py
```
Make sure it's running on `http://localhost:5000`

### 3. Create Test Audio (Optional)
If you don't have an audio file, generate one:
```bash
# Create a 30-second test file with different tones
python create_test_audio.py --duration 30 --output test_audio.wav

# Or create a conversation-style audio
python create_test_audio.py --type conversation --duration 45 --output conversation.wav
```

### 4. Run the Test
```bash
# Basic test with your audio file
python audio_test_script.py --audio_file your_audio.wav

# Or with generated test audio
python audio_test_script.py --audio_file test_audio.wav
```

## 🔧 Advanced Usage

### Different Test Modes

**Both Streams Mode (Default):**
- Same chunk → both client_audio and agent_audio
```bash
python audio_test_script.py --audio_file audio.wav --test_mode both
```

**Alternating Mode:**
- Even chunks (0,2,4...) → client_audio
- Odd chunks (1,3,5...) → agent_audio
```bash
python audio_test_script.py --audio_file audio.wav --test_mode alternating
```

**Client Only:**
- All chunks → client_audio
```bash
python audio_test_script.py --audio_file audio.wav --test_mode client_only
```

**Agent Only:**
- All chunks → agent_audio  
```bash
python audio_test_script.py --audio_file audio.wav --test_mode agent_only
```

### Custom Configurations

**Custom Chunk Duration:**
```bash
python audio_test_script.py --audio_file audio.wav --chunk_duration 5
```

**Custom Server URL:**
```bash
python audio_test_script.py --audio_file audio.wav --server_url http://192.168.1.100:5000
```

**Custom Call ID:**
```bash
python audio_test_script.py --audio_file audio.wav --call_id my_test_001
```

**With Custom Script:**
```bash
python audio_test_script.py --audio_file audio.wav --script audioscript.txt
```

## 📊 Understanding the Output

The test script provides detailed real-time feedback:

```
============================================================
AUDIO CHUNKING TEST SCRIPT
============================================================
[TEST] Testing connection to http://localhost:5000...
[TEST] Server responded with status 404
[DB] Creating initial call record in database...
[DB] Created call entry: 67a2b8c9d4e1f2a3b4c5d6e7
[DB] Successfully created call record: 67a2b8c9d4e1f2a3b4c5d6e7
[INFO] Call ID: 67a2b8c9d4e1f2a3b4c5d6e7
[INFO] Test mode: both
[INFO] Chunk duration: 10.0s
[LOAD] Loading audio file: test_audio.wav
[LOAD] Audio loaded successfully:
[LOAD] - Duration: 30.00 seconds
[LOAD] - Sample rate: 44100 Hz
[LOAD] - Channels: 1
[LOAD] - Sample width: 2 bytes
[CHUNK] Creating 10.0s chunks...
[CHUNK] Created 3 chunks
[CHUNK] - Chunk 1: 10.00s
[CHUNK] - Chunk 2: 10.00s  
[CHUNK] - Chunk 3: 10.00s
[INFO] Processing 3 chunks...
----------------------------------------
[SEND] Sending chunk 1/3 (UPDATE) to /update_call
[SEND] - Call ID: 67a2b8c9d4e1f2a3b4c5d6e7
[SEND] - Client audio: ✓
[SEND] - Agent audio: ✓
[SUCCESS] Chunk 1 processed in 2.34s
[RESULT] - Adherence: 45.2%
[RESULT] - Script completion: 25.0%
[RESULT] - CQS: 0.78
[RESULT] - Duration: 10.00s
----------------------------------------
[SEND] Sending chunk 2/3 (UPDATE) to /update_call
[SEND] - Call ID: 67a2b8c9d4e1f2a3b4c5d6e7
[SEND] - Client audio: ✓
[SEND] - Agent audio: ✓
[SUCCESS] Chunk 2 processed in 2.15s
[RESULT] - Adherence: 67.3%
[RESULT] - Script completion: 50.0%
[RESULT] - CQS: 0.82
[RESULT] - Duration: 10.00s
----------------------------------------
[SEND] Sending chunk 3/3 (FINAL) to /final_update
[SEND] - Call ID: 67a2b8c9d4e1f2a3b4c5d6e7
[SEND] - Client audio: ✓
[SEND] - Agent audio: ✓
[SUCCESS] Chunk 3 processed in 3.45s
[RESULT] - Adherence: 78.9%
[RESULT] - Script completion: 75.0%
[RESULT] - CQS: 0.86
[RESULT] - Duration: 10.00s
[FINAL] Final Analysis Complete:
[FINAL] - Real-time score: 78.9%
[FINAL] - Script completion: 75.0%
[FINAL] - Analysis method: adaptive_windows
============================================================
TEST SUMMARY
============================================================
[SUMMARY] Total chunks: 3
[SUMMARY] Successful: 3
[SUMMARY] Failed: 0
[SUMMARY] Success rate: 100.0%
[SUMMARY] Call ID: 67a2b8c9d4e1f2a3b4c5d6e7
```

## 🎯 Key Features

### 1. **Realistic Simulation**
- Creates proper database records before testing
- Mimics real-time streaming behavior
- Proper endpoint routing (`/update_call` → `/final_update`)
- Sends same audio to both client and agent streams (default)

### 2. **Comprehensive Testing**
- Multiple test modes for different scenarios
- Custom script support
- Configurable chunk sizes
- Error handling and recovery

### 3. **Detailed Monitoring**
- Real-time performance metrics
- Adherence scoring progression
- Script completion tracking
- Request timing analysis

### 4. **Flexible Audio Support**
- WAV, MP3, M4A, FLAC, AAC support
- Automatic format detection
- Synthetic audio generation
- Custom duration and patterns

## 🔍 What Gets Tested

### API Functionality
- ✅ Database call record creation
- ✅ Audio file upload handling
- ✅ Transcription processing
- ✅ Script adherence analysis
- ✅ CQS calculation
- ✅ Emotion detection
- ✅ Database storage and updates
- ✅ Progressive analysis updates

### Performance Metrics
- ✅ Request processing time
- ✅ Memory usage patterns
- ✅ Concurrent request handling
- ✅ Error recovery
- ✅ SBERT optimization

### Analysis Quality
- ✅ Adaptive windows functionality
- ✅ Script completion tracking
- ✅ Real-time adherence scores
- ✅ Custom script parsing
- ✅ Checkpoint matching

## ⚠️ Important Notes

### Before Testing
1. **Start app2.py first** - The server must be running
2. **Check dependencies** - Run `pip install -r test_requirements.txt`
3. **Verify MongoDB** - Make sure your database is accessible
4. **Audio format** - Most formats supported, WAV recommended

### During Testing
- The script waits 0.5s between chunks to avoid overwhelming the server
- Each request has a 60-second timeout
- Failed chunks are retried once automatically
- Connection issues are logged with suggested fixes

### MongoDB Integration
- Script automatically creates initial call record in database
- Each test gets a unique MongoDB ObjectId as call_id
- Audio files are stored in the database with progressive updates
- Transcription and analysis data updated with each chunk
- Final analysis completion on last chunk
- Proper agent_id tracking for test identification

## 🐛 Troubleshooting

### Connection Errors
```
[ERROR] Cannot connect to server at http://localhost:5000
```
**Solution:** Make sure app2.py is running: `python app2.py`

### Audio Loading Errors
```
[ERROR] Failed to load audio file: [Errno 2] No such file or directory
```
**Solution:** Check file path and permissions, or create test audio:
```bash
python create_test_audio.py --output test.wav
```

### Memory Issues
```
[ERROR] Chunk X timed out after 60 seconds
```
**Solution:** 
- Reduce chunk duration: `--chunk_duration 5`
- Check server resources
- Ensure SBERT optimizations are working

### Format Issues
```
[ERROR] Failed to convert audio to WAV bytes
```
**Solution:** Install additional codecs or convert to WAV first

## 🚦 Test Scenarios

### Scenario 1: Basic Functionality
```bash
python create_test_audio.py --duration 20
python audio_test_script.py --audio_file test_audio.wav --chunk_duration 10
```

### Scenario 2: High-Frequency Updates
```bash
python audio_test_script.py --audio_file audio.wav --chunk_duration 3
```

### Scenario 3: Different Stream Types
```bash
# Test both streams (default)
python audio_test_script.py --audio_file audio.wav --test_mode both

# Test alternating mode
python audio_test_script.py --audio_file audio.wav --test_mode alternating

# Test client-only analysis
python audio_test_script.py --audio_file audio.wav --test_mode client_only

# Test agent-only analysis  
python audio_test_script.py --audio_file audio.wav --test_mode agent_only
```

### Scenario 4: Custom Script Testing
```bash
python audio_test_script.py --audio_file audio.wav --script audioscript.txt
```

### Scenario 5: Performance Testing
```bash
# Create longer audio for stress testing
python create_test_audio.py --duration 120 --output long_test.wav

# Test with small chunks for high update frequency
python audio_test_script.py --audio_file long_test.wav --chunk_duration 2
```

This testing suite provides comprehensive validation of your call analysis system's functionality, performance, and reliability! 🎉 