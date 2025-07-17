# Concurrent Call Testing Guide

## Overview
The `concurrent_test_script.py` is designed to test the system's ability to handle multiple simultaneous calls, validating the database-backed concurrent processing architecture and thread safety.

## Key Features

### 🔄 **Concurrent Processing**
- Simulates multiple calls running simultaneously
- Tests database thread safety with GridFS storage
- Validates resource management under load
- Measures performance metrics and efficiency

### 📊 **Comprehensive Metrics**
- Individual call success/failure rates
- Processing time analysis
- Chunks per second throughput
- Concurrent efficiency measurements
- Detailed error reporting

### 🎯 **Flexible Testing Modes**
- **Both**: Same audio to client and agent streams
- **Agent Only**: Tests agent script adherence
- **Client Only**: Tests client emotion analysis
- **Alternating**: Distributes chunks between streams

## Usage Examples

### Basic Concurrent Test
```bash
python concurrent_test_script.py --audio_file audiotest.wav --num_calls 3
```

### With Transcript for Adherence Testing
```bash
python concurrent_test_script.py \
    --audio_file audiotest.wav \
    --num_calls 5 \
    --transcript test_transcript.xml
```

### High Load Testing
```bash
python concurrent_test_script.py \
    --audio_file audiotest.wav \
    --num_calls 10 \
    --max_workers 15 \
    --chunk_duration 8 \
    --test_mode both
```

### Performance Benchmarking
```bash
python concurrent_test_script.py \
    --audio_file audiotest.wav \
    --num_calls 8 \
    --transcript test_transcript.xml \
    --server_url http://localhost:5000 \
    --max_workers 12 \
    --test_mode agent_only
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--audio_file` | string | required | Path to audio file for testing |
| `--num_calls` | int | 3 | Number of concurrent calls to simulate |
| `--server_url` | string | http://localhost:5000 | Server URL |
| `--chunk_duration` | int | 10 | Chunk duration in seconds |
| `--max_workers` | int | 10 | Maximum worker threads |
| `--transcript` | string | optional | Path to transcript file |
| `--test_mode` | choice | both | Audio distribution mode |

## Expected Output

### During Execution
```
====================================================================================================
CONCURRENT CALL TESTING - 5 SIMULTANEOUS CALLS
====================================================================================================
[INFO] Server connection successful
[INFO] Number of concurrent calls: 5
[INFO] Chunk duration: 10.0s
[INFO] Test mode: both
[INFO] Max workers: 10
[INFO] Transcript loaded: 2847 characters
[INFO] Audio chunks created: 4 chunks
[INFO] Starting concurrent processing...
----------------------------------------------------------------------------------------------------
[CALL-concurrent_test_a1b2c3d4] Starting call processing with 4 chunks
[CALL-concurrent_test_e5f6g7h8] Starting call processing with 4 chunks
[CALL-concurrent_test_i9j0k1l2] Starting call processing with 4 chunks
[CALL-concurrent_test_m3n4o5p6] Starting call processing with 4 chunks
[CALL-concurrent_test_q7r8s9t0] Starting call processing with 4 chunks

[CALL-concurrent_test_a1b2c3d4] Chunk 1/4 (UPDATE) - Adherence: 22.2% - Time: 2.34s
[CALL-concurrent_test_e5f6g7h8] Chunk 1/4 (UPDATE) - Adherence: 22.2% - Time: 2.41s
[CALL-concurrent_test_i9j0k1l2] Chunk 1/4 (UPDATE) - Adherence: 22.2% - Time: 2.38s
[CALL-concurrent_test_m3n4o5p6] Chunk 1/4 (UPDATE) - Adherence: 22.2% - Time: 2.45s
[CALL-concurrent_test_q7r8s9t0] Chunk 1/4 (UPDATE) - Adherence: 22.2% - Time: 2.51s

[CALL-concurrent_test_a1b2c3d4] Chunk 2/4 (UPDATE) - Adherence: 44.4% - Time: 2.12s
[CALL-concurrent_test_e5f6g7h8] Chunk 2/4 (UPDATE) - Adherence: 44.4% - Time: 2.18s
...

[CALL-concurrent_test_a1b2c3d4] Chunk 4/4 (FINAL) - Adherence: 77.8% - Time: 3.22s
[CALL-concurrent_test_e5f6g7h8] Chunk 4/4 (FINAL) - Adherence: 77.8% - Time: 3.18s
[CALL-concurrent_test_i9j0k1l2] Chunk 4/4 (FINAL) - Adherence: 77.8% - Time: 3.25s
[CALL-concurrent_test_m3n4o5p6] Chunk 4/4 (FINAL) - Adherence: 77.8% - Time: 3.31s
[CALL-concurrent_test_q7r8s9t0] Chunk 4/4 (FINAL) - Adherence: 77.8% - Time: 3.28s

[COMPLETE] concurrent_test_a1b2c3d4 - Status: completed - Chunks: 4/4 - Duration: 12.45s
[COMPLETE] concurrent_test_e5f6g7h8 - Status: completed - Chunks: 4/4 - Duration: 12.52s
[COMPLETE] concurrent_test_i9j0k1l2 - Status: completed - Chunks: 4/4 - Duration: 12.48s
[COMPLETE] concurrent_test_m3n4o5p6 - Status: completed - Chunks: 4/4 - Duration: 12.61s
[COMPLETE] concurrent_test_q7r8s9t0 - Status: completed - Chunks: 4/4 - Duration: 12.58s
```

### Final Results Summary
```
====================================================================================================
CONCURRENT TEST RESULTS
====================================================================================================
[SUMMARY] Total calls: 5
[SUMMARY] Completed successfully: 5
[SUMMARY] Failed: 0
[SUMMARY] Success rate: 100.0%
[SUMMARY] Total test duration: 12.61s
[SUMMARY] Average call duration: 2.52s
[SUMMARY] Total chunks processed: 20
[SUMMARY] Total processing time: 50.15s
[SUMMARY] Average chunk processing time: 2.51s
[PERFORMANCE] Chunks per second: 1.59
[PERFORMANCE] Calls per second: 0.40
[PERFORMANCE] Concurrent efficiency: 3.98x

====================================================================================================
INDIVIDUAL CALL RESULTS
====================================================================================================
[concurrent_test_a1b2c3d4] Status: completed
  - Duration: 12.45s
  - Chunks: 4/4
  - Processing time: 10.02s
  - Final adherence: 77.8%
  - Script completion: 88.9%

[concurrent_test_e5f6g7h8] Status: completed
  - Duration: 12.52s
  - Chunks: 4/4
  - Processing time: 10.08s
  - Final adherence: 77.8%
  - Script completion: 88.9%

[concurrent_test_i9j0k1l2] Status: completed
  - Duration: 12.48s
  - Chunks: 4/4
  - Processing time: 10.05s
  - Final adherence: 77.8%
  - Script completion: 88.9%

[concurrent_test_m3n4o5p6] Status: completed
  - Duration: 12.61s
  - Chunks: 4/4
  - Processing time: 10.15s
  - Final adherence: 77.8%
  - Script completion: 88.9%

[concurrent_test_q7r8s9t0] Status: completed
  - Duration: 12.58s
  - Chunks: 4/4
  - Processing time: 10.12s
  - Final adherence: 77.8%
  - Script completion: 88.9%
```

## Performance Metrics Explained

### **Concurrent Efficiency**
- **Formula**: `Total Processing Time / Total Test Duration`
- **Example**: `50.15s / 12.61s = 3.98x`
- **Meaning**: The system processed 3.98x more work concurrently than it would sequentially

### **Throughput Metrics**
- **Chunks per second**: Total chunks processed ÷ test duration
- **Calls per second**: Total calls ÷ test duration
- **Average processing time**: Individual chunk processing time

### **Success Metrics**
- **Success rate**: Percentage of calls completed successfully
- **Chunk completion**: Individual chunk success rates
- **Error analysis**: Detailed failure information

## Testing Scenarios

### 1. **Light Load Testing** (2-3 calls)
```bash
python concurrent_test_script.py --audio_file audiotest.wav --num_calls 2
```
- Tests basic concurrent functionality
- Validates thread safety
- Establishes baseline performance

### 2. **Medium Load Testing** (5-8 calls)
```bash
python concurrent_test_script.py --audio_file audiotest.wav --num_calls 6 --transcript test_transcript.xml
```
- Tests system under moderate load
- Validates database performance
- Tests transcript processing concurrency

### 3. **High Load Testing** (10+ calls)
```bash
python concurrent_test_script.py --audio_file audiotest.wav --num_calls 12 --max_workers 15
```
- Stress tests the system
- Identifies performance bottlenecks
- Tests resource management

### 4. **Endurance Testing**
```bash
python concurrent_test_script.py --audio_file audiotest.wav --num_calls 8 --chunk_duration 15
```
- Tests with longer processing times
- Validates memory management
- Tests connection pooling

## Troubleshooting

### **High Failure Rates**
- Reduce `--num_calls` or increase `--max_workers`
- Check server resource availability
- Verify database connection limits

### **Poor Performance**
- Monitor server CPU and memory usage
- Check database performance
- Adjust `--chunk_duration` for optimal balance

### **Connection Issues**
- Verify server is running and accessible
- Check firewall and network settings
- Increase timeout values if needed

### **Memory Issues**
- Reduce concurrent calls
- Use smaller audio files
- Monitor system memory usage

## System Requirements

### **Server Requirements**
- Sufficient CPU cores for concurrent processing
- Adequate RAM for multiple audio buffers
- MongoDB with enough connections
- GridFS storage capacity

### **Client Requirements**
- Python 3.7+
- Required packages: `requests`, `pydub`, `concurrent.futures`
- Audio file in supported format (WAV, MP3, etc.)

## Best Practices

1. **Start Small**: Begin with 2-3 concurrent calls
2. **Monitor Resources**: Watch server CPU, memory, and database performance
3. **Gradual Scaling**: Increase load gradually to find optimal limits
4. **Error Analysis**: Review individual call failures for patterns
5. **Performance Baselines**: Establish baseline metrics for comparison

This concurrent testing framework provides comprehensive validation of your system's ability to handle multiple simultaneous calls while maintaining accuracy and performance. 