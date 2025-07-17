import os
from pymongo import MongoClient
from gridfs import GridFS
from datetime import datetime, timezone
import json
from typing import Optional, Dict, Any, List
from bson import ObjectId
import threading

class MongoDB:
    """MongoDB connection and operations manager with GridFS support for audio storage"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.fs = None
        self.calls_collection = None
        self._gridfs_lock = threading.Lock()  # Thread safety for GridFS operations
        self.connect()
    
    def connect(self):
        """Establish MongoDB connection"""
        try:
            # Get MongoDB connection details from environment
            mongo_uri = os.getenv('MONGODB_URI')
            db_name = os.getenv('MONGODB_DB_NAME', 'verisync_analysis')
            
            if mongo_uri:
                # Use MongoDB URI (for cloud databases like MongoDB Atlas)
                self.client = MongoClient(mongo_uri)
            else:
                # Use individual connection parameters
                host = os.getenv('MONGODB_HOST', 'localhost')
                port = int(os.getenv('MONGODB_PORT', 27017))
                username = os.getenv('MONGODB_USER')
                password = os.getenv('MONGODB_PASS')
                
                if username and password:
                    self.client = MongoClient(
                        host=host,
                        port=port,
                        username=username,
                        password=password,
                        authSource=db_name
                    )
                else:
                    self.client = MongoClient(host=host, port=port)
            
            # Select database
            self.db = self.client[db_name]
            
            # Initialize GridFS for audio file storage
            self.fs = GridFS(self.db)
            
            # Get collections
            self.calls_collection = self.db.calls
            
            # Test connection
            self.client.admin.command('ping')
            print(f"[DB] Successfully connected to MongoDB: {db_name}")
            
            # Create indexes for better performance
            self._create_indexes()
            
        except Exception as e:
            print(f"[DB ERROR] Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            # Index for calls collection
            self.calls_collection.create_index("call_id", unique=True)
            self.calls_collection.create_index("agent_id")
            self.calls_collection.create_index("date")
            self.calls_collection.create_index("status")
            
            print("[DB] Database indexes created successfully")
        except Exception as e:
            print(f"[DB WARNING] Failed to create indexes: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("[DB] MongoDB connection closed")
    
    def _combine_wav_audio(self, existing_audio: bytes, new_audio: bytes) -> bytes:
        """
        Properly combine two WAV audio streams using torchaudio
        
        Args:
            existing_audio: Existing audio bytes
            new_audio: New audio bytes to append
            
        Returns:
            Combined audio bytes
        """
        if not new_audio:
            return existing_audio
        if not existing_audio:
            return new_audio
            
        try:
            import torchaudio
            import torch
            from io import BytesIO
            
            # Load both audio streams
            existing_buffer = BytesIO(existing_audio)
            new_buffer = BytesIO(new_audio)
            
            existing_waveform, sr1 = torchaudio.load(existing_buffer)
            new_waveform, sr2 = torchaudio.load(new_buffer)
            
            # Handle sample rate mismatch
            if sr1 != sr2:
                print(f"[DB_AUDIO] Sample rate mismatch: {sr1} vs {sr2}. Resampling to {sr1}.")
                resampler = torchaudio.transforms.Resample(sr2, sr1)
                new_waveform = resampler(new_waveform)
            
            # Concatenate waveforms
            combined_waveform = torch.cat((existing_waveform, new_waveform), dim=1)
            
            # Save back to WAV format
            output_buffer = BytesIO()
            torchaudio.save(output_buffer, combined_waveform, sr1, format="wav")
            output_buffer.seek(0)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            print(f"[DB_AUDIO_ERROR] Failed to combine WAV audio: {e}. Using raw concatenation.")
            # Fallback to raw concatenation (may cause issues with WAV headers)
            return existing_audio + new_audio

    def store_or_append_audio(self, call_id: str, audio_data: bytes, 
                             audio_type: str) -> str:
        """
        Store or append audio data for a call with thread-safe operations
        
        Args:
            call_id: Unique call identifier
            audio_data: Binary audio data to append
            audio_type: 'agent' or 'client'
        
        Returns:
            GridFS file ID as string
        """
        try:
            filename = f"{call_id}_{audio_type}.wav"
            
            # Thread-safe GridFS operations using lock (GridFS doesn't support transactions)
            with self._gridfs_lock:
                # Check if audio file already exists
                existing_file = self.fs.find_one({"filename": filename})
                
                if existing_file:
                    # Read existing audio data
                    existing_data = existing_file.read()
                    # Properly combine WAV audio streams
                    combined_audio = self._combine_wav_audio(existing_data, audio_data)
                    # Delete old file
                    self.fs.delete(existing_file._id)
                else:
                    # First audio for this call and type
                    combined_audio = audio_data
                
                # Store the combined audio
                file_id = self.fs.put(
                    combined_audio,
                    filename=filename,
                    metadata={
                        "call_id": call_id,
                        "audio_type": audio_type,
                        "timestamp": datetime.now(timezone.utc),
                        "content_type": "audio/wav",
                        "size": len(combined_audio),
                        "chunk_count": existing_file.metadata.get("chunk_count", 0) + 1 if existing_file else 1
                    }
                )
            
            return str(file_id)
            
        except Exception as e:
            print(f"[DB ERROR] Failed to store/append audio: {e}")
            raise

    def get_audio_slice(self, call_id: str, audio_type: str, start_time: float, end_time: float = None) -> Optional[bytes]:
        """
        Get a slice of audio from the stored audio file
        
        Args:
            call_id: Unique call identifier
            audio_type: 'agent' or 'client'
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of file)
        
        Returns:
            Sliced audio bytes or None if not found
        """
        try:
            import torchaudio
            from io import BytesIO
            
            print(f"[AUDIO_SLICE_DEBUG] Attempting to slice {call_id}_{audio_type} from {start_time:.2f}s to {end_time or 'end'}")
            
            # Get the full audio file
            full_audio = self.get_call_audio(call_id, audio_type)
            if not full_audio:
                print(f"[AUDIO_SLICE_DEBUG] No audio found for {call_id}_{audio_type}")
                return None
            
            print(f"[AUDIO_SLICE_DEBUG] Full audio size: {len(full_audio)} bytes")
            
            # Load audio and slice it
            buffer = BytesIO(full_audio)
            waveform, sample_rate = torchaudio.load(buffer)
            
            print(f"[AUDIO_SLICE_DEBUG] Audio loaded: {waveform.shape[1]} frames, {sample_rate} Hz")
            print(f"[AUDIO_SLICE_DEBUG] Audio duration: {waveform.shape[1] / sample_rate:.2f} seconds")
            
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate) if end_time is not None else waveform.shape[1]
            
            print(f"[AUDIO_SLICE_DEBUG] Slice frames: {start_frame} to {end_frame}")
            print(f"[AUDIO_SLICE_DEBUG] Slice duration: {(end_frame - start_frame) / sample_rate:.2f} seconds")
            
            # Ensure slice indices are within bounds
            original_start = start_frame
            original_end = end_frame
            start_frame = max(0, start_frame)
            end_frame = min(waveform.shape[1], end_frame)
            
            if original_start != start_frame or original_end != end_frame:
                print(f"[AUDIO_SLICE_DEBUG] Adjusted slice bounds: {start_frame} to {end_frame}")
            
            if start_frame >= end_frame:
                print(f"[AUDIO_SLICE_DEBUG] Invalid slice: start_frame ({start_frame}) >= end_frame ({end_frame})")
                return None
                
            sliced_waveform = waveform[:, start_frame:end_frame]
            print(f"[AUDIO_SLICE_DEBUG] Sliced waveform shape: {sliced_waveform.shape}")
            
            # Convert back to bytes
            sliced_buffer = BytesIO()
            torchaudio.save(sliced_buffer, sliced_waveform, sample_rate, format="wav")
            sliced_buffer.seek(0)
            
            result = sliced_buffer.getvalue()
            print(f"[AUDIO_SLICE_DEBUG] Slice result: {len(result)} bytes")
            
            return result
            
        except Exception as e:
            print(f"[AUDIO_SLICE_ERROR] Failed to slice audio {call_id}_{audio_type}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def store_call_segments(self, call_id: str, audio_type: str, segments: List[Dict]) -> bool:
        """
        DEPRECATED: Segments are now stored directly in the call document's conversation array.
        This method is kept for backward compatibility but does nothing.
        """
        # Segments are now stored in the main call document's conversation array
        # This eliminates the need for separate collections and improves efficiency
        return True

    def get_call_segments(self, call_id: str, audio_type: str) -> List[Dict]:
        """
        DEPRECATED: Segments are now retrieved from the call document's conversation array.
        This method is kept for backward compatibility.
        """
        try:
            # Get segments from the main call document's conversation array
            call_doc = self.calls_collection.find_one({"call_id": call_id})
            if not call_doc:
                return []
            
            conversation = call_doc.get('conversation', [])
            segments = [entry for entry in conversation if entry.get('speaker') == audio_type]
            
            # Convert to expected format
            formatted_segments = []
            for segment in segments:
                formatted_segment = {
                    'text': segment.get('text', ''),
                    'start': segment.get('start_time', 0),
                    'end': segment.get('end_time', 0),
                    'confidence': segment.get('confidence', 1.0)
                }
                formatted_segments.append(formatted_segment)
            
            return formatted_segments
            
        except Exception as e:
            return []

    def get_last_segment_time(self, call_id: str, audio_type: str) -> float:
        """
        DEPRECATED: Last segment time is now computed from the call document's conversation array.
        This method is kept for backward compatibility.
        """
        try:
            call_doc = self.calls_collection.find_one({"call_id": call_id})
            if not call_doc:
                return 0.0
            
            conversation = call_doc.get('conversation', [])
            speaker_segments = [entry for entry in conversation if entry.get('speaker') == audio_type]
            
            if speaker_segments:
                # Get the last segment by start_time
                last_segment = max(speaker_segments, key=lambda x: x.get('start_time', 0))
                return last_segment.get('start_time', 0.0)
            
            return 0.0
            
        except Exception as e:
            return 0.0

    def cleanup_call_data(self, call_id: str):
        """
        Clean up call-related data including audio files.
        Segments are now part of the main call document so no separate cleanup needed.
        """
        try:
            # Delete audio files
            self.delete_call_audio(call_id)
            
            # NOTE: Segments are now stored in the main call document's conversation array
            # so no separate cleanup is needed. This improves efficiency by eliminating
            # the need to manage multiple collections.
            
        except Exception as e:
            pass
    
    def get_call_audio(self, call_id: str, audio_type: str) -> Optional[bytes]:
        """Retrieve complete audio for a call by type"""
        try:
            filename = f"{call_id}_{audio_type}.wav"
            grid_out = self.fs.find_one({"filename": filename})
            if grid_out:
                return grid_out.read()
            return None
        except Exception as e:
            print(f"[DB ERROR] Failed to retrieve audio {filename}: {e}")
            return None

    def get_call_audio_info(self, call_id: str) -> Dict:
        """Get audio file information for a call"""
        try:
            audio_info = {"agent": None, "client": None}
            
            for audio_type in ["agent", "client"]:
                filename = f"{call_id}_{audio_type}.wav"
                file_info = self.fs.find_one({"filename": filename})
                if file_info:
                    audio_info[audio_type] = {
                        "file_id": str(file_info._id),
                        "filename": filename,
                        "size": file_info.metadata.get("size", file_info.length),
                        "timestamp": file_info.metadata.get("timestamp", file_info.uploadDate)
                    }
            
            return audio_info
        except Exception as e:
            print(f"[DB ERROR] Failed to get audio info for call {call_id}: {e}")
            return {"agent": None, "client": None}

    def delete_call_audio(self, call_id: str):
        """Delete all audio files for a call"""
        try:
            deleted_count = 0
            for audio_type in ["agent", "client"]:
                filename = f"{call_id}_{audio_type}.wav"
                file_info = self.fs.find_one({"filename": filename})
                if file_info:
                    self.fs.delete(file_info._id)
                    deleted_count += 1
            
        except Exception as e:
            print(f"[DB ERROR] Failed to delete audio files for call {call_id}: {e}")

# =============================================================================
# OPTIMIZED SCHEMA UTILITY FUNCTIONS
# =============================================================================

def compute_agent_text(conversation: List[Dict]) -> str:
    """Compute agent text from conversation array"""
    return " ".join([
        segment['text'] for segment in conversation 
        if segment.get('speaker') == 'agent'
    ]).strip()

def compute_client_text(conversation: List[Dict]) -> str:
    """Compute client text from conversation array"""
    return " ".join([
        segment['text'] for segment in conversation 
        if segment.get('speaker') == 'client'
    ]).strip()

def compute_combined_text(conversation: List[Dict]) -> str:
    """Compute combined text from conversation array"""
    agent_text = compute_agent_text(conversation)
    client_text = compute_client_text(conversation)
    return f"Agent: {agent_text}\nClient: {client_text}".strip()

def compute_timestamped_dialogue(conversation: List[Dict]) -> List[Dict]:
    """Convert conversation to timestamped dialogue format"""
    return [
        {
            'timestamp': segment.get('start_time', 0),
            'speaker': segment.get('speaker', '').title(),
            'text': segment.get('text', '')
        }
        for segment in conversation
    ]

def compute_agent_segments(conversation: List[Dict]) -> List[Dict]:
    """Extract agent segments from conversation"""
    return [
        {
            'text': segment['text'],
            'start': segment.get('start_time', 0),
            'end': segment.get('end_time', 0)
        }
        for segment in conversation
        if segment.get('speaker') == 'agent'
    ]

def compute_client_segments(conversation: List[Dict]) -> List[Dict]:
    """Extract client segments from conversation"""
    return [
        {
            'text': segment['text'],
            'start': segment.get('start_time', 0),
            'end': segment.get('end_time', 0)
        }
        for segment in conversation
        if segment.get('speaker') == 'client'
    ]

def create_conversation_entry(speaker: str, text: str, start_time: float, 
                            end_time: float, segment_id: str = None, 
                            confidence: float = 1.0) -> Dict:
    """Create a standardized conversation entry"""
    return {
        'speaker': speaker.lower(),
        'text': text.strip(),
        'start_time': start_time,
        'end_time': end_time,
        'segment_id': segment_id or f"seg_{int(start_time * 1000)}",
        'confidence': confidence
    }

def build_conversation_from_segments(agent_segments: List[Dict], 
                                   client_segments: List[Dict]) -> List[Dict]:
    """Build conversation array from separate agent and client segments"""
    conversation = []
    
    # Add agent segments
    for i, segment in enumerate(agent_segments):
        conversation.append(create_conversation_entry(
            speaker='agent',
            text=segment.get('text', ''),
            start_time=segment.get('start', 0),
            end_time=segment.get('end', 0),
            confidence=segment.get('confidence', 1.0)
        ))
    
    # Add client segments
    for i, segment in enumerate(client_segments):
        conversation.append(create_conversation_entry(
            speaker='client',
            text=segment.get('text', ''),
            start_time=segment.get('start', 0),
            end_time=segment.get('end', 0),
            confidence=segment.get('confidence', 1.0)
        ))
    
    # Sort by start time
    conversation.sort(key=lambda x: x['start_time'])
    
    return conversation

# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================

def get_transcription_legacy_format(conversation: List[Dict]) -> Dict:
    """Get transcription in legacy format for backward compatibility"""
    return {
        "agent": compute_agent_text(conversation),
        "client": compute_client_text(conversation),
        "combined": compute_combined_text(conversation),
        "timestamped_dialogue": compute_timestamped_dialogue(conversation),
        "agent_segments": compute_agent_segments(conversation),
        "client_segments": compute_client_segments(conversation)
    }

# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

def migrate_call_to_optimized_schema(call_doc: Dict) -> Dict:
    """Migrate a single call document from old schema to optimized schema"""
    try:
        # Skip if already using optimized schema
        if "conversation" in call_doc and call_doc["conversation"]:
            return call_doc
            
        # Extract segments from old transcription format
        transcription = call_doc.get("transcription", {})
        agent_segments = transcription.get("agent_segments", [])
        client_segments = transcription.get("client_segments", [])
        
        # Build optimized conversation array
        conversation = build_conversation_from_segments(agent_segments, client_segments)
        
        # Calculate audio metadata
        total_segments = len(conversation)
        total_speech_duration = sum(
            segment.get('end_time', 0) - segment.get('start_time', 0) 
            for segment in conversation
        )
        
        # Update call document
        call_doc["conversation"] = conversation
        call_doc["audio_metadata"] = {
            "agent_file_id": None,
            "client_file_id": None, 
            "total_segments": total_segments,
            "total_speech_duration": total_speech_duration
        }
        
        print(f"[MIGRATION] Migrated call {call_doc.get('call_id', 'unknown')} with {total_segments} segments")
        return call_doc
        
    except Exception as e:
        print(f"[MIGRATION ERROR] Failed to migrate call {call_doc.get('call_id', 'unknown')}: {e}")
        return call_doc

def migrate_all_calls_to_optimized_schema(db: MongoDB, batch_size: int = 100):
    """Migrate all calls in the database to optimized schema"""
    try:
        calls_collection = db.calls_collection
        
        # Count total calls to migrate
        total_calls = calls_collection.count_documents({})
        print(f"[MIGRATION] Starting migration of {total_calls} calls...")
        
        migrated_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process calls in batches
        for skip in range(0, total_calls, batch_size):
            batch = calls_collection.find({}).skip(skip).limit(batch_size)
            
            for call_doc in batch:
                try:
                    # Skip if already migrated
                    if "conversation" in call_doc and call_doc["conversation"]:
                        skipped_count += 1
                        continue
                        
                    # Migrate the call
                    migrated_call = migrate_call_to_optimized_schema(call_doc)
                    
                    # Update in database
                    calls_collection.update_one(
                        {"_id": call_doc["_id"]},
                        {"$set": {
                            "conversation": migrated_call["conversation"],
                            "audio_metadata": migrated_call["audio_metadata"]
                        }}
                    )
                    
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"[MIGRATION ERROR] Failed to migrate call {call_doc.get('call_id', 'unknown')}: {e}")
                    error_count += 1
                    
            print(f"[MIGRATION] Processed {min(skip + batch_size, total_calls)}/{total_calls} calls...")
        
        print(f"[MIGRATION] Migration completed!")
        print(f"[MIGRATION] - Migrated: {migrated_count}")
        print(f"[MIGRATION] - Skipped: {skipped_count}")
        print(f"[MIGRATION] - Errors: {error_count}")
        
        return {
            "total_calls": total_calls,
            "migrated": migrated_count,
            "skipped": skipped_count,
            "errors": error_count
        }
        
    except Exception as e:
        print(f"[MIGRATION ERROR] Migration failed: {e}")
        return {"error": str(e)}

def analyze_schema_usage(db: MongoDB) -> Dict:
    """Analyze current schema usage in the database"""
    try:
        calls_collection = db.calls_collection
        
        # Count calls by schema type
        total_calls = calls_collection.count_documents({})
        
        if total_calls == 0:
            return {
                "total_calls": 0,
                "optimized_calls": 0,
                "legacy_calls": 0,
                "optimization_percentage": 0,
                "storage_analysis": {
                    "avg_transcription_size_bytes": 0,
                    "avg_conversation_size_bytes": 0,
                    "potential_savings_bytes": 0,
                    "potential_savings_percentage": 0
                }
            }
        
        optimized_calls = calls_collection.count_documents({"conversation": {"$exists": True, "$ne": []}})
        legacy_calls = total_calls - optimized_calls
        
        # Simple storage analysis without complex aggregation
        try:
            # Sample a few documents to estimate sizes
            sample_docs = list(calls_collection.find({}).limit(min(10, total_calls)))
            
            transcription_sizes = []
            conversation_sizes = []
            
            for doc in sample_docs:
                # Calculate transcription size
                transcription = doc.get("transcription", {})
                if transcription:
                    # Estimate size based on text length
                    agent_text = transcription.get("agent", "")
                    client_text = transcription.get("client", "")
                    combined_text = transcription.get("combined", "")
                    transcription_size = len(agent_text) + len(client_text) + len(combined_text)
                    transcription_sizes.append(transcription_size)
                
                # Calculate conversation size
                conversation = doc.get("conversation", [])
                if conversation:
                    conversation_size = sum(len(entry.get("text", "")) for entry in conversation)
                    conversation_sizes.append(conversation_size)
                else:
                    conversation_sizes.append(0)
            
            # Calculate averages
            transcription_size = sum(transcription_sizes) / len(transcription_sizes) if transcription_sizes else 0
            conversation_size = sum(conversation_sizes) / len(conversation_sizes) if conversation_sizes else 0
            potential_savings = max(0, transcription_size - conversation_size)
            savings_percentage = (potential_savings / transcription_size * 100) if transcription_size > 0 else 0
            
        except Exception as analysis_error:
            print(f"[ANALYSIS WARNING] Storage analysis failed: {analysis_error}")
            # Fallback to basic analysis without size calculations
            transcription_size = 0
            conversation_size = 0
            potential_savings = 0
            savings_percentage = 0
        
        return {
            "total_calls": total_calls,
            "optimized_calls": optimized_calls,
            "legacy_calls": legacy_calls,
            "optimization_percentage": (optimized_calls / total_calls * 100) if total_calls > 0 else 0,
            "storage_analysis": {
                "avg_transcription_size_bytes": transcription_size,
                "avg_conversation_size_bytes": conversation_size,
                "potential_savings_bytes": potential_savings,
                "potential_savings_percentage": savings_percentage
            }
        }
        
    except Exception as e:
        print(f"[ANALYSIS ERROR] Schema analysis failed: {e}")
        return {"error": str(e)}

# Global MongoDB instance
mongodb = MongoDB()

def get_db() -> MongoDB:
    """Get the global MongoDB instance"""
    return mongodb

# =============================================================================
# PROPOSED OPTIMIZED SCHEMA
# =============================================================================
"""
OPTIMIZED CALL DOCUMENT SCHEMA:

{
  "_id": ObjectId("..."),
  "call_id": "unique_call_identifier",
  "agent_id": "agent_identifier",
  "customer_id": "customer_identifier", 
  "date": ISODate("2024-01-01T00:00:00Z"),
  "status": "initiated|in_progress|completed",
  "duration": 0,
  
  // SINGLE SOURCE OF TRUTH: All conversation data in one place
  "conversation": [
    {
      "speaker": "agent|client",
      "text": "What was said",
      "start_time": 12.34,
      "end_time": 15.67,
      "segment_id": "seg_001",
      "confidence": 0.95
    }
  ],
  
  // COMPUTED FIELDS (generated on-demand, not stored)
  // - agent_text: derived from conversation where speaker="agent"
  // - client_text: derived from conversation where speaker="client"  
  // - combined_text: derived by formatting conversation chronologically
  // - timestamped_dialogue: just rename conversation fields
  
  "script_text": "XML script content",
  "cqs": 0,
  "quality": 100,
  "emotions": {...},
  "adherence": {...},
  "agent_quality": {...},
  "summary": "AI-generated call summary",
  "tags": "complaint, refund, resolved",
  
  // METADATA
  "audio_metadata": {
    "agent_file_id": "GridFS_ID",
    "client_file_id": "GridFS_ID",
    "total_segments": 25,
    "total_speech_duration": 180.5
  }
}

BENEFITS:
1. ✅ 70% storage reduction (no text duplication)
2. ✅ Single source of truth for conversation
3. ✅ Consistent data (no sync issues)
4. ✅ Backward compatible with computed getters
5. ✅ Better performance (less I/O)
6. ✅ Easier maintenance
""" 


