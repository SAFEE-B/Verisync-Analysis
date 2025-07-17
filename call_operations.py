# call_operations.py

from pymongo.errors import DuplicateKeyError
from datetime import datetime, timezone
import gridfs
from bson.objectid import ObjectId
from db_config import (
    get_db, 
    build_conversation_from_segments, 
    get_transcription_legacy_format,
    compute_agent_text,
    compute_client_text
)
import threading

class CallOperations:
    """Database operations for call management with optimized schema"""
    
    def __init__(self, db):
        self.db = db
        self.calls_collection = db.calls_collection
        self.fs = db.fs
        self.lock = threading.Lock()

    def create_call(self, call_id, agent_id=None, customer_id=None, script_text=None):
        """Creates a new call document in the database using fully optimized schema"""
        try:
            # Check if call already exists
            if self.get_call(call_id):
                return True
            
            # FULLY OPTIMIZED SCHEMA - No redundant fields
            call_doc = {
                # Core identifiers
                "call_id": call_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "date": datetime.now(timezone.utc),
                "status": "initiated",
                "duration": 0,
                
                # SINGLE SOURCE OF TRUTH: All conversation data
                "conversation": [],
                
                # Script and analysis data
                "script_text": script_text,
                "cqs": 0,
                "quality": 100,
                # UPDATED: emotions as array of chunks + final emotion analysis
                "emotions": [],  # Array of chunk emotions
                "final_emotions": {},  # Final emotion analysis of complete transcript
                "adherence": {},
                "agent_quality": {},
                "summary": "",
                "tags": "",
                
                # OPTIMIZED: Chunk tracking for efficient operations
                "chunk_count": 0,
                "last_updated": datetime.now(timezone.utc),
                
                # OPTIMIZED: Minimal audio metadata (computed fields removed)
                "audio_metadata": {
                    "agent_file_id": None,
                    "client_file_id": None
                }
            }
            
            result = self.calls_collection.insert_one(call_doc)
            
            # Verify the insertion
            verification = self.calls_collection.find_one({"call_id": call_id})
            if not verification:
                return False
                
            return True
            
        except Exception as e:
            return False

    def get_call(self, call_id):
        """
        Retrieve call with backward compatibility for emotion structure
        Returns call with both new chunk-based emotions and computed legacy emotions field
        """
        try:
            call = self.calls_collection.find_one({"call_id": call_id})
            if not call:
                return None
            
            # Add computed fields for backward compatibility
            call = self._add_computed_fields(call)
            
            return call
            
        except Exception as e:
            return None
    
    def _add_computed_fields(self, call):
        """
        Add computed fields for backward compatibility with optimized schema
        """
        if not call:
            return call
        
        # Compute transcription fields from conversation array
        conversation = call.get('conversation', [])
        
        # Generate agent and client text
        agent_text = ' '.join([entry['text'] for entry in conversation if entry.get('speaker') == 'agent'])
        client_text = ' '.join([entry['text'] for entry in conversation if entry.get('speaker') == 'client'])
        
        # Generate combined transcription
        combined_text = '\n'.join([f"{entry['speaker'].title()}: {entry['text']}" for entry in conversation])
        
        # Generate timestamped dialogue
        timestamped_dialogue = [
            {
                "timestamp": entry.get('start_time', 0),
                "speaker": entry.get('speaker', '').title(),
                "text": entry.get('text', '')
            }
            for entry in conversation
        ]
        
        # Add computed transcription field for backward compatibility
        call['transcription'] = {
            'agent': agent_text,
            'client': client_text,
            'combined': combined_text,
            'timestamped_dialogue': timestamped_dialogue,
            'agent_segments': [entry for entry in conversation if entry.get('speaker') == 'agent'],
            'client_segments': [entry for entry in conversation if entry.get('speaker') == 'client']
        }
        
        # Add computed audio metadata
        audio_metadata = call.get('audio_metadata', {})
        if isinstance(audio_metadata, dict):
            # Compute total segments from conversation
            audio_metadata['total_segments'] = len(conversation)
            
            # Compute total speech duration
            total_duration = 0
            for entry in conversation:
                if 'start_time' in entry and 'end_time' in entry:
                    total_duration += entry['end_time'] - entry['start_time']
            audio_metadata['total_speech_duration'] = total_duration
            
            call['audio_metadata'] = audio_metadata
        
        # Handle emotion backward compatibility
        emotions_array = call.get('emotions', [])
        if isinstance(emotions_array, list) and emotions_array:
            # For backward compatibility, provide the latest chunk emotions as the main emotions field
            latest_chunk = emotions_array[-1] if emotions_array else {}
            call['emotions_legacy'] = latest_chunk.get('emotions', {})
            
            # Add emotion summary statistics
            call['emotion_analysis'] = {
                'total_chunks': len(emotions_array),
                'chunk_emotions': emotions_array,
                'final_emotions': call.get('final_emotions', {}),
                'latest_chunk_emotions': latest_chunk.get('emotions', {}),
                'average_cqs': sum(chunk.get('cqs', 0) for chunk in emotions_array) / len(emotions_array) if emotions_array else 0,
                'average_quality': sum(chunk.get('quality', 0) for chunk in emotions_array) / len(emotions_array) if emotions_array else 0
            }
        else:
            # Handle legacy format or empty emotions
            call['emotions_legacy'] = call.get('emotions', {}) if isinstance(call.get('emotions'), dict) else {}
            call['emotion_analysis'] = {
                'total_chunks': 0,
                'chunk_emotions': [],
                'final_emotions': call.get('final_emotions', {}),
                'latest_chunk_emotions': {},
                'average_cqs': 0,
                'average_quality': 0
            }
        
        return call

    def store_audio_and_update_call(self, call_id, client_audio, agent_audio, is_final):
        """Store audio files in GridFS and return file IDs - NOTE: Called within lock context"""
        try:
            print(f"[DB_DEBUG] Starting store_audio_and_update_call for call_id: {call_id}")
            # NOTE: No lock here as this is called from within store_audio_chunk_and_process which already has the lock
            
            # Store client audio
            client_file_id = None
            if client_audio:
                print(f"[DB_DEBUG] Storing client audio...")
                client_audio.seek(0)
                client_file_id = self.fs.put(
                    client_audio,
                    filename=f"{call_id}_client.wav",
                    metadata={"call_id": call_id, "type": "client", "is_final": is_final}
                )
                print(f"[DB_DEBUG] Client audio stored with ID: {client_file_id}")
            
            # Store agent audio
            agent_file_id = None
            if agent_audio:
                print(f"[DB_DEBUG] Storing agent audio...")
                agent_audio.seek(0)
                agent_file_id = self.fs.put(
                    agent_audio,
                    filename=f"{call_id}_agent.wav",
                    metadata={"call_id": call_id, "type": "agent", "is_final": is_final}
                )
                print(f"[DB_DEBUG] Agent audio stored with ID: {agent_file_id}")
            
            # Update call with audio file IDs
            update_doc = {"$set": {}}
            if client_file_id:
                update_doc["$set"]["audio_metadata.client_file_id"] = client_file_id
            if agent_file_id:
                update_doc["$set"]["audio_metadata.agent_file_id"] = agent_file_id
            
            print(f"[DB_DEBUG] Updating call document with audio metadata...")
            if update_doc["$set"]:
                self.calls_collection.update_one({"call_id": call_id}, update_doc)
                print(f"[DB_DEBUG] Call document updated successfully")
            
            result = {"stored_audio": {"client_id": str(client_file_id), "agent_id": str(agent_file_id)}}
            print(f"[DB_DEBUG] store_audio_and_update_call completed successfully")
            return result
                
        except Exception as e:
            print(f"[DB_DEBUG] ERROR in store_audio_and_update_call: {e}")
            import traceback
            traceback.print_exc()
            return {"stored_audio": {}}
            
    def insert_partial_update(self, call_id, duration, cqs, adherence, emotions, transcription, quality):
        """Insert partial update using optimized chunk-based emotion tracking with atomic operations"""
        try:

            
            # Build conversation array from transcription data
            conversation = []
            if isinstance(transcription, dict):
                agent_segments = transcription.get('agent_segments', [])
                client_segments = transcription.get('client_segments', [])
                conversation = build_conversation_from_segments(agent_segments, client_segments)
            
            # OPTIMIZED: Use atomic operations with efficient chunk numbering
            # This eliminates the need for separate queries and complex aggregation
            current_timestamp = datetime.now(timezone.utc)
            
            # OPTIMIZED: Get current chunk count and increment atomically
            # This is much more efficient than separate queries
            result = self.calls_collection.find_one_and_update(
                {"call_id": call_id},
                {
                    "$set": {
                        "cqs": cqs,
                        "adherence": adherence,
                        "quality": quality,
                        "status": "in_progress",
                        "last_updated": current_timestamp,
                        # OPTIMIZED: Store only conversation array - single source of truth
                        "conversation": conversation
                    },
                    "$inc": {
                        "duration": duration,
                        # OPTIMIZED: Track total chunks for easier querying
                        "chunk_count": 1
                    }
                },
                return_document=True  # Return updated document
            )
            
            if result:
                # Use the updated chunk_count for the chunk number
                chunk_number = result.get("chunk_count", 1)
                
                # Create chunk emotion entry with correct chunk number
                chunk_emotion_entry = {
                    "chunk_number": chunk_number,
                    "timestamp": current_timestamp,
                    "duration": duration,
                    "emotions": emotions,
                    "cqs": cqs,
                    "quality": quality,
                    "client_text_length": len(transcription.get('client', '')) if isinstance(transcription, dict) else 0
                }
                
                # OPTIMIZED: Single push operation with correct chunk number
                self.calls_collection.update_one(
                    {"call_id": call_id},
                    {"$push": {"emotions": chunk_emotion_entry}}
                )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    


    def complete_call_update(self, call_id, agent_text, client_text, combined, cqs, overall_adherence, 
                             agent_quality, summary, emotions, duration, quality, tags, timestamped_dialogue,
                             agent_segments, client_segments, final_emotions=None):
        """Complete call update using fully optimized schema with final emotion analysis"""
        try:
            # Build optimized conversation array
            conversation = build_conversation_from_segments(agent_segments, client_segments)
            
            # OPTIMIZED FINAL UPDATE - Only essential fields, no redundant storage
            update_doc = {
                "$set": {
                    "cqs": cqs,
                    "adherence": overall_adherence,
                    "agent_quality": agent_quality,
                    "summary": summary,
                    "quality": quality,
                    "tags": tags,
                    "status": "completed",
                    # OPTIMIZED: Single source of truth - no redundant storage
                    "conversation": conversation
                },
                "$inc": {"duration": duration}
            }
            
            # Add final emotions if provided
            if final_emotions:
                update_doc["$set"]["final_emotions"] = final_emotions
            
            result = self.calls_collection.update_one({"call_id": call_id}, update_doc)
            
        except Exception as e:
            import traceback
            traceback.print_exc()

    # =============================================================================
    # OPTIMIZED SCHEMA HELPERS
    # =============================================================================
    
    def get_conversation(self, call_id):
        """Get conversation array for a call"""
        try:
            call_doc = self.calls_collection.find_one({"call_id": call_id}, {"conversation": 1})
            return call_doc.get("conversation", []) if call_doc else []
        except Exception as e:
            return []
    
    def get_agent_text(self, call_id):
        """Get agent text computed from conversation"""
        conversation = self.get_conversation(call_id)
        return compute_agent_text(conversation)
    
    def get_client_text(self, call_id):
        """Get client text computed from conversation"""
        conversation = self.get_conversation(call_id)
        return compute_client_text(conversation)
    
    def store_audio_chunk_and_process(self, call_id, client_audio, agent_audio):
        """Store audio chunk and process segments using database-backed approach"""
        try:
            print(f"[DB_DEBUG] Starting store_audio_chunk_and_process for call_id: {call_id}")
            print(f"[DB_DEBUG] - Client audio provided: {client_audio is not None}")
            print(f"[DB_DEBUG] - Agent audio provided: {agent_audio is not None}")
            
            with self.lock:
                print(f"[DB_DEBUG] Acquired lock, storing audio chunks")
                # Store audio chunks in GridFS
                stored_audio = self.store_audio_and_update_call(call_id, client_audio, agent_audio, False)
                print(f"[DB_DEBUG] Audio stored: {stored_audio}")
                
                print(f"[DB_DEBUG] Getting conversation for call_id: {call_id}")
                # Get current segments from conversation
                conversation = self.get_conversation(call_id)
                agent_segments = [entry for entry in conversation if entry.get('speaker') == 'agent']
                client_segments = [entry for entry in conversation if entry.get('speaker') == 'client']
                
                print(f"[DB_DEBUG] Retrieved {len(agent_segments)} agent segments and {len(client_segments)} client segments")
                
                # Calculate overlap times (use last segment end time if available)
                agent_overlap_start = 0.0
                client_overlap_start = 0.0
                
                if agent_segments:
                    agent_overlap_start = agent_segments[-1].get('end_time', 0.0)
                if client_segments:
                    client_overlap_start = client_segments[-1].get('end_time', 0.0)
                
                print(f"[DB_DEBUG] Overlap times - Agent: {agent_overlap_start}, Client: {client_overlap_start}")
                
                # Prepare audio for transcription (read the audio data)
                agent_audio_for_transcription = None
                client_audio_for_transcription = None
                
                if agent_audio:
                    print(f"[DB_DEBUG] Reading agent audio data")
                    agent_audio.seek(0)
                    agent_audio_for_transcription = agent_audio.read()
                    agent_audio.seek(0)
                    print(f"[DB_DEBUG] Agent audio size: {len(agent_audio_for_transcription) if agent_audio_for_transcription else 0} bytes")
                
                if client_audio:
                    print(f"[DB_DEBUG] Reading client audio data")
                    client_audio.seek(0)
                    client_audio_for_transcription = client_audio.read()
                    client_audio.seek(0)
                    print(f"[DB_DEBUG] Client audio size: {len(client_audio_for_transcription) if client_audio_for_transcription else 0} bytes")
                
                result = {
                    "stored_audio": stored_audio.get("stored_audio", {}),
                    "agent_segments": agent_segments,
                    "client_segments": client_segments,
                    "agent_overlap_start": agent_overlap_start,
                    "client_overlap_start": client_overlap_start,
                    "agent_audio_for_transcription": agent_audio_for_transcription,
                    "client_audio_for_transcription": client_audio_for_transcription
                }
                
                print(f"[DB_DEBUG] store_audio_chunk_and_process completed successfully")
                return result
                
        except Exception as e:
            print(f"[DB_DEBUG] ERROR in store_audio_chunk_and_process: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stored_audio": {},
                "agent_segments": [],
                "client_segments": [],
                "agent_overlap_start": 0.0,
                "client_overlap_start": 0.0,
                "agent_audio_for_transcription": None,
                "client_audio_for_transcription": None
            }

    def update_call_segments(self, call_id, agent_segments=None, client_segments=None):
        """Update call segments in the conversation array"""
        try:
            if agent_segments or client_segments:
                # Build new conversation from segments
                all_segments = []
                if agent_segments:
                    for segment in agent_segments:
                        # Normalize segment format
                        normalized_segment = {
                            'speaker': 'agent',
                            'text': segment.get('text', ''),
                            'start_time': segment.get('start', segment.get('start_time', 0)),
                            'end_time': segment.get('end', segment.get('end_time', 0)),
                            'confidence': segment.get('confidence', 1.0)
                        }
                        all_segments.append(normalized_segment)
                if client_segments:
                    for segment in client_segments:
                        # Normalize segment format
                        normalized_segment = {
                            'speaker': 'client',
                            'text': segment.get('text', ''),
                            'start_time': segment.get('start', segment.get('start_time', 0)),
                            'end_time': segment.get('end', segment.get('end_time', 0)),
                            'confidence': segment.get('confidence', 1.0)
                        }
                        all_segments.append(normalized_segment)
                
                # Sort by start time
                all_segments.sort(key=lambda x: x.get('start_time', 0))
                
                # Update conversation
                self.calls_collection.update_one(
                    {"call_id": call_id},
                    {"$set": {"conversation": all_segments}}
                )
                
        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_full_call_audio_and_segments(self, call_id):
        """Get complete audio and segments for a call"""
        try:
            call = self.get_call(call_id)
            if not call:
                return {}
            
            # Get audio file IDs
            audio_metadata = call.get('audio_metadata', {})
            client_file_id = audio_metadata.get('client_file_id')
            agent_file_id = audio_metadata.get('agent_file_id')
            
            # Retrieve audio data
            client_audio = None
            agent_audio = None
            
            if client_file_id:
                client_file = self.fs.get(ObjectId(client_file_id))
                client_audio = client_file.read()
            
            if agent_file_id:
                agent_file = self.fs.get(ObjectId(agent_file_id))
                agent_audio = agent_file.read()
            
            # Get segments from conversation
            conversation = call.get('conversation', [])
            agent_segments = [entry for entry in conversation if entry.get('speaker') == 'agent']
            client_segments = [entry for entry in conversation if entry.get('speaker') == 'client']
            
            return {
                "client_audio": client_audio,
                "agent_audio": agent_audio,
                "agent_segments": agent_segments,
                "client_segments": client_segments
            }
            
        except Exception as e:
            return {}

    def cleanup_call_resources(self, call_id):
        """Clean up GridFS files for a completed call"""
        try:
            call = self.get_call(call_id)
            if not call:
                return
            
            audio_metadata = call.get('audio_metadata', {})
            
            # Delete GridFS files
            if audio_metadata.get('client_file_id'):
                self.fs.delete(ObjectId(audio_metadata['client_file_id']))
            if audio_metadata.get('agent_file_id'):
                self.fs.delete(ObjectId(audio_metadata['agent_file_id']))
            
            # Clear audio metadata
            self.calls_collection.update_one(
                {"call_id": call_id},
                {"$set": {"audio_metadata": {}}}
            )
            
        except Exception as e:
            pass


# Singleton instance to be used by the app
def get_call_operations():
    """Get CallOperations instance with database connection"""
    db = get_db()
    return CallOperations(db)