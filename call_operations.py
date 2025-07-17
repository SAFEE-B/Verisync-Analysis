# call_operations.py

from pymongo.errors import DuplicateKeyError
from datetime import datetime
import gridfs
from bson.objectid import ObjectId
from db_config import (
    get_db, 
    build_conversation_from_segments, 
    get_transcription_legacy_format,
    compute_agent_text,
    compute_client_text
)

class CallOperations:
    """Database operations for call management with optimized schema"""
    
    def __init__(self, db):
        self.db = db
        self.calls_collection = db.db.calls  # Access the actual MongoDB database
        self.fs = gridfs.GridFS(db.db)       # Access the actual MongoDB database

    def create_call(self, call_id, agent_id=None, customer_id=None, script_text=None):
        """Create a new call record with optimized schema"""
        try:
            print(f"[DB_OPS] 📝 Creating call document for call_id: {call_id}")
            print(f"[DB_OPS] ├── Agent ID: {agent_id}")
            print(f"[DB_OPS] ├── Customer ID: {customer_id}")
            print(f"[DB_OPS] ├── Script text length: {len(script_text) if script_text else 0} characters")
            
            call_doc = {
                "call_id": call_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "date": datetime.utcnow(),
                "status": "initiated",
                "duration": 0,
                
                # OPTIMIZED: Single source of truth for conversation
                "conversation": [],
                
                # LEGACY SUPPORT: Keep old structure for backward compatibility
                "transcription": {
                    "agent": "",
                    "client": "",
                    "combined": "",
                    "timestamped_dialogue": [],
                    "agent_segments": [],
                    "client_segments": []
                },
                
                "script_text": script_text,
                "cqs": 0,
                "quality": 100,
                "emotions": {},
                "adherence": {},
                "agent_quality": {},
                "summary": "",
                "tags": "",
                
                # OPTIMIZED: Audio metadata
                "audio_metadata": {
                    "agent_file_id": None,
                    "client_file_id": None,
                    "total_segments": 0,
                    "total_speech_duration": 0
                }
            }
            
            print(f"[DB_OPS] 💾 Inserting call document into database...")
            result = self.calls_collection.insert_one(call_doc)
            print(f"[DB_OPS] ✅ Successfully created call {call_id} with MongoDB ObjectId: {result.inserted_id}")
            print(f"[DB_OPS] 📊 Document size: {len(str(call_doc))} bytes")
            
            # Verify the insertion
            verification = self.calls_collection.find_one({"call_id": call_id})
            if verification:
                print(f"[DB_OPS] ✅ Verification: Call {call_id} found in database with status: {verification['status']}")
            else:
                print(f"[DB_OPS] ❌ Verification failed: Call {call_id} not found in database")
                
            return True
            
        except DuplicateKeyError:
            print(f"[DB_OPS] ⚠️  Call {call_id} already exists, skipping creation")
            return True
        except Exception as e:
            print(f"[DB_OPS] ❌ Error creating call {call_id}: {e}")
            return False

    def get_call(self, call_id):
        """Retrieve a call record by ID with dynamic transcription generation"""
        try:
            call_doc = self.calls_collection.find_one({"call_id": call_id})
            
            if not call_doc:
                return None
                
            # If using optimized schema, generate legacy transcription on-demand
            if "conversation" in call_doc and call_doc["conversation"]:
                legacy_transcription = get_transcription_legacy_format(call_doc["conversation"])
                call_doc["transcription"] = legacy_transcription
                
            return call_doc
            
        except Exception as e:
            print(f"[DB_OPS] Error retrieving call {call_id}: {e}")
            return None

    def store_audio_and_update_call(self, call_id, client_audio, agent_audio, is_final):
        """Store audio files in GridFS and update call with audio references"""
        try:
            print(f"[DB_OPS] 🎵 Starting audio storage for call_id: {call_id}")
            print(f"[DB_OPS] ├── Is final: {is_final}")
            
            stored_audio = {}
            
            if client_audio:
                client_audio.seek(0)
                client_audio_size = len(client_audio.read())
                client_audio.seek(0)
                
                filename = f"{call_id}_client_{'final' if is_final else 'chunk'}.wav"
                print(f"[DB_OPS] ├── Storing client audio: {filename} ({client_audio_size} bytes)")
                
                client_file_id = self.fs.put(client_audio, filename=filename)
                stored_audio['client_audio_id'] = str(client_file_id)
                print(f"[DB_OPS] ├── Client audio stored with GridFS ID: {client_file_id}")
                
                # Update audio metadata
                result = self.calls_collection.update_one(
                    {"call_id": call_id},
                    {"$set": {"audio_metadata.client_file_id": str(client_file_id)}}
                )
                print(f"[DB_OPS] ├── Client audio metadata updated: {result.modified_count} documents")
            
            if agent_audio:
                agent_audio.seek(0)
                agent_audio_size = len(agent_audio.read())
                agent_audio.seek(0)
                
                filename = f"{call_id}_agent_{'final' if is_final else 'chunk'}.wav"
                print(f"[DB_OPS] ├── Storing agent audio: {filename} ({agent_audio_size} bytes)")
                
                agent_file_id = self.fs.put(agent_audio, filename=filename)
                stored_audio['agent_audio_id'] = str(agent_file_id)
                print(f"[DB_OPS] ├── Agent audio stored with GridFS ID: {agent_file_id}")
                
                # Update audio metadata
                result = self.calls_collection.update_one(
                    {"call_id": call_id},
                    {"$set": {"audio_metadata.agent_file_id": str(agent_file_id)}}
                )
                print(f"[DB_OPS] ├── Agent audio metadata updated: {result.modified_count} documents")
            
            print(f"[DB_OPS] ✅ Audio storage completed for call {call_id}")
            print(f"[DB_OPS] ├── Stored files: {list(stored_audio.keys())}")
            
            return {"stored_audio": stored_audio}
            
        except Exception as e:
            print(f"[DB_OPS] ❌ Error storing audio for call {call_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"stored_audio": {}}
            
    def insert_partial_update(self, call_id, duration, cqs, adherence, emotions, transcription, quality):
        """Insert partial update using optimized schema"""
        try:
            print(f"[DB_OPS] 🔄 Starting partial update for call_id: {call_id}")
            
            # Build conversation array from transcription data
            conversation = []
            if isinstance(transcription, dict):
                agent_segments = transcription.get('agent_segments', [])
                client_segments = transcription.get('client_segments', [])
                conversation = build_conversation_from_segments(agent_segments, client_segments)
                
                print(f"[DB_OPS] ├── Agent segments: {len(agent_segments)}")
                print(f"[DB_OPS] ├── Client segments: {len(client_segments)}")
                print(f"[DB_OPS] ├── Built conversation entries: {len(conversation)}")
            
            # Calculate audio metadata
            total_segments = len(conversation)
            total_speech_duration = sum(
                segment.get('end_time', 0) - segment.get('start_time', 0) 
                for segment in conversation
            )
            
            print(f"[DB_OPS] ├── Duration increment: {duration}s")
            print(f"[DB_OPS] ├── CQS: {cqs}")
            print(f"[DB_OPS] ├── Quality: {quality}")
            print(f"[DB_OPS] ├── Adherence overall: {adherence.get('overall', 0) if isinstance(adherence, dict) else 0}")
            print(f"[DB_OPS] ├── Emotions: {len(emotions)} emotion types")
            print(f"[DB_OPS] ├── Total speech duration: {total_speech_duration:.2f}s")
            
            update_doc = {
                "$set": {
                    "cqs": cqs,
                    "adherence": adherence,
                    "emotions": emotions,
                    "quality": quality,
                    "status": "in_progress",
                    # OPTIMIZED: Store only conversation array - no redundant storage
                    "conversation": conversation,
                    "audio_metadata.total_segments": total_segments,
                    "audio_metadata.total_speech_duration": total_speech_duration
                },
                "$inc": {"duration": duration},
                "$unset": {
                    # Remove legacy transcription fields to save storage
                    "transcription.agent": "",
                    "transcription.client": "",
                    "transcription.combined": "",
                    "transcription.timestamped_dialogue": "",
                    "transcription.agent_segments": "",
                    "transcription.client_segments": ""
                }
            }
            
            print(f"[DB_OPS] 💾 Executing partial update to database...")
            result = self.calls_collection.update_one({"call_id": call_id}, update_doc)
            
            if result.matched_count > 0:
                print(f"[DB_OPS] ✅ Partial update successful for call {call_id}")
                print(f"[DB_OPS] ├── Documents matched: {result.matched_count}")
                print(f"[DB_OPS] ├── Documents modified: {result.modified_count}")
                print(f"[DB_OPS] ├── Total segments stored: {total_segments}")
                print(f"[DB_OPS] ├── Status updated to: in_progress")
                
                # Verify the update
                verification = self.calls_collection.find_one({"call_id": call_id})
                if verification:
                    print(f"[DB_OPS] ✅ Verification: Call {call_id} status is now: {verification['status']}")
                    print(f"[DB_OPS] ├── Current duration: {verification.get('duration', 0)}s")
                    print(f"[DB_OPS] ├── Current CQS: {verification.get('cqs', 0)}")
                    print(f"[DB_OPS] ├── Conversation entries: {len(verification.get('conversation', []))}")
            else:
                print(f"[DB_OPS] ❌ Partial update failed: No documents matched for call {call_id}")
            
        except Exception as e:
            print(f"[DB_OPS] ❌ Error inserting partial update for call {call_id}: {e}")
            import traceback
            traceback.print_exc()

    def complete_call_update(self, call_id, agent_text, client_text, combined, cqs, overall_adherence, 
                             agent_quality, summary, emotions, duration, quality, tags,
                             agent_segments, client_segments):
        """Complete call update using optimized schema (timestamped_dialogue argument removed)"""
        try:
            print(f"[DB_OPS] 🏁 Starting final update for call_id: {call_id}")
            
            # Build optimized conversation array
            conversation = build_conversation_from_segments(agent_segments, client_segments)
            
            # Calculate final audio metadata
            total_segments = len(conversation)
            total_speech_duration = sum(
                segment.get('end_time', 0) - segment.get('start_time', 0) 
                for segment in conversation
            )
            
            print(f"[DB_OPS] ├── Agent text length: {len(agent_text)} characters")
            print(f"[DB_OPS] ├── Client text length: {len(client_text)} characters")
            print(f"[DB_OPS] ├── Combined text length: {len(combined)} characters")
            print(f"[DB_OPS] ├── Agent segments: {len(agent_segments)}")
            print(f"[DB_OPS] ├── Client segments: {len(client_segments)}")
            print(f"[DB_OPS] ├── Conversation entries: {len(conversation)}")
            print(f"[DB_OPS] ├── Total speech duration: {total_speech_duration:.2f}s")
            print(f"[DB_OPS] ├── Final CQS: {cqs}")
            print(f"[DB_OPS] ├── Final Quality: {quality}")
            print(f"[DB_OPS] ├── Final Adherence: {overall_adherence.get('overall', 0) if isinstance(overall_adherence, dict) else 0}")
            print(f"[DB_OPS] ├── Summary length: {len(summary)} characters")
            print(f"[DB_OPS] ├── Tags: {tags}")
            print(f"[DB_OPS] ├── Duration increment: {duration}s")
            print(f"[DB_OPS] ├── Agent quality questions: {len(agent_quality) if isinstance(agent_quality, dict) else 0}")
            print(f"[DB_OPS] ├── Emotion types: {len(emotions) if isinstance(emotions, dict) else 0}")
            # Removed timestamped_dialogue print
            
            # Store only optimized conversation array - legacy fields computed on-demand
            update_doc = {
                "$set": {
                    "cqs": cqs,
                    "adherence": overall_adherence,
                    "agent_quality": agent_quality,
                    "summary": summary,
                    "emotions": emotions,
                    "quality": quality,
                    "tags": tags,
                    "status": "completed",
                    # OPTIMIZED: Single source of truth - no redundant storage
                    "conversation": conversation,
                    "audio_metadata.total_segments": total_segments,
                    "audio_metadata.total_speech_duration": total_speech_duration
                },
                "$inc": {"duration": duration},
                "$unset": {
                    # Remove legacy transcription fields to save storage
                    "transcription.agent": "",
                    "transcription.client": "",
                    "transcription.combined": "",
                    "transcription.timestamped_dialogue": "",
                    "transcription.agent_segments": "",
                    "transcription.client_segments": ""
                }
            }
            
            print(f"[DB_OPS] 💾 Executing final update to database...")
            result = self.calls_collection.update_one({"call_id": call_id}, update_doc)
            
            if result.matched_count > 0:
                print(f"[DB_OPS] ✅ Final update successful for call {call_id}")
                print(f"[DB_OPS] ├── Documents matched: {result.matched_count}")
                print(f"[DB_OPS] ├── Documents modified: {result.modified_count}")
                print(f"[DB_OPS] ├── Status updated to: completed")
                
                # Verify the final update
                verification = self.calls_collection.find_one({"call_id": call_id})
                if verification:
                    print(f"[DB_OPS] ✅ Final verification: Call {call_id} completed successfully")
                    print(f"[DB_OPS] ├── Final status: {verification['status']}")
                    print(f"[DB_OPS] ├── Final duration: {verification.get('duration', 0)}s")
                    print(f"[DB_OPS] ├── Final CQS: {verification.get('cqs', 0)}")
                    print(f"[DB_OPS] ├── Final quality: {verification.get('quality', 0)}")
                    print(f"[DB_OPS] ├── Final conversation entries: {len(verification.get('conversation', []))}")
                    print(f"[DB_OPS] ├── Final summary present: {bool(verification.get('summary', ''))}")
                    print(f"[DB_OPS] ├── Final tags present: {bool(verification.get('tags', ''))}")
                    print(f"[DB_OPS] ├── Audio metadata segments: {verification.get('audio_metadata', {}).get('total_segments', 0)}")
                    print(f"[DB_OPS] ├── Audio metadata duration: {verification.get('audio_metadata', {}).get('total_speech_duration', 0):.2f}s")
                else:
                    print(f"[DB_OPS] ❌ Final verification failed: Call {call_id} not found in database")
            else:
                print(f"[DB_OPS] ❌ Final update failed: No documents matched for call {call_id}")
            
        except Exception as e:
            print(f"[DB_OPS] ❌ Error completing call {call_id}: {e}")
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
            print(f"[DB_OPS] Error getting conversation for call {call_id}: {e}")
            return []
    
    def get_agent_text(self, call_id):
        """Get agent text computed from conversation"""
        conversation = self.get_conversation(call_id)
        return compute_agent_text(conversation)
    
    def get_client_text(self, call_id):
        """Get client text computed from conversation"""
        conversation = self.get_conversation(call_id)
        return compute_client_text(conversation)
    
    def get_audio_metadata(self, call_id):
        """Get audio metadata for a call"""
        try:
            call_doc = self.calls_collection.find_one({"call_id": call_id}, {"audio_metadata": 1})
            return call_doc.get("audio_metadata", {}) if call_doc else {}
        except Exception as e:
            print(f"[DB_OPS] Error getting audio metadata for call {call_id}: {e}")
            return {}


# Singleton instance to be used by the app
def get_call_operations():
    """Get a CallOperations instance with database connection"""
    db = get_db()
    return CallOperations(db)