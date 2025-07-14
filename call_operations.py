# call_operations.py

from pymongo.errors import DuplicateKeyError
from datetime import datetime
import gridfs
from bson.objectid import ObjectId
from db_config import get_db

class CallOperations:
    """Database operations for call management with segment-based storage"""
    
    def __init__(self, db):
        self.db = db
        self.calls_collection = db.db.calls  # Access the actual MongoDB database
        self.fs = gridfs.GridFS(db.db)       # Access the actual MongoDB database

    def create_call(self, call_id, agent_id=None, customer_id=None, script_text=None):
        """Create a new call record in the database with segment-based structure"""
        try:
            call_doc = {
                "call_id": call_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "date": datetime.utcnow(),
                "status": "initiated",
                "duration": 0,
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
                "tags": ""
            }
            
            result = self.calls_collection.insert_one(call_doc)
            print(f"[DB_OPS] Created call {call_id} with segment-based structure: {result.inserted_id}")
            return True
            
        except DuplicateKeyError:
            print(f"[DB_OPS] Call {call_id} already exists, skipping creation")
            return True
        except Exception as e:
            print(f"[DB_OPS] Error creating call {call_id}: {e}")
            return False

    def get_call(self, call_id):
        """Retrieve a call record by ID"""
        try:
            return self.calls_collection.find_one({"call_id": call_id})
        except Exception as e:
            print(f"[DB_OPS] Error retrieving call {call_id}: {e}")
            return None

    def store_audio_and_update_call(self, call_id, client_audio, agent_audio, is_final):
        """Store audio files in GridFS and update call with audio references"""
        try:
            stored_audio = {}
            
            if client_audio:
                client_audio.seek(0)
                client_file_id = self.fs.put(client_audio, filename=f"{call_id}_client_{'final' if is_final else 'chunk'}.wav")
                stored_audio['client_audio_id'] = str(client_file_id)
            
            if agent_audio:
                agent_audio.seek(0)
                agent_file_id = self.fs.put(agent_audio, filename=f"{call_id}_agent_{'final' if is_final else 'chunk'}.wav")
                stored_audio['agent_audio_id'] = str(agent_file_id)
            
            return {"stored_audio": stored_audio}
            
        except Exception as e:
            print(f"[DB_OPS] Error storing audio for call {call_id}: {e}")
            return {"stored_audio": {}}
            
    def insert_partial_update(self, call_id, duration, cqs, adherence, emotions, transcription, quality):
        """Insert partial update data during call processing with segment-based transcription"""
        try:
            self.calls_collection.update_one(
                {"call_id": call_id},
                {
                    "$set": {
                        "cqs": cqs,
                        "adherence": adherence,
                        "emotions": emotions,
                        "transcription": transcription,
                        "quality": quality,
                        "status": "in_progress"
                    },
                    "$inc": {"duration": duration}
                }
            )
        except Exception as e:
            print(f"[DB_OPS] Error inserting partial update for call {call_id}: {e}")

    def complete_call_update(self, call_id, agent_text, client_text, combined, cqs, overall_adherence, 
                             agent_quality, summary, emotions, duration, quality, tags, timestamped_dialogue,
                             agent_segments, client_segments):
        """Performs the final update to a call document upon completion with segment-based data"""
        try:
            # Create the final transcription object with segments
            final_transcription = {
                "agent": agent_text,
                "client": client_text,
                "combined": combined,
                "timestamped_dialogue": timestamped_dialogue,
                "agent_segments": agent_segments,
                "client_segments": client_segments
            }
            
            self.calls_collection.update_one(
                {"call_id": call_id},
                {
                    "$set": {
                        "cqs": cqs,
                        "adherence": overall_adherence,
                        "agent_quality": agent_quality,
                        "summary": summary,
                        "emotions": emotions,
                        "quality": quality,
                        "tags": tags,
                        "transcription": final_transcription,
                        "status": "completed"
                    },
                    "$inc": {"duration": duration}
                }
            )
            print(f"[DB_OPS] Completed final update for call {call_id} with segment-based data")
        except Exception as e:
            print(f"[DB_OPS] Error completing call {call_id}: {e}")


# Singleton instance to be used by the app
def get_call_operations():
    """Get a CallOperations instance with database connection"""
    db = get_db()
    return CallOperations(db)