# call_operations.py

from bson import ObjectId
import gridfs
from db_config import get_db
from datetime import datetime

class CallOperations:
    def __init__(self, db):
        self.db = db
        self.calls_collection = db.calls_collection
        self.fs = db.fs  # Use the GridFS instance from the MongoDB object

    def create_call(self, call_id, agent_id=None, customer_id=None, script_text=None):
        """Creates a new call document in the database."""
        try:
            # Check if a call with this ID already exists to prevent duplicates
            if self.get_call(call_id):
                print(f"[DB_OPS] Call with ID {call_id} already exists.")
                return True

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
                    "agent_words": [],
                    "client_words": []
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
            self.calls_collection.insert_one(call_doc)
            print(f"[DB_OPS] Created new call record for call_id: {call_id}")
            return True
        except Exception as e:
            # Handle potential race condition where another process creates the call
            if "duplicate key" in str(e).lower():
                print(f"[DB_OPS] Race condition avoided: Call with ID {call_id} already exists.")
                return True
            print(f"[DB_OPS] Error creating call {call_id}: {e}")
            return False

    def get_call(self, call_id):
        """Fetches a single call document by its ID."""
        try:
            return self.calls_collection.find_one({"call_id": call_id})
        except Exception as e:
            print(f"[DB_OPS] Error getting call {call_id}: {e}")
            return None

    def store_audio_and_update_call(self, call_id, client_audio, agent_audio, is_final):
        """Stores audio files by appending to existing GridFS files."""
        try:
            # Use the append-logic from db_config
            client_audio_id = self.db.store_or_append_audio(call_id, client_audio.read(), 'client')
            agent_audio_id = self.db.store_or_append_audio(call_id, agent_audio.read(), 'agent')
            
            # This function should only be responsible for storing audio.
            # The main app logic will handle updating other fields.
            return {"stored_audio": {"client_id": str(client_audio_id), "agent_id": str(agent_audio_id)}}
            
        except Exception as e:
            print(f"[DB_OPS] Error storing audio for call {call_id}: {e}")
            return {}
            
    def insert_partial_update(self, call_id, duration, cqs, adherence, emotions, transcription, quality):
        """Inserts a partial analysis update into the call document."""
        try:
            self.calls_collection.update_one(
                {"call_id": call_id},
                {
                    "$set": {
                        "cqs": cqs,
                        "adherence": adherence,
                        "emotions": emotions,
                        "transcription": transcription,
                        "quality": quality
                    },
                    "$inc": {"duration": duration}
                }
            )
        except Exception as e:
            print(f"[DB_OPS] Error inserting partial update for call {call_id}: {e}")

    def complete_call_update(self, call_id, agent_text, client_text, combined, cqs, overall_adherence, 
                             agent_quality, summary, emotions, duration, quality, tags, timestamped_dialogue,
                             agent_words, client_words):
        """Performs the final update to a call document upon completion."""
        try:
            # Create the final transcription object
            final_transcription = {
                "agent": agent_text,
                "client": client_text,
                "combined": combined,
                "timestamped_dialogue": timestamped_dialogue,
                "agent_words": agent_words,
                "client_words": client_words
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
                        "duration": duration,
                        "quality": quality,
                        "tags": tags,
                        "transcription": final_transcription,
                        "status": "completed"
                    }
                }
            )
        except Exception as e:
            print(f"[DB_OPS] Error completing call update for {call_id}: {e}")


# Singleton instance to be used by the app
def get_call_operations():
    db = get_db()
    return CallOperations(db)