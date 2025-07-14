import os
from pymongo import MongoClient
from gridfs import GridFS
from datetime import datetime
import json
from typing import Optional, Dict, Any, List
from bson import ObjectId

class MongoDB:
    """MongoDB connection and operations manager with GridFS support for audio storage"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.fs = None
        self.calls_collection = None
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
    
    def store_or_append_audio(self, call_id: str, audio_data: bytes, 
                             audio_type: str) -> str:
        """
        Store or append audio data for a call
        
        Args:
            call_id: Unique call identifier
            audio_data: Binary audio data to append
            audio_type: 'agent' or 'client'
        
        Returns:
            GridFS file ID as string
        """
        try:
            filename = f"{call_id}_{audio_type}.wav"
            
            # Check if audio file already exists
            existing_file = None
            try:
                existing_file = self.fs.find_one({"filename": filename})
            except:
                pass
            
            if existing_file:
                # Read existing audio data
                existing_data = existing_file.read()
                # Concatenate new audio data
                combined_audio = existing_data + audio_data
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
                    "timestamp": datetime.utcnow(),
                    "content_type": "audio/wav",
                    "size": len(combined_audio)
                }
            )
            
            print(f"[DB] Stored/updated audio: {filename} (ID: {file_id}, Size: {len(combined_audio)} bytes)")
            return str(file_id)
            
        except Exception as e:
            print(f"[DB ERROR] Failed to store/append audio: {e}")
            raise
    
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
                    print(f"[DB] Deleted audio file: {filename}")
            
            if deleted_count > 0:
                print(f"[DB] Deleted {deleted_count} audio files for call {call_id}")
            
        except Exception as e:
            print(f"[DB ERROR] Failed to delete audio files for call {call_id}: {e}")

# Global MongoDB instance
mongodb = MongoDB()

def get_db() -> MongoDB:
    """Get the global MongoDB instance"""
    return mongodb 


