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
from bson.objectid import ObjectId
import io
import struct
import os
import tempfile

class CallOperations:
    """Database operations for call management with optimized schema"""
    
    def __init__(self, db):
        self.db = db
        self.calls_collection = db.calls_collection
        self.fs = db.fs
        self.lock = threading.Lock()
        self.audio_cache_dir = os.path.abspath('./audio_cache')
        os.makedirs(self.audio_cache_dir, exist_ok=True)

    # =============================================================================
    # AUDIO PROCESSING HELPERS
    # =============================================================================
    
    def _seconds_to_byte_offset(self, seconds, sample_rate=24000, channels=1, sample_width=2):
        """Convert seconds to byte offset for audio data"""
        try:
            return int(seconds * sample_rate * channels * sample_width)
        except:
            return 0
    
    def _extract_pcm_from_wav(self, wav_data):
        """
        Extract raw PCM data from WAV file, removing headers
        Returns: (pcm_data, sample_rate, channels, sample_width)
        """
        try:
            if not wav_data:
                return None, None, None, None
            
            import wave
            import tempfile
            import os
            
            # Write WAV data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_filename = temp_file.name
            
            # Read WAV file and extract PCM data
            with wave.open(temp_filename, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                pcm_data = wav_file.readframes(wav_file.getnframes())
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            return pcm_data, sample_rate, channels, sample_width
            
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error extracting PCM from WAV: {e}")
            # Fallback: assume it's already PCM data
            return wav_data, 24000, 1, 2
    
    def _combine_pcm_data(self, pcm_data_list, sample_rate=24000, channels=1, sample_width=2):
        """
        Combine multiple PCM data chunks ensuring consistent format
        """
        try:
            if not pcm_data_list:
                return None
            
            # Filter out None values
            valid_pcm_data = [pcm for pcm in pcm_data_list if pcm is not None]
            if not valid_pcm_data:
                return None
            
            # Simply concatenate PCM data (all should have same format)
            combined_pcm = b''.join(valid_pcm_data)
            
            return combined_pcm
            
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error combining PCM data: {e}")
            return None
    
    def _create_wav_file(self, pcm_data, sample_rate=24000, channels=1, sample_width=2):
        """
        Convert raw PCM data to proper WAV format with headers
        This ensures the audio can be processed by APIs like Groq
        """
        try:
            if not pcm_data:
                return None
            
            import wave
            import tempfile
            import os
            
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Write WAV file with proper headers
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            # Read the WAV file back as bytes
            with open(temp_filename, 'rb') as wav_file:
                wav_data = wav_file.read()
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            return wav_data
            
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error creating WAV file: {e}")
            return pcm_data  # Fallback to original data
    
    def _pcm_data_to_wav_bytesio(self, pcm_data, sample_rate=24000, channels=1, sample_width=2):
        """
        Convert raw PCM data to WAV format and return as BytesIO object
        """
        try:
            if not pcm_data:
                return None
            
            wav_data = self._create_wav_file(pcm_data, sample_rate, channels, sample_width)
            if wav_data:
                return io.BytesIO(wav_data)
            return None
            
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error converting PCM to WAV BytesIO: {e}")
            return None
    
    def _get_previous_audio_from_local_cache(self, call_id, speaker_type):
        """Get previous audio data from local cache for a specific speaker, extracting PCM data"""
        try:
            pcm_data = self._read_audio_cache(call_id, speaker_type)
            if not pcm_data:
                return None
            # Assume default format for local cache
            return pcm_data
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error getting previous audio from local cache: {e}")
            return None
    
    def _get_previous_audio_from_gridfs(self, call_id, speaker_type):
        """Get previous audio data from GridFS for a specific speaker, extracting PCM data"""
        try:
            call = self.calls_collection.find_one({"call_id": call_id})
            if not call:
                return None
            audio_metadata = call.get('audio_metadata', {})
            file_id = None
            if speaker_type == 'agent':
                file_id = audio_metadata.get('agent_file_id')
            elif speaker_type == 'client':
                file_id = audio_metadata.get('client_file_id')
            if file_id:
                audio_file = self.fs.get(ObjectId(file_id))
                wav_data = audio_file.read()
                pcm_data, sample_rate, channels, sample_width = self._extract_pcm_from_wav(wav_data)
                print(f"[AUDIO_DEBUG] Retrieved {speaker_type} audio from GridFS: WAV {len(wav_data)} bytes -> PCM {len(pcm_data) if pcm_data else 0} bytes ({sample_rate}Hz, {channels}ch, {sample_width}bytes)")
                return pcm_data
            return None
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error getting previous audio from GridFS: {e}")
            return None
    
    def _combine_and_slice_audio(self, previous_audio_bytes, new_audio_bytes, overlap_start_seconds):
        """
        Combine previous audio with new audio and return only the portion from overlap_start onwards
        """
        try:
            if not previous_audio_bytes and not new_audio_bytes:
                return None
            
            # If no previous audio, return new audio from overlap point
            if not previous_audio_bytes:
                if overlap_start_seconds <= 0:
                    return new_audio_bytes
                # Slice new audio from overlap point
                byte_offset = self._seconds_to_byte_offset(overlap_start_seconds)
                return new_audio_bytes[byte_offset:] if byte_offset < len(new_audio_bytes) else new_audio_bytes
            
            # If no new audio, return previous audio from overlap point
            if not new_audio_bytes:
                if overlap_start_seconds <= 0:
                    return previous_audio_bytes
                byte_offset = self._seconds_to_byte_offset(overlap_start_seconds)
                return previous_audio_bytes[byte_offset:] if byte_offset < len(previous_audio_bytes) else previous_audio_bytes
            
            # Combine both audio streams
            combined_audio = previous_audio_bytes + new_audio_bytes
            
            # Calculate byte offset for overlap start
            byte_offset = self._seconds_to_byte_offset(overlap_start_seconds)
            
            # Return audio from overlap point to end
            if byte_offset >= len(combined_audio):
                # If overlap point is beyond combined audio, return the new chunk
                return new_audio_bytes
            
            return combined_audio[byte_offset:]
            
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error combining and slicing audio: {e}")
            # Fallback to new audio if there's an error
            return new_audio_bytes if new_audio_bytes else previous_audio_bytes
    
    def _slice_audio_from_time(self, audio_bytes, start_time_seconds):
        """
        Slice audio data starting from a specific time point
        """
        try:
            if not audio_bytes:
                return None
            
            if start_time_seconds <= 0:
                return audio_bytes
            
            # Calculate byte offset for start time
            byte_offset = self._seconds_to_byte_offset(start_time_seconds)
            
            # Return audio from start time to end
            if byte_offset >= len(audio_bytes):
                # If start time is beyond audio length, return empty or minimal audio
                return None
            
            return audio_bytes[byte_offset:]
            
        except Exception as e:
            print(f"[AUDIO_DEBUG] Error slicing audio from time: {e}")
            return audio_bytes

    def create_call(self, call_id, agent_id=None, sip_id=None, script_text=None):
        """Creates a new call document in the database using fully optimized schema"""
        try:
            # Check if call already exists
            if self.get_call(call_id):
                return True
            
            # FULLY OPTIMIZED SCHEMA - No redundant fields
            call_doc = {
                # Core identifiers
                "call_id": call_id,
                "agent_id": ObjectId(agent_id),
                "sip_id": sip_id,
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
            print("Ran till the end of create call")
            # Verify the insertion
            verification = self.calls_collection.find_one({"call_id": call_id})
            if not verification:
                return False
                
            return True
            
        except Exception as e:
            print("Error in call creation.Error is: ", e)
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
        """Store audio files in local cache for intermediate updates, and in GridFS only at final update."""
        try:
            print(f"[DB_DEBUG] Starting store_audio_and_update_call for call_id: {call_id}")
            call_doc = self.calls_collection.find_one({"call_id": call_id})
            audio_metadata = call_doc.get('audio_metadata', {}) if call_doc else {}
            
            # Store client audio
            client_file_id = None
            if client_audio:
                print(f"[DB_DEBUG] Processing client audio...")
                client_audio.seek(0)
                new_client_audio_bytes = client_audio.read()
                new_client_pcm, sample_rate, channels, sample_width = self._extract_pcm_from_wav(new_client_audio_bytes)
                print(f"[DB_DEBUG] New client audio - PCM: {len(new_client_pcm) if new_client_pcm else 0} bytes, SR: {sample_rate}, CH: {channels}, SW: {sample_width}")
                if not is_final:
                    self._append_to_audio_cache(call_id, 'client', new_client_pcm)
                    combined_client_pcm = self._read_audio_cache(call_id, 'client')
                    print(f"[DB_DEBUG] Local cache client PCM: {len(combined_client_pcm)} bytes")
                else:
                    # On final update, combine all and store in GridFS
                    self._append_to_audio_cache(call_id, 'client', new_client_pcm)
                    combined_client_pcm = self._read_audio_cache(call_id, 'client')
                    combined_client_wav = self._create_wav_file(combined_client_pcm, sample_rate, channels, sample_width)
                    client_file_id = self.fs.put(
                        io.BytesIO(combined_client_wav),
                        filename=f"{call_id}_client_combined.wav",
                        metadata={
                            "call_id": call_id, 
                            "type": "client", 
                            "is_final": True,
                            "combined": True,
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "sample_width": sample_width,
                            "pcm_size": len(combined_client_pcm),
                            "wav_size": len(combined_client_wav)
                        }
                    )
                    print(f"[DB_DEBUG] Final client WAV stored with ID: {client_file_id} (PCM: {len(combined_client_pcm)}, WAV: {len(combined_client_wav)} bytes)")
                    self._delete_audio_cache(call_id)
            
            # Store agent audio
            agent_file_id = None
            if agent_audio:
                print(f"[DB_DEBUG] Processing agent audio...")
                agent_audio.seek(0)
                new_agent_audio_bytes = agent_audio.read()
                new_agent_pcm, sample_rate, channels, sample_width = self._extract_pcm_from_wav(new_agent_audio_bytes)
                print(f"[DB_DEBUG] New agent audio - PCM: {len(new_agent_pcm) if new_agent_pcm else 0} bytes, SR: {sample_rate}, CH: {channels}, SW: {sample_width}")
                if not is_final:
                    self._append_to_audio_cache(call_id, 'agent', new_agent_pcm)
                    combined_agent_pcm = self._read_audio_cache(call_id, 'agent')
                    print(f"[DB_DEBUG] Local cache agent PCM: {len(combined_agent_pcm)} bytes")
                else:
                    self._append_to_audio_cache(call_id, 'agent', new_agent_pcm)
                    combined_agent_pcm = self._read_audio_cache(call_id, 'agent')
                    combined_agent_wav = self._create_wav_file(combined_agent_pcm, sample_rate, channels, sample_width)
                    agent_file_id = self.fs.put(
                        io.BytesIO(combined_agent_wav),
                        filename=f"{call_id}_agent_combined.wav",
                        metadata={
                            "call_id": call_id, 
                            "type": "agent", 
                            "is_final": True,
                            "combined": True,
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "sample_width": sample_width,
                            "pcm_size": len(combined_agent_pcm),
                            "wav_size": len(combined_agent_wav)
                        }
                    )
                    print(f"[DB_DEBUG] Final agent WAV stored with ID: {agent_file_id} (PCM: {len(combined_agent_pcm)}, WAV: {len(combined_agent_wav)} bytes)")
                    self._delete_audio_cache(call_id)
            
            # Update call with audio file IDs only at final update
            update_doc = {"$set": {}}
            if is_final:
                if client_file_id:
                    update_doc["$set"]["audio_metadata.client_file_id"] = client_file_id
                if agent_file_id:
                    update_doc["$set"]["audio_metadata.agent_file_id"] = agent_file_id
                print(f"[DB_DEBUG] Updating call document with audio metadata...")
                if update_doc["$set"]:
                    self.calls_collection.update_one({"call_id": call_id}, update_doc)
                    print(f"[DB_DEBUG] Call document updated successfully")
                    self._delete_audio_cache(call_id)
            
            result = {"stored_audio": {"client_id": str(client_file_id) if client_file_id else None, "agent_id": str(agent_file_id) if agent_file_id else None}}
            print(f"[DB_DEBUG] store_audio_and_update_call completed successfully")
            return result
        except Exception as e:
            print(f"[DB_DEBUG] ERROR in store_audio_and_update_call: {e}")
            import traceback
            traceback.print_exc()
            return {"stored_audio": {}}
            
    def insert_partial_update(self, call_id, duration, cqs, adherence, emotions, transcription, quality):
        """
        Insert partial update using the new optimized flow:
        - Updates conversation segments with the newer transcription results
        - Maintains chunk-based emotion tracking with atomic operations
        - Ensures all data stays consistent within the main call schema
        """
        try:
            print(f"[PARTIAL_UPDATE] Starting insert_partial_update for call_id: {call_id}")
            
            # STEP 1: Build conversation array from transcription data
            conversation = []
            if isinstance(transcription, dict):
                agent_segments = transcription.get('agent_segments', [])
                client_segments = transcription.get('client_segments', [])
                conversation = build_conversation_from_segments(agent_segments, client_segments)
                print(f"[PARTIAL_UPDATE] Built conversation with {len(conversation)} total segments")
            
            # STEP 2: Calculate processing statistics
            current_timestamp = datetime.now(timezone.utc)
            agent_text_length = len(transcription.get('agent', '')) if isinstance(transcription, dict) else 0
            client_text_length = len(transcription.get('client', '')) if isinstance(transcription, dict) else 0
            
            print(f"[PARTIAL_UPDATE] Processing stats - Agent text: {agent_text_length} chars, Client text: {client_text_length} chars")
            
            # STEP 3: Atomic update of main call fields with new conversation segments
            print(f"[PARTIAL_UPDATE] Step 3: Updating main call fields and conversation...")
            result = self.calls_collection.find_one_and_update(
                {"call_id": call_id},
                {
                    "$set": {
                        "cqs": cqs,
                        "adherence": adherence,
                        "quality": quality,
                        "status": "in_progress",
                        "last_updated": current_timestamp,
                        # UPDATED: Store the complete updated conversation array
                        "conversation": conversation
                    },
                    "$inc": {
                        "duration": duration,
                        # Track total chunks processed
                        "chunk_count": 1
                    }
                },
                return_document=True  # Return updated document
            )
            
            if result:
                chunk_number = result.get("chunk_count", 1)
                print(f"[PARTIAL_UPDATE] Successfully updated call, now at chunk #{chunk_number}")
                
                # STEP 4: Add chunk emotion entry to emotions array
                chunk_emotion_entry = {
                    "chunk_number": chunk_number,
                    "timestamp": current_timestamp,
                    "duration": duration,
                    "emotions": emotions,
                    "cqs": cqs,
                    "quality": quality,
                    "agent_text_length": agent_text_length,
                    "client_text_length": client_text_length,
                    # Additional context for this chunk
                    "processing_context": {
                        "total_conversation_segments": len(conversation),
                        "agent_segments_count": len([s for s in conversation if s.get('speaker') == 'agent']),
                        "client_segments_count": len([s for s in conversation if s.get('speaker') == 'client'])
                    }
                }
                
                # STEP 5: Append emotion chunk to emotions array
                print(f"[PARTIAL_UPDATE] Step 5: Adding emotion chunk #{chunk_number}...")
                self.calls_collection.update_one(
                    {"call_id": call_id},
                    {"$push": {"emotions": chunk_emotion_entry}}
                )
                
                print(f"[PARTIAL_UPDATE] Successfully completed partial update for chunk #{chunk_number}")
                
                # STEP 6: Optional verification and logging
                updated_call = self.calls_collection.find_one({"call_id": call_id}, {
                    "chunk_count": 1, 
                    "conversation": 1, 
                    "emotions": 1,
                    "status": 1
                })
                
                if updated_call:
                    print(f"[PARTIAL_UPDATE] Verification - Call status: {updated_call.get('status')}")
                    print(f"[PARTIAL_UPDATE] Verification - Total chunks: {updated_call.get('chunk_count', 0)}")
                    print(f"[PARTIAL_UPDATE] Verification - Conversation segments: {len(updated_call.get('conversation', []))}")
                    print(f"[PARTIAL_UPDATE] Verification - Emotion chunks: {len(updated_call.get('emotions', []))}")
                
            else:
                print(f"[PARTIAL_UPDATE] WARNING: Call document not found or update failed for call_id: {call_id}")
            
        except Exception as e:
            print(f"[PARTIAL_UPDATE] ERROR in insert_partial_update: {e}")
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
        """
        Store audio chunk and process segments using the new optimized flow:
        1. Store new chunks in local cache (combined with previous)
        2. Get complete audio from last segment start to current chunk end from local cache
        3. Return this audio segment for processing (adherence, emotions, etc.)
        At final update, save the complete audio to MongoDB and clean up local cache.
        """
        try:
            print(f"[CHUNK_PROCESS] Starting store_audio_chunk_and_process for call_id: {call_id}")
            with self.lock:
                # STEP 1: Store new audio chunks (combine with existing)
                print(f"[CHUNK_PROCESS] Step 1: Storing audio chunks...")
                # Always store in local cache, never in GridFS here
                stored_audio = self.store_audio_and_update_call(call_id, client_audio, agent_audio, is_final=False)
                print(f"[CHUNK_PROCESS] Audio stored: {stored_audio}")
                # STEP 2: Get current conversation segments to determine last segment times
                print(f"[CHUNK_PROCESS] Step 2: Getting current conversation segments...")
                conversation = self.get_conversation(call_id)
                agent_segments = []
                client_segments = []
                last_agent_segment_start = 0.0
                last_client_segment_start = 0.0
                for entry in conversation:
                    converted_entry = {
                        'text': entry.get('text', ''),
                        'start': entry.get('start_time', 0),
                        'end': entry.get('end_time', 0),
                        'confidence': entry.get('confidence', 1.0)
                    }
                    if entry.get('speaker') == 'agent':
                        agent_segments.append(converted_entry)
                        last_agent_segment_start = max(last_agent_segment_start, entry.get('start_time', 0))
                    elif entry.get('speaker') == 'client':
                        client_segments.append(converted_entry)
                        last_client_segment_start = max(last_client_segment_start, entry.get('start_time', 0))
                print(f"[CHUNK_PROCESS] Found {len(agent_segments)} agent segments, {len(client_segments)} client segments")
                print(f"[CHUNK_PROCESS] Last agent segment start: {last_agent_segment_start}s")
                print(f"[CHUNK_PROCESS] Last client segment start: {last_client_segment_start}s")
                # STEP 3: Get complete combined audio from local cache
                print(f"[CHUNK_PROCESS] Step 3: Getting complete combined audio from local cache...")
                complete_client_audio = self._get_previous_audio_from_local_cache(call_id, 'client')
                complete_agent_audio = self._get_previous_audio_from_local_cache(call_id, 'agent')
                print(f"[CHUNK_PROCESS] Complete agent audio: {len(complete_agent_audio) if complete_agent_audio else 0} bytes")
                print(f"[CHUNK_PROCESS] Complete client audio: {len(complete_client_audio) if complete_client_audio else 0} bytes")
                # STEP 4: Extract audio from last segment start to end of current chunk
                print(f"[CHUNK_PROCESS] Step 4: Extracting audio from last segment start to chunk end...")
                processing_agent_audio_pcm = self._slice_audio_from_time(complete_agent_audio, last_agent_segment_start)
                processing_client_audio_pcm = self._slice_audio_from_time(complete_client_audio, last_client_segment_start)
                print(f"[CHUNK_PROCESS] Agent PCM audio (from {last_agent_segment_start}s): {len(processing_agent_audio_pcm) if processing_agent_audio_pcm else 0} bytes")
                print(f"[CHUNK_PROCESS] Client PCM audio (from {last_client_segment_start}s): {len(processing_client_audio_pcm) if processing_client_audio_pcm else 0} bytes")
                # STEP 5: Convert PCM data to proper WAV format for API compatibility
                print(f"[CHUNK_PROCESS] Step 5: Converting PCM to WAV format...")
                processing_agent_audio = self._pcm_data_to_wav_bytesio(processing_agent_audio_pcm)
                processing_client_audio = self._pcm_data_to_wav_bytesio(processing_client_audio_pcm)
                print(f"[CHUNK_PROCESS] Agent WAV audio for processing: {processing_agent_audio.getvalue().__len__() if processing_agent_audio else 0} bytes")
                print(f"[CHUNK_PROCESS] Client WAV audio for processing: {processing_client_audio.getvalue().__len__() if processing_client_audio else 0} bytes")
                # STEP 6: Prepare result with all necessary data
                result = {
                    "stored_audio": stored_audio.get("stored_audio", {}),
                    "agent_segments": agent_segments,
                    "client_segments": client_segments,
                    "agent_overlap_start": last_agent_segment_start,
                    "client_overlap_start": last_client_segment_start,
                    "agent_audio_for_transcription": processing_agent_audio,
                    "client_audio_for_transcription": processing_client_audio,
                    "processing_context": {
                        "last_agent_segment_start": last_agent_segment_start,
                        "last_client_segment_start": last_client_segment_start,
                        "total_agent_segments": len(agent_segments),
                        "total_client_segments": len(client_segments),
                        "complete_audio_available": {
                            "agent": complete_agent_audio is not None,
                            "client": complete_client_audio is not None
                        }
                    }
                }
                print(f"[CHUNK_PROCESS] store_audio_chunk_and_process completed successfully")
                return result
        except Exception as e:
            print(f"[CHUNK_PROCESS] ERROR in store_audio_chunk_and_process: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stored_audio": {},
                "agent_segments": [],
                "client_segments": [],
                "agent_overlap_start": 0.0,
                "client_overlap_start": 0.0,
                "agent_audio_for_transcription": None,
                "client_audio_for_transcription": None,
                "processing_context": {
                    "last_agent_segment_start": 0.0,
                    "last_client_segment_start": 0.0,
                    "total_agent_segments": 0,
                    "total_client_segments": 0,
                    "complete_audio_available": {"agent": False, "client": False}
                }
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
            # For final update, fetch from GridFS; otherwise, fetch from local cache
            audio_metadata = call.get('audio_metadata', {})
            client_file_id = audio_metadata.get('client_file_id')
            agent_file_id = audio_metadata.get('agent_file_id')
            # Retrieve audio data
            client_audio = None
            agent_audio = None
            if client_file_id and agent_file_id:
                # If both exist, assume final update, fetch from GridFS
                client_audio = None
                agent_audio = None
                try:
                    client_audio = self.fs.get(ObjectId(client_file_id)).read()
                except Exception:
                    pass
                try:
                    agent_audio = self.fs.get(ObjectId(agent_file_id)).read()
                except Exception:
                    pass
            else:
                # Otherwise, fetch from local cache
                client_audio = self._read_audio_cache(call_id, 'client')
                agent_audio = self._read_audio_cache(call_id, 'agent')
            # Get segments from conversation and convert to expected format
            conversation = call.get('conversation', [])
            agent_segments = []
            client_segments = []
            for entry in conversation:
                converted_entry = {
                    'text': entry.get('text', ''),
                    'start': entry.get('start_time', 0),
                    'end': entry.get('end_time', 0),
                    'confidence': entry.get('confidence', 1.0)
                }
                if entry.get('speaker') == 'agent':
                    agent_segments.append(converted_entry)
                elif entry.get('speaker') == 'client':
                    client_segments.append(converted_entry)
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

    def _get_audio_cache_path(self, call_id, speaker_type):
        return os.path.join(self.audio_cache_dir, f'{call_id}_{speaker_type}.pcm')

    def _append_to_audio_cache(self, call_id, speaker_type, pcm_data):
        if not pcm_data:
            return
        path = self._get_audio_cache_path(call_id, speaker_type)
        with open(path, 'ab') as f:
            f.write(pcm_data)

    def _read_audio_cache(self, call_id, speaker_type):
        path = self._get_audio_cache_path(call_id, speaker_type)
        if not os.path.exists(path):
            return b''
        with open(path, 'rb') as f:
            return f.read()

    def _delete_audio_cache(self, call_id):
        for speaker_type in ['agent', 'client']:
            path = self._get_audio_cache_path(call_id, speaker_type)
            if os.path.exists(path):
                os.remove(path)


# Singleton instance to be used by the app
def get_call_operations():
    """Get CallOperations instance with database connection"""
    db = get_db()
    return CallOperations(db)