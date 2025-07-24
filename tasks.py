from celery import Celery
import redis
import os
from datetime import datetime, timezone
import traceback
from io import BytesIO
import concurrent.futures
import sys
from app2_segment_utils import (
    speech_to_text, call_summary, get_tags, resegment_based_on_punctuation, merge_transcription_segments, process_whisper_segments,
    check_script_adherence_adaptive_windows, get_current_call_script, get_response, calculate_cqs, get_quality, build_conversation_from_segments
)
from call_queue_manager import call_queue_manager


# Initialize Celery app
app = Celery('tasks', 
             broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
             backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'))

# Redis client for locking (separate DB to avoid conflicts)
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/1'))

# Import your processing functions
from call_operations import get_call_operations
from db_config import get_db

# Import processing functions from the original app
import sys
import importlib.util

# Remove acquire_call_lock and release_call_lock and all lock usage

@app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_create_call_job(self, call_id, agent_id, sip_id, script_text):
    """
    Process create_call job with per-call_id locking
    """
    try:
        print(f"[TASK] Starting create_call job for call_id: {call_id}")
        call_ops = get_call_operations()
        success = call_ops.create_call(call_id, agent_id, sip_id, script_text)
        if success:
            print(f"[TASK] Successfully created call: {call_id}")
            return {"status": "success", "call_id": call_id}
        else:
            raise Exception(f"Failed to create call: {call_id}")
    except Exception as e:
        print(f"[TASK ERROR] Error in create_call job for {call_id}: {e}")
        traceback.print_exc()
        raise
    finally:
        call_queue_manager.job_done(call_id)

@app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_update_call_job(self, call_id, client_audio_data, agent_audio_data, transcript):
    """
    Process update_call job with per-call_id locking
    """
    try:
        print(f"[TASK] Starting update_call job for call_id: {call_id}")
        # Reconstruct audio files from bytes
        client_audio = BytesIO(client_audio_data) if client_audio_data else None
        agent_audio = BytesIO(agent_audio_data) if agent_audio_data else None
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)
        if not existing_call:
            raise Exception(f"No record found for call_id: {call_id}")
        processing_data = call_ops.store_audio_chunk_and_process(call_id, client_audio, agent_audio)
        previous_agent_segments = processing_data.get("agent_segments", [])
        previous_client_segments = processing_data.get("client_segments", [])
        last_agent_segment_start = processing_data.get("agent_overlap_start", 0.0)
        last_client_segment_start = processing_data.get("client_overlap_start", 0.0)
        agent_audio_for_transcription = processing_data.get("agent_audio_for_transcription")
        client_audio_for_transcription = processing_data.get("client_audio_for_transcription")
        def transcribe_agent_audio():
            if agent_audio_for_transcription:
                _, _, _, all_words = speech_to_text(agent_audio_for_transcription)
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_agent_segment_start
                    segment['end'] += last_agent_segment_start
                return adjusted_segments
            return []
        def transcribe_client_audio():
            if client_audio_for_transcription:
                _, _, _, all_words = speech_to_text(client_audio_for_transcription)
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_client_segment_start
                    segment['end'] += last_client_segment_start
                return adjusted_segments
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            agent_future = executor.submit(transcribe_agent_audio)
            client_future = executor.submit(transcribe_client_audio)
            new_agent_segments = agent_future.result()
            new_client_segments = client_future.result()
        all_agent_segments = merge_transcription_segments(previous_agent_segments, new_agent_segments, last_agent_segment_start)
        all_client_segments = merge_transcription_segments(previous_client_segments, new_client_segments, last_client_segment_start)
        all_agent_segments = sorted(all_agent_segments, key=lambda x: x.get('start', 0))
        all_client_segments = sorted(all_client_segments, key=lambda x: x.get('start', 0))
        total_agent_text = " ".join([s['text'] for s in all_agent_segments]).strip()
        total_client_text = " ".join([s['text'] for s in all_client_segments]).strip()
        if not total_agent_text and not total_client_text:
            print(f"[TASK] No text to analyze for call_id: {call_id}")
            return {"status": "success", "message": "Audio silent, no analysis performed."}
        duration = max(len(client_audio_data) if client_audio_data else 0, 
                      len(agent_audio_data) if agent_audio_data else 0) / 32000
        script_checkpoints = get_current_call_script(transcript)
        
        combined_transcription = f"Agent: {total_agent_text}\nClient: {total_client_text}".strip()
        processed_agent_segments = process_whisper_segments(all_agent_segments)
        processed_client_segments = process_whisper_segments(all_client_segments)
        agent_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Agent', 'text': s['text']} for s in processed_agent_segments]
        client_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Client', 'text': s['text']} for s in processed_client_segments]
        all_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])
        all_transcription = {
            "agent": total_agent_text,
            "client": total_client_text,
            "combined": combined_transcription,
            "timestamped_dialogue": all_timestamped_dialogue,
            "agent_segments": all_agent_segments,
            "client_segments": all_client_segments
        }
        
        results = check_script_adherence_adaptive_windows(total_agent_text, processed_agent_segments, script_checkpoints)
        overall_adherence = results.get("real_time_adherence_score", 0)
        script_completion = results.get("script_completion_percentage", 0)
        adherence_data = {
            'overall': overall_adherence,
            'script_completion': script_completion,
            'details': results.get("checkpoint_results", []),
            'window_size_usage': results.get("window_size_usage", {})
        }
        # response = get_response(total_client_text)
        response=[[
        {"label": "joy", "score": 0.9},
        {"label": "neutral", "score": 0.1}
    ]]
        CQS, emotions = calculate_cqs(response, adherence_data.get('script_completion', 0.0))
        quality = get_quality(emotions)
        call_ops.insert_partial_update(call_id, duration, CQS, adherence_data, emotions, all_transcription, quality)
        print(f"[TASK] Successfully processed update_call for call_id: {call_id}")
        return {
            "status": "success",
            "call_id": call_id,
            "overall_adherence": overall_adherence,
            "script_completion": script_completion
        }
    except Exception as e:
        print(f"[TASK ERROR] Error in update_call job for {call_id}: {e}")
        traceback.print_exc()
        raise
    finally:
        call_queue_manager.job_done(call_id)

@app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_final_update_job(self, call_id, client_audio_data, agent_audio_data, transcript):
    """
    Process final_update job with per-call_id locking
    """
    try:
        print(f"[TASK] Starting final_update job for call_id: {call_id}")
        from call_operations import get_call_operations
        call_ops = get_call_operations()
        existing_call = call_ops.get_call(call_id)
        if not existing_call:
            raise Exception(f"No record found for call_id: {call_id}")
        from io import BytesIO
        client_audio = BytesIO(client_audio_data) if client_audio_data else None
        agent_audio = BytesIO(agent_audio_data) if agent_audio_data else None
        processing_data = call_ops.store_audio_chunk_and_process(call_id, client_audio, agent_audio)
        previous_agent_segments = processing_data.get("agent_segments", [])
        previous_client_segments = processing_data.get("client_segments", [])
        last_agent_segment_start = processing_data.get("agent_overlap_start", 0.0)
        last_client_segment_start = processing_data.get("client_overlap_start", 0.0)
        agent_audio_for_transcription = processing_data.get("agent_audio_for_transcription")
        client_audio_for_transcription = processing_data.get("client_audio_for_transcription")
        import concurrent.futures
        def transcribe_agent_audio():
            if agent_audio_for_transcription:
                _, _, _, all_words = speech_to_text(agent_audio_for_transcription)
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_agent_segment_start
                    segment['end'] += last_agent_segment_start
                return adjusted_segments
            return []
        def transcribe_client_audio():
            if client_audio_for_transcription:
                _, _, _, all_words = speech_to_text(client_audio_for_transcription)
                adjusted_segments = resegment_based_on_punctuation(all_words)
                for segment in adjusted_segments:
                    segment['start'] += last_client_segment_start
                    segment['end'] += last_client_segment_start
                return adjusted_segments
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            agent_future = executor.submit(transcribe_agent_audio)
            client_future = executor.submit(transcribe_client_audio)
            new_agent_segments = agent_future.result()
            new_client_segments = client_future.result()
        final_agent_segments = merge_transcription_segments(previous_agent_segments, new_agent_segments, last_agent_segment_start)
        final_client_segments = merge_transcription_segments(previous_client_segments, new_client_segments, last_client_segment_start)
        final_agent_segments = sorted(final_agent_segments, key=lambda x: x.get('start', 0))
        final_client_segments = sorted(final_client_segments, key=lambda x: x.get('start', 0))
        final_agent_text = " ".join([s['text'] for s in final_agent_segments]).strip()
        final_client_text = " ".join([s['text'] for s in final_client_segments]).strip()
        if not final_agent_text and not final_client_text:
            call_ops.complete_call_update(
                call_id=call_id,
                agent_text="", client_text="",
                combined="", cqs=0,
                overall_adherence={}, agent_quality={},
                summary="", emotions={}, duration=existing_call.get('duration', 0),
                quality=0, tags="", timestamped_dialogue=[],
                agent_segments=[], client_segments=[]
            )
            return {"status": "success", "message": "Call completed with no transcription data."}
        chunk_duration = max(len(client_audio_data) if client_audio_data else 0, len(agent_audio_data) if agent_audio_data else 0) / 32000
        total_duration = existing_call.get('duration', 0) + chunk_duration
        script_checkpoints = get_current_call_script(transcript)
        combined_transcription = f"Agent: {final_agent_text}\nClient: {final_client_text}".strip()
        processed_agent_segments = process_whisper_segments(final_agent_segments)
        processed_client_segments = process_whisper_segments(final_client_segments)
        agent_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Agent', 'text': s['text']} for s in processed_agent_segments]
        client_dialogues = [{'timestamp': s.get('start', 0), 'speaker': 'Client', 'text': s['text']} for s in processed_client_segments]
        final_timestamped_dialogue = sorted(agent_dialogues + client_dialogues, key=lambda x: x['timestamp'])
        results = check_script_adherence_adaptive_windows(final_agent_text, processed_agent_segments, script_checkpoints)
        final_adherence_data = {
            "overall": results.get("real_time_adherence_score", 0),
            "script_completion": results.get("script_completion_percentage", 0),
            "details": results.get("checkpoint_results", []),
            "window_size_usage": results.get("window_size_usage", {}),
            "method": "adaptive_windows_segments"
        }
        # final_emotion_response = get_response(final_client_text)
        final_emotion_response=[[
        {"label": "joy", "score": 0.9},
        {"label": "neutral", "score": 0.1}
    ]]
        final_CQS, final_emotions = calculate_cqs(final_emotion_response, final_adherence_data.get('script_completion', 0.0))
        final_quality = get_quality(final_emotions)
        # response = get_response(final_client_text)
        response = [[
        {"label": "joy", "score": 0.9},
        {"label": "neutral", "score": 0.1}
    ]]
        CQS, emotions = calculate_cqs(response, final_adherence_data.get('script_completion', 0.0))
        quality = get_quality(emotions)
        agent_quality = {}
        summary = call_summary(final_agent_text, final_client_text)
        tags_obj = get_tags(f"Client:\n{final_client_text}\nAgent:\n{final_agent_text}")
        tags = ', '.join(tags_obj.get('Tags', []))
        call_ops.complete_call_update(
            call_id=call_id,
            agent_text=final_agent_text, client_text=final_client_text,
            combined=combined_transcription, cqs=final_CQS,
            overall_adherence=final_adherence_data, agent_quality=agent_quality,
            summary=summary, emotions=emotions, duration=total_duration,
            quality=final_quality, tags=tags, timestamped_dialogue=final_timestamped_dialogue,
            agent_segments=final_agent_segments, client_segments=final_client_segments,
            final_emotions=final_emotions
        )
        print(f"[TASK] Successfully processed final_update for call_id: {call_id}")
        return {"status": "success", "call_id": call_id}
    except Exception as e:
        print(f"[TASK ERROR] Error in final_update job for {call_id}: {e}")
        traceback.print_exc()
        raise
    finally:
        call_queue_manager.job_done(call_id)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,  # Important for per-call_id serialization
    task_acks_late=True,
    worker_max_tasks_per_child=100,
) 