import os
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from tasks import process_create_call_job, process_update_call_job, process_final_update_job
from io import BytesIO
import traceback
from db_config import get_db
from call_operations import get_call_operations
from call_queue_manager import call_queue_manager

app = Flask(__name__)
CORS(app)
load_dotenv()

def load_script_from_file(file_path="audioscript.txt"):
    try:
        script_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            return None
    except Exception as e:
        print(f"[SCRIPT LOADER] Error loading script file: {e}")
        return None

@app.route('/create_call', methods=['POST'])
def create_call():
    if 'call_id' not in request.form:
        return jsonify({"error": "call_id is a required field"}), 400
    call_id = request.form.get('call_id')
    agent_id = request.form.get('agent_id')
    sip_id = request.form.get('sip_id')
    script_text = request.form.get('transcript')
    try:
        call_queue_manager.add_job(
            call_id,
            process_create_call_job,
            args=(call_id, agent_id, sip_id, script_text),
            kwargs={}
        )
        return jsonify({
            "status": "queued", 
            "message": f"Call creation for {call_id} has been queued for processing."
        }), 202
    except Exception as e:
        print(f"[QUEUE ERROR] Failed to enqueue create_call job: {e}")
        return jsonify({"error": "Failed to queue call creation job"}), 500

@app.route('/update_call', methods=['POST'])
def update_call():
    call_id = request.form.get('call_id')
    if not call_id:
        return jsonify({"error": "call_id is required"}), 400
    client_audio_data = None
    agent_audio_data = None
    client_audio_chunk = request.files.get('client_audio')
    agent_audio_chunk = request.files.get('agent_audio')
    if client_audio_chunk:
        client_audio_chunk.seek(0)
        client_audio_data = client_audio_chunk.read()
    if agent_audio_chunk:
        agent_audio_chunk.seek(0)
        agent_audio_data = agent_audio_chunk.read()
    transcript = request.form.get('transcript')
    if not transcript:
        transcript = load_script_from_file("audioscript.txt")
    try:
        call_queue_manager.add_job(
            call_id,
            process_update_call_job,
            args=(call_id, client_audio_data, agent_audio_data, transcript),
            kwargs={}
        )
        return jsonify({
            "status": "queued", 
            "message": f"Update call for {call_id} has been queued for processing."
        }), 202
    except Exception as e:
        print(f"[UPDATE_CALL ERROR] Failed to enqueue update_call job: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to queue update call job"}), 500

@app.route('/final_update', methods=['POST'])
def final_update():
    call_id = request.form.get('call_id')
    if not call_id:
        return jsonify({"error": "call_id is required"}), 400
    client_audio_data = None
    agent_audio_data = None
    client_audio = request.files.get('client_audio')
    agent_audio = request.files.get('agent_audio')
    if client_audio:
        client_audio.seek(0)
        client_audio_data = client_audio.read()
    if agent_audio:
        agent_audio.seek(0)
        agent_audio_data = agent_audio.read()
    transcript = request.form.get('transcript')
    if not transcript:
        transcript = load_script_from_file("audioscript.txt")
    try:
        call_queue_manager.add_job(
            call_id,
            process_final_update_job,
            args=(call_id, client_audio_data, agent_audio_data, transcript),
            kwargs={}
        )
        return jsonify({
            "status": "queued", 
            "message": f"Final update for {call_id} has been queued for processing."
        }), 202
    except Exception as e:
        print(f"[FINAL_UPDATE ERROR] Failed to enqueue final_update job: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to queue final update job"}), 500

@app.route('/cleanup_call', methods=['POST'])
def cleanup_call():
    call_id = request.form.get('call_id')
    if not call_id:
        return jsonify({"error": "call_id is required"}), 400
    try:
        call_ops = get_call_operations()
        call_ops.cleanup_call_resources(call_id)
        return jsonify({
            "status": "success",
            "message": f"Resources cleaned up for call {call_id}"
        })
    except Exception as e:
        print(f"[CLEANUP_ERROR] Error cleaning up call {call_id}: {e}")
        return jsonify({"error": "Failed to cleanup call resources"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        call_ops = get_call_operations()
        db_status = "connected" if call_ops.db.client.admin.command('ping') else "disconnected"
        return jsonify({
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = '0.0.0.0'
    threaded = True
    processes = 1
    print(f"[SERVER] Starting Flask server at http://{host}:{port}")
    try:
        app.run(host=host, port=port, debug=debug, threaded=threaded, processes=processes, use_reloader=False)
    except Exception as e:
        print(f"[SERVER ERROR] Failed to start server: {e}")
    finally:
        try:
            mongodb = get_db()
            mongodb.close()
            print("[SERVER] Database connection closed")
        except:
            pass
        print("[SERVER] Server stopped") 