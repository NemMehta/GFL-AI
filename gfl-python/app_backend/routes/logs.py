# app_backend/routes/logs.py

import os
from flask import Blueprint, jsonify, Response, stream_with_context

LOG_FILE = os.path.join("logs", "app.log")

logs_bp = Blueprint("logs", __name__)

# -------------------- Get all logs (latest snapshot) --------------------
@logs_bp.route("/logs", methods=["GET"])
def get_logs():
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"status": "error", "message": "Log file not found"}), 404

        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.readlines()

        return jsonify({
            "status": "success",
            "lines": [line.strip() for line in content[-200:]]  # last 200 lines
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------- Stream logs live (Server-Sent Events) --------------------
@logs_bp.route("/logs/stream", methods=["GET"])
def stream_logs():
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"status": "error", "message": "Log file not found"}), 404

        def generate():
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                f.seek(0, os.SEEK_END)  # start at end of file
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {line.strip()}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
