

import os

from flask import Flask
# from app_backend.routes.fish import fish_bp
from app_backend.routes.fish import fish_bp
from app_backend.config import STATIC_DIR  # <-- add this import
from app_backend.routes.logs import logs_bp
import logging
from logging.handlers import TimedRotatingFileHandler


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Make sure logs folder exists

log_file = os.path.join(log_dir, "app.log")


# Set up timed rotating log handler: rotate every hour, keep only last 2 files
handler = TimedRotatingFileHandler(
    log_file,
    when="H",           # Rotate every hour
    interval=1,
    backupCount=2,      # Keep logs for 2 hours max
    encoding="utf-8",
    utc=True,            # Optional: Use UTC for timestamps
    delay=True
)

# Set format
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)

# Get root logger and configure
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')
# Set max upload size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

app.register_blueprint(fish_bp)
app.register_blueprint(logs_bp, url_prefix="/api")

print("Serving /static from:", app.static_folder)  # quick sanity log

@app.route('/')
def landing():
    return "Server is running", 200

@app.route('/health')
def health():
    return "health is running", 200


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000, threaded=True)
