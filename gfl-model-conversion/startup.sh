


# #!/bin/bash
# echo "Starting Gunicorn..."
# gunicorn --bind=0.0.0.0:8000 app:app


#!/bin/bash
set -e

echo "=== Azure Startup Script: YOLO Conversion API ==="

# -----------------------------
# 1. Install runtime system packages (non-persistent)
# -----------------------------
echo "=== Installing system packages ==="
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

# -----------------------------
# 2. Setup persistent virtual environment
# -----------------------------
VENV_DIR="/home/site/venv"

echo "=== Checking virtual environment at $VENV_DIR ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating new persistent venv ==="
    python3 -m venv $VENV_DIR
else
    echo "=== Using existing venv ==="
fi

# Activate venv
echo "=== Activating venv ==="
source $VENV_DIR/bin/activate

# -----------------------------
# 3. Install requirements
# -----------------------------
echo "=== Installing Python dependencies ==="
pip install --upgrade pip

# Install all dependencies NEW and UPGRADED
pip install --upgrade -r /home/site/wwwroot/requirements.txt

echo "=== Python dependencies installed successfully ==="

# -----------------------------
# 4. Start Gunicorn
# -----------------------------
# APP_MODULE="app:app"
PORT=${PORT:-8000}   # Azure automatically sets PORT

echo "=== Starting Gunicorn on port $PORT ==="
exec gunicorn --bind=0.0.0.0:$PORT --timeout 1500 convertion:app



















# #!/bin/bash
# set -e

# echo "=== Azure Startup Script ==="

# VENV_DIR="/home/site/venv"
# REQ="/home/site/wwwroot/requirements.txt"

# if [ ! -d "$VENV_DIR" ]; then
#     echo "=== venv NOT found. Creating and installing requirements ==="
#     python3 -m venv $VENV_DIR
#     source $VENV_DIR/bin/activate
#     pip install --upgrade pip
#     pip install -r $REQ
# else
#     echo "=== venv found. Activating ==="
#     source $VENV_DIR/bin/activate

#     echo "=== Ensuring all dependencies exist ==="
#     pip install --upgrade pip
#     pip install -r $REQ
# fi

# echo "=== Ensuring gunicorn exists ==="
# /home/site/venv/bin/pip install gunicorn

# echo "=== Starting Gunicorn ==="
# exec /home/site/venv/bin/gunicorn \
#     --bind 0.0.0.0:${PORT:-8000} \
#     --workers 1 \
#     --threads 4 \
#     --timeout 600 \
#     app:app
# #----------------------------------
