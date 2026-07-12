

#!/bin/bash
set -e

echo "=== Installing system packages (temporary each restart) ==="
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0


echo "=== Checking for persistent virtual environment at /home/site/venv ==="

if [ ! -d "/home/site/venv" ]; then
    echo "=== Creating new persistent venv ==="
    python3 -m venv /home/site/venv
    /home/site/venv/bin/pip install --upgrade pip
    /home/site/venv/bin/pip install -r /home/site/wwwroot/requirements.txt
else
    echo "=== Using existing venv ==="
    source /home/site/venv/bin/activate
    /home/site/venv/bin/pip install --upgrade pip
    /home/site/venv/bin/pip install -r /home/site/wwwroot/requirements.txt --upgrade
fi

echo "=== Starting Gunicorn ==="
source /home/site/venv/bin/activate
exec gunicorn --bind=0.0.0.0:$PORT --timeout 1500 app:app





# #!/bin/bash

# echo "=== Updating package lists and installing system libraries ==="
# apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# echo "=== Upgrading pip and installing Python packages ==="
# pip install --upgrade pip
# pip install -r /home/site/wwwroot/requirements.txt

# echo "=== Starting Gunicorn ==="
# gunicorn --bind=0.0.0.0:$PORT --timeout 600 app:app









# #!/bin/bash

# echo "=== Updating package lists and installing system libraries ==="
# apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# export PERSISTENT_VENV=/home/site/venv
# export PYTHONPATH=$PERSISTENT_VENV/lib/python3.10/site-packages:$PYTHONPATH

# mkdir -p $PERSISTENT_VENV/lib/python3.10/site-packages

# echo "=== Installing Python packages if not already installed ==="
# if [ ! -d "$PERSISTENT_VENV/lib/python3.10/site-packages/ultralytics" ]; then
#     echo "📦 ultralytics missing — installing only that package..."
#     pip install --upgrade pip
#     pip install --target=$PERSISTENT_VENV/lib/python3.10/site-packages ultralytics --force-reinstall
# else
#     echo "✅ Packages already installed. Skipping pip install."
# fi

# echo "=== Starting Gunicorn ==="
# gunicorn --bind=0.0.0.0:$PORT --timeout 600 app:app
