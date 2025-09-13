#!/bin/bash

# Launch the Dance Sync Analysis GUI with MediaPipe support

echo "Launching Dance Sync Analysis GUI..."

# Check if virtual environment exists
if [ ! -d "venv_mediapipe" ]; then
    echo "Error: Virtual environment not found. Please run the installation first."
    echo "Run: ./install.sh"
    exit 1
fi

# Activate virtual environment and launch GUI
source venv_mediapipe/bin/activate

# Check if required packages are installed
python3 -c "import mediapipe, cv2, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required packages not installed. Please run:"
    echo "source venv_mediapipe/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Launch the GUI
python3 dance_gui.py
