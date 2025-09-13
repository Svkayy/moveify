#!/bin/bash

# Dance Sync Analysis Installation Script
# This script helps install all dependencies for the dance sync analysis tool

echo "Dance Sync Analysis - Installation Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✓ pip3 found: $(pip3 --version)"

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
    echo "To activate in the future, run: source venv/bin/activate"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed successfully"
else
    echo "✗ Failed to install Python dependencies"
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: FFmpeg is not installed. This is required for video processing."
    echo "Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
else
    echo "✓ FFmpeg found: $(ffmpeg -version | head -n1)"
fi

# Check for Praat (optional)
if ! command -v praat &> /dev/null; then
    echo "Note: Praat is not installed. This is optional for advanced audio analysis."
    echo "To install Praat:"
    echo "  Download from https://www.fon.hum.uva.nl/praat/download_mac.html"
else
    echo "✓ Praat found: $(praat --version 2>/dev/null || echo 'installed')"
fi

# Create video directory
mkdir -p video
echo "✓ Created video directory"

# Make scripts executable
chmod +x dance.py
chmod +x test_dance_sync.py
chmod +x example_usage.py
echo "✓ Made scripts executable"

echo ""
echo "Installation completed!"
echo "======================"
echo ""
echo "Next steps:"
echo "1. Add your dance videos to the 'video/' directory"
echo "2. Run the test suite: python3 test_dance_sync.py"
echo "3. Run the analysis: python3 dance.py video/your_dance.mov video/model_dance.mov"
echo ""
echo "For more information, see README.md"
