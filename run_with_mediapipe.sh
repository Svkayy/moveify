#!/bin/bash

# Script to run Dance Sync Analysis with MediaPipe in the correct Python environment

echo "Dance Sync Analysis - MediaPipe Version"
echo "======================================"

# Activate the virtual environment with MediaPipe
source venv_mediapipe/bin/activate

# Check if videos are provided as arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./run_with_mediapipe.sh video1.mov video2.mov"
    echo ""
    echo "Available videos in video/ directory:"
    ls -la video/ 2>/dev/null || echo "No videos found in video/ directory"
    echo ""
    echo "To add videos:"
    echo "1. Copy your dance videos to the video/ directory"
    echo "2. Make sure they are in .mov format"
    echo "3. Run: ./run_with_mediapipe.sh video/your_dance.mov video/model_dance.mov"
    exit 1
fi

# Check if both video files exist
if [ ! -f "$1" ]; then
    echo "Error: Video file '$1' not found"
    exit 1
fi

if [ ! -f "$2" ]; then
    echo "Error: Video file '$2' not found"
    exit 1
fi

echo "Running dance sync analysis..."
echo "Video 1: $1"
echo "Video 2: $2"
echo ""

# Run the analysis
python3 dance.py "$1" "$2"

echo ""
echo "Analysis complete! Check the output files:"
echo "- comparison_output.mp4 (side-by-side comparison video)"
echo "- dance_analysis_report.json (detailed results)"
echo "- sync_scores.png (score visualization)"
