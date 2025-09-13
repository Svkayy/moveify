# Moveify - Dance Sync Analysis

**Upload any dance clip (or record yourself), and we auto-turn it into beat-synced steps that coach you in real time—then score your accuracy.**

Dance Sync is a program that allows you to compare your dancing to a model dancer's performance. When learning a dance, it can be hard to pinpoint where you're going wrong by just looking in the mirror. Instead, record a video of your dancing and plug it into Dance Sync - it will analyze your movement and compare it to the video of the original dancer.

You will get a comparison video with alerts where you are not in sync and a final score at the end.

## Features

- **Pose Estimation**: Uses MediaPipe to detect and track human pose landmarks
- **Audio Synchronization**: Automatically syncs videos using cross-correlation of audio tracks
- **Angle Analysis**: Compares limb angles between dancers to measure synchronization
- **Visual Comparison**: Creates side-by-side comparison videos with sync indicators
- **Detailed Reporting**: Generates comprehensive analysis reports with limb-specific scores
- **Score Visualization**: Creates graphs showing sync scores over time
- **GUI Interface**: Easy-to-use graphical interface for video selection
- **Real-time Coaching**: Beat-synced steps that coach you in real time

## How It Works

### 1. Pose Comparison

We compare the **angles** of the dancers' arms, legs, torso, etc.:

1. Use MediaPipe Pose Estimation to find the positions of a dancer's joints
2. For the limbs we want to compare, find the corresponding joint positions
3. Compute **angle** using (x,y) coordinates of joints and trigonometry
4. Find difference between the limb's gradient (i.e. angle) of dancer1 and dancer2
5. Repeat and average for ALL limbs in one frame, then repeat for all frames

### 2. Audio Synchronization

When input videos have different lengths or start times:

1. Extract audio from both videos
2. Compute cross-correlation of the sound waves for different time lags
3. Find the time lag where cross-correlation is maximum (phase difference)
4. Trim the video that starts later to align both dancers

## Installation

### Prerequisites

- Python 3.11 (required for MediaPipe compatibility)
- macOS 12.5.1+ (recommended) or compatible system
- FFmpeg for video processing
- Praat for advanced audio analysis (optional)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Svkayy/moveify.git
   cd moveify
   ```

2. **Run the installation script**:
   ```bash
   ./install.sh
   ```

3. **Or install manually**:
   ```bash
   # Create virtual environment with Python 3.11
   python3.11 -m venv venv_mediapipe
   source venv_mediapipe/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### System Dependencies

**FFmpeg** (required for video processing):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Praat** (optional, for advanced audio analysis):
- Download from [Praat website](https://www.fon.hum.uva.nl/praat/download_mac.html)

## Usage

### Method 1: Command Line (Recommended)

1. **Add your videos to the video folder**:
   ```bash
   cp /path/to/your/dance.mov video/
   cp /path/to/model/dance.mov video/
   ```

2. **Run the analysis**:
   ```bash
   ./run_with_mediapipe.sh video/your_dance.mov video/model_dance.mov
   ```

### Method 2: Direct Python Command

```bash
source venv_mediapipe/bin/activate
python3 dance.py video/your_dance.mov video/model_dance.mov
```

### Method 3: GUI Interface

```bash
./launch_gui.sh
```

### Command Line Arguments

- `video1`: Path to first video (your dance)
- `video2`: Path to second video (model dance)
- `--output-dir`: Output directory for results (default: current directory)
- `--no-video`: Skip video generation (analysis only)

### Video Requirements

- Format: `.mov` recommended (`.mp4`, `.avi` also supported)
- Dimensions: Same resolution for both videos (e.g., 1280x720)
- Content: Clear view of the dancer's full body
- Lighting: Good lighting for pose detection
- Duration: Any length (will be automatically synchronized)

## Output Files

The analysis generates several output files:

1. **`comparison_output.mp4`**: Side-by-side comparison video with sync indicators
2. **`dance_analysis_report.json`**: Detailed analysis results in JSON format
3. **`sync_scores.png`**: Graph showing sync scores over time
4. **`crosscorrelation_results.txt`**: Audio synchronization results (if using Praat)

## Understanding the Results

### Sync Score

The sync score is a percentage (0-100%) indicating how well synchronized the dancers are:
- **90-100%**: Excellent synchronization
- **70-89%**: Good synchronization
- **50-69%**: Moderate synchronization
- **0-49%**: Poor synchronization

### Limb Analysis

The analysis breaks down performance by body part:
- **Left/Right Arm**: Shoulder-elbow-wrist angles
- **Left/Right Leg**: Hip-knee-ankle angles
- **Left/Right Torso**: Shoulder-hip alignment

### Frame-by-Frame Analysis

Each frame is analyzed individually, allowing you to see:
- Which parts of the dance are most challenging
- Where synchronization breaks down
- Overall improvement over time

## Example Output

```
DANCE SYNC ANALYSIS RESULTS
==================================================
Video 1: video/your_dance.mov
Video 2: video/model_dance.mov
Audio offset: 0.15 seconds
Frames analyzed: 450/500
Average sync score: 78.5%
Best sync score: 95.2%
Worst sync score: 45.1%

Limb Performance Analysis:
  Left Arm: 82.3% (avg diff: 31.8°)
  Right Arm: 75.1% (avg diff: 44.9°)
  Left Leg: 85.7% (avg diff: 25.7°)
  Right Leg: 79.2% (avg diff: 37.4°)
  Left Torso: 88.1% (avg diff: 21.5°)
  Right Torso: 81.6% (avg diff: 33.2°)

Comparison video: comparison_output.mp4
Detailed report: dance_analysis_report.json
Score visualization: sync_scores.png
```

## Project Structure

```
moveify/
├── dance.py                    # Main dance sync analysis script
├── dance_gui.py                # GUI interface for video selection
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── crosscorrelate.praat        # Praat script for audio analysis
├── test_dance_sync.py          # Test suite
├── example_usage.py            # Usage examples
├── setup.py                    # Package setup
├── install.sh                  # Installation script
├── launch_gui.sh               # GUI launcher
├── run_with_mediapipe.sh       # Command line runner
├── create_demo_videos.py       # Demo video generator
├── .gitignore                  # Git ignore rules
└── video/                      # Directory for dance videos
```

## Troubleshooting

### Common Issues

1. **"Could not extract audio from videos"**
   - Ensure videos contain audio tracks
   - Check video format compatibility

2. **"Could not extract pose landmarks"**
   - Ensure dancer is fully visible in frame
   - Check lighting conditions
   - Try different video angles

3. **Poor sync scores**
   - Ensure both videos show the same dance routine
   - Check that videos are properly synchronized
   - Verify video quality and lighting

4. **Memory issues with long videos**
   - Use shorter video segments
   - Reduce video resolution
   - Process videos in chunks

### Performance Tips

- Use videos with good lighting and clear view of the dancer
- Ensure the dancer is fully visible in the frame
- Use videos of similar length and quality
- Close other applications to free up memory

## Technical Details

### Pose Estimation

- Uses MediaPipe Pose solution
- Detects 33 pose landmarks
- Tracks 6 key limb angles for comparison
- Handles partial occlusion and multiple people

### Audio Synchronization

- Extracts audio using librosa
- Performs cross-correlation analysis
- Finds optimal time alignment
- Handles different audio formats and sampling rates

### Angle Calculation

- Uses dot product formula for angle calculation
- Normalizes angles to 0-180 degree range
- Handles missing or low-confidence landmarks
- Provides robust comparison metrics

## Contributing

This project combines dance analysis technology with real-time coaching capabilities. Contributions are welcome!

## Acknowledgments

- **Original Inspiration**: [dance-sync-analysis](https://github.com/Mruchus/dance-sync-analysis) by Mruchus
- **MediaPipe**: Google's pose estimation solution
- **Cross-correlation Tutorial**: Dr Spyros Kousidis' video syncing tutorial
- **Dance Analysis Concept**: Inspired by [this YouTube video](https://www.youtube.com/watch?v=zxhXj72WClE)

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the generated analysis reports
3. Ensure all dependencies are properly installed
4. Verify video format and quality requirements

Happy dancing! 🕺💃