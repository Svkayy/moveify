#!/usr/bin/env python3
"""
Example usage of Dance Sync Analysis
This script demonstrates how to use the DanceSyncAnalyzer class programmatically
"""

import os
import sys
from dance import DanceSyncAnalyzer

def example_basic_usage():
    """Basic example of using the dance sync analyzer."""
    print("=== Basic Usage Example ===")
    
    # Initialize the analyzer
    analyzer = DanceSyncAnalyzer()
    
    # Example video paths (replace with your actual video paths)
    video1_path = "video/your_dance.mov"
    video2_path = "video/model_dance.mov"
    
    # Check if videos exist
    if not os.path.exists(video1_path):
        print(f"Video 1 not found: {video1_path}")
        print("Please add your dance video to the video/ directory")
        return False
    
    if not os.path.exists(video2_path):
        print(f"Video 2 not found: {video2_path}")
        print("Please add the model dance video to the video/ directory")
        return False
    
    # Run the analysis
    print("Running dance sync analysis...")
    results = analyzer.analyze_dance_sync(video1_path, video2_path)
    
    if results:
        print("Analysis completed successfully!")
        print(f"Average sync score: {results['average_sync_score']:.2f}%")
        return True
    else:
        print("Analysis failed!")
        return False

def example_step_by_step():
    """Step-by-step example showing individual components."""
    print("\n=== Step-by-Step Example ===")
    
    analyzer = DanceSyncAnalyzer()
    
    # Example video path
    video_path = "video/your_dance.mov"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return False
    
    # Step 1: Extract audio
    print("Step 1: Extracting audio...")
    audio, sr = analyzer.extract_audio(video_path)
    if audio is not None:
        print(f"✓ Audio extracted: {len(audio)} samples at {sr} Hz")
    else:
        print("✗ Audio extraction failed")
        return False
    
    # Step 2: Extract pose landmarks
    print("Step 2: Extracting pose landmarks...")
    landmarks = analyzer.extract_pose_landmarks(video_path)
    if landmarks:
        print(f"✓ Landmarks extracted: {len(landmarks)} frames")
    else:
        print("✗ Landmark extraction failed")
        return False
    
    # Step 3: Calculate angles for first frame
    print("Step 3: Calculating angles...")
    if landmarks[0]:
        angles = analyzer.calculate_limb_angles(landmarks[0])
        valid_angles = [a for a in angles if a is not None]
        print(f"✓ Angles calculated: {len(valid_angles)}/{len(angles)} valid angles")
        for i, angle in enumerate(valid_angles):
            if angle is not None:
                print(f"  {analyzer.limb_names[i]}: {angle:.1f}°")
    else:
        print("✗ No landmarks in first frame")
        return False
    
    return True

def example_custom_analysis():
    """Example of custom analysis with specific parameters."""
    print("\n=== Custom Analysis Example ===")
    
    analyzer = DanceSyncAnalyzer()
    
    # Example: Analyze only specific limbs
    custom_connections = [
        (11, 13, 15),  # Left arm only
        (12, 14, 16),  # Right arm only
    ]
    
    # Temporarily modify the analyzer's limb connections
    original_connections = analyzer.limb_connections
    analyzer.limb_connections = custom_connections
    
    print("Analyzing only arm movements...")
    
    # Restore original connections
    analyzer.limb_connections = original_connections
    
    return True

def main():
    """Main function to run examples."""
    print("Dance Sync Analysis - Example Usage")
    print("=" * 50)
    
    # Check if video directory exists
    if not os.path.exists("video"):
        print("Creating video directory...")
        os.makedirs("video")
    
    # Run examples
    examples = [
        ("Step-by-Step Analysis", example_step_by_step),
        ("Custom Analysis", example_custom_analysis),
        ("Basic Usage", example_basic_usage),
    ]
    
    for name, example_func in examples:
        print(f"\n--- {name} ---")
        try:
            success = example_func()
            if success:
                print(f"✓ {name} completed successfully")
            else:
                print(f"✗ {name} failed")
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print("Example usage completed!")
    print("\nTo run the full analysis:")
    print("python dance.py video/your_dance.mov video/model_dance.mov")

if __name__ == "__main__":
    main()
