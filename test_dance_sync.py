#!/usr/bin/env python3
"""
Test script for Dance Sync Analysis
This script provides a simple way to test the dance sync functionality
"""

import os
import sys
from dance import DanceSyncAnalyzer

def test_pose_estimation():
    """Test pose estimation functionality."""
    print("Testing pose estimation...")
    analyzer = DanceSyncAnalyzer()
    
    # Test with a sample video if available
    test_video = "video/test_dance.mov"
    if os.path.exists(test_video):
        print(f"Testing with {test_video}")
        landmarks = analyzer.extract_pose_landmarks(test_video)
        if landmarks:
            print(f"✓ Successfully extracted {len(landmarks)} frames of landmarks")
            return True
        else:
            print("✗ Failed to extract landmarks")
            return False
    else:
        print(f"Test video {test_video} not found. Please add a test video to test pose estimation.")
        return False

def test_audio_sync():
    """Test audio synchronization functionality."""
    print("Testing audio synchronization...")
    analyzer = DanceSyncAnalyzer()
    
    # Test with sample audio files if available
    audio1 = "video/test_audio1.wav"
    audio2 = "video/test_audio2.wav"
    
    if os.path.exists(audio1) and os.path.exists(audio2):
        print(f"Testing with {audio1} and {audio2}")
        audio1_data, sr1 = analyzer.extract_audio(audio1)
        audio2_data, sr2 = analyzer.extract_audio(audio2)
        
        if audio1_data is not None and audio2_data is not None:
            offset = analyzer.find_audio_offset(audio1_data, audio2_data, sr1)
            print(f"✓ Audio offset calculated: {offset} samples")
            return True
        else:
            print("✗ Failed to extract audio")
            return False
    else:
        print(f"Test audio files not found. Please add test audio files to test audio sync.")
        return False

def test_angle_calculation():
    """Test angle calculation functionality."""
    print("Testing angle calculation...")
    analyzer = DanceSyncAnalyzer()
    
    # Create sample landmark data
    sample_landmarks = [
        {'x': 0.5, 'y': 0.3, 'z': 0.0, 'visibility': 0.9},  # shoulder
        {'x': 0.6, 'y': 0.5, 'z': 0.0, 'visibility': 0.9},  # elbow
        {'x': 0.7, 'y': 0.7, 'z': 0.0, 'visibility': 0.9},  # wrist
    ]
    
    # Test angle calculation
    angle = analyzer.calculate_angle(sample_landmarks[0], sample_landmarks[1], sample_landmarks[2])
    if angle is not None:
        print(f"✓ Angle calculation successful: {angle:.2f} degrees")
        return True
    else:
        print("✗ Angle calculation failed")
        return False

def test_sync_score():
    """Test synchronization score calculation."""
    print("Testing sync score calculation...")
    analyzer = DanceSyncAnalyzer()
    
    # Create sample angle data
    angles1 = [90.0, 45.0, 120.0, 60.0, 30.0, 75.0]
    angles2 = [85.0, 50.0, 125.0, 55.0, 35.0, 80.0]
    
    score = analyzer.calculate_sync_score(angles1, angles2)
    if score > 0:
        print(f"✓ Sync score calculation successful: {score:.2f}%")
        return True
    else:
        print("✗ Sync score calculation failed")
        return False

def run_all_tests():
    """Run all available tests."""
    print("=" * 50)
    print("DANCE SYNC ANALYSIS - TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Angle Calculation", test_angle_calculation),
        ("Sync Score Calculation", test_sync_score),
        ("Pose Estimation", test_pose_estimation),
        ("Audio Synchronization", test_audio_sync),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Dance Sync is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

def create_sample_video_info():
    """Create information about sample video requirements."""
    print("\n" + "=" * 50)
    print("SAMPLE VIDEO REQUIREMENTS")
    print("=" * 50)
    print("To test the full functionality, you need:")
    print("1. Two dance videos in .mov format")
    print("2. Videos should be the same resolution (e.g., 1280x720)")
    print("3. Clear view of the dancer's full body")
    print("4. Good lighting for pose detection")
    print("5. Place videos in the 'video/' directory")
    print("\nExample usage:")
    print("python dance.py video/your_dance.mov video/model_dance.mov")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Dance Sync Analysis Test Suite")
        print("Usage: python test_dance_sync.py [--help]")
        print("\nThis script tests the core functionality of the dance sync analyzer.")
        print("For full testing, add sample videos to the 'video/' directory.")
        sys.exit(0)
    
    success = run_all_tests()
    create_sample_video_info()
    
    if not success:
        sys.exit(1)
