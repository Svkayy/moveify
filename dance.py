#!/usr/bin/env python3
"""
Dance Sync Analysis - Compare your dance moves to a model dancer
Originally called 'Pose', this program analyzes movement and compares it to a reference video.
"""

import cv2
import mediapipe as mp
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import json

class DanceSyncAnalyzer:
    def __init__(self):
        """Initialize the dance sync analyzer with MediaPipe pose estimation."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define limb connections for angle calculation
        self.limb_connections = [
            # Arms
            (11, 13, 15),  # Left arm: shoulder, elbow, wrist
            (12, 14, 16),  # Right arm: shoulder, elbow, wrist
            # Legs
            (23, 25, 27),  # Left leg: hip, knee, ankle
            (24, 26, 28),  # Right leg: hip, knee, ankle
            # Torso
            (11, 12, 24),  # Left shoulder, right shoulder, right hip
            (12, 11, 23),  # Right shoulder, left shoulder, left hip
        ]
        
        self.limb_names = [
            "Left Arm", "Right Arm", "Left Leg", "Right Leg", 
            "Left Torso", "Right Torso"
        ]

    def extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video file."""
        try:
            # Use librosa to load audio
            audio, sr = librosa.load(video_path, sr=None)
            return audio, sr
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return None, None

    def find_audio_offset(self, audio1: np.ndarray, audio2: np.ndarray, sr: int) -> int:
        """Find the time offset between two audio signals using cross-correlation."""
        # Ensure both audio signals are the same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Compute cross-correlation
        correlation = signal.correlate(audio1, audio2, mode='full')
        
        # Find the peak (maximum correlation)
        peak_index = np.argmax(correlation)
        
        # Convert to time offset
        offset_samples = peak_index - (len(audio2) - 1)
        offset_seconds = offset_samples / sr
        
        return int(offset_seconds * sr)  # Return offset in samples

    def extract_pose_landmarks(self, video_path: str) -> List[Dict]:
        """Extract pose landmarks from video using MediaPipe."""
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                landmarks_list.append(landmarks)
            else:
                # If no pose detected, add None
                landmarks_list.append(None)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        print(f"Extracted landmarks from {len(landmarks_list)} frames")
        return landmarks_list

    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points using dot product."""
        if not all([p1, p2, p3]):
            return None
            
        # Convert to numpy arrays
        a = np.array([p1['x'], p1['y']])
        b = np.array([p2['x'], p2['y']])
        c = np.array([p3['x'], p3['y']])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def calculate_limb_angles(self, landmarks: List[Dict]) -> List[float]:
        """Calculate angles for all defined limb connections."""
        if not landmarks:
            return [None] * len(self.limb_connections)
        
        angles = []
        for connection in self.limb_connections:
            p1_idx, p2_idx, p3_idx = connection
            if (p1_idx < len(landmarks) and p2_idx < len(landmarks) and 
                p3_idx < len(landmarks)):
                angle = self.calculate_angle(
                    landmarks[p1_idx], landmarks[p2_idx], landmarks[p3_idx]
                )
                angles.append(angle)
            else:
                angles.append(None)
        
        return angles

    def calculate_sync_score(self, angles1: List[float], angles2: List[float]) -> float:
        """Calculate synchronization score between two sets of angles."""
        if not angles1 or not angles2:
            return 0.0
        
        valid_angles = 0
        total_diff = 0.0
        
        for a1, a2 in zip(angles1, angles2):
            if a1 is not None and a2 is not None:
                # Calculate absolute difference
                diff = abs(a1 - a2)
                # Normalize by 180 degrees (maximum possible difference)
                normalized_diff = min(diff, 180 - diff) / 180.0
                total_diff += normalized_diff
                valid_angles += 1
        
        if valid_angles == 0:
            return 0.0
        
        # Return score as percentage (100 - average difference)
        avg_diff = total_diff / valid_angles
        return (1.0 - avg_diff) * 100

    def create_comparison_video(self, video1_path: str, video2_path: str, 
                              landmarks1: List, landmarks2: List, 
                              output_path: str = "comparison_output.mp4") -> str:
        """Create side-by-side comparison video with sync indicators."""
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both videos")
            return None
        
        # Get video properties
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        frame_idx = 0
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Draw pose landmarks on frames
            if frame_idx < len(landmarks1) and landmarks1[frame_idx]:
                frame1 = self.draw_pose_landmarks(frame1, landmarks1[frame_idx])
            
            if frame_idx < len(landmarks2) and landmarks2[frame_idx]:
                frame2 = self.draw_pose_landmarks(frame2, landmarks2[frame_idx])
            
            # Calculate sync score for this frame
            if (frame_idx < len(landmarks1) and frame_idx < len(landmarks2) and
                landmarks1[frame_idx] and landmarks2[frame_idx]):
                angles1 = self.calculate_limb_angles(landmarks1[frame_idx])
                angles2 = self.calculate_limb_angles(landmarks2[frame_idx])
                sync_score = self.calculate_sync_score(angles1, angles2)
                
                # Add sync indicator
                color = (0, 255, 0) if sync_score > 70 else (0, 0, 255)
                cv2.putText(frame1, f"Sync: {sync_score:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame2, f"Sync: {sync_score:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Combine frames side by side
            combined_frame = np.hstack((frame1, frame2))
            out.write(combined_frame)
            
            frame_idx += 1
        
        cap1.release()
        cap2.release()
        out.release()
        
        print(f"Comparison video saved as: {output_path}")
        return output_path

    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """Draw pose landmarks on frame."""
        h, w, _ = frame.shape
        
        for landmark in landmarks:
            if landmark['visibility'] > 0.5:  # Only draw visible landmarks
                x = int(landmark['x'] * w)
                y = int(landmark['y'] * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        for connection in self.limb_connections:
            p1_idx, p2_idx, p3_idx = connection
            if (p1_idx < len(landmarks) and p2_idx < len(landmarks) and 
                p3_idx < len(landmarks) and
                landmarks[p1_idx]['visibility'] > 0.5 and
                landmarks[p2_idx]['visibility'] > 0.5 and
                landmarks[p3_idx]['visibility'] > 0.5):
                
                p1 = (int(landmarks[p1_idx]['x'] * w), int(landmarks[p1_idx]['y'] * h))
                p2 = (int(landmarks[p2_idx]['x'] * w), int(landmarks[p2_idx]['y'] * h))
                p3 = (int(landmarks[p3_idx]['x'] * w), int(landmarks[p3_idx]['y'] * h))
                
                cv2.line(frame, p1, p2, (255, 0, 0), 2)
                cv2.line(frame, p2, p3, (255, 0, 0), 2)
        
        return frame

    def analyze_dance_sync(self, video1_path: str, video2_path: str) -> Dict:
        """Main function to analyze dance synchronization between two videos."""
        print("Starting dance sync analysis...")
        
        # Step 1: Extract audio and find offset
        print("Step 1: Extracting audio and finding synchronization offset...")
        audio1, sr1 = self.extract_audio(video1_path)
        audio2, sr2 = self.extract_audio(video2_path)
        
        if audio1 is None or audio2 is None:
            print("Error: Could not extract audio from videos")
            return None
        
        offset = self.find_audio_offset(audio1, audio2, sr1)
        print(f"Audio offset: {offset} samples ({offset/sr1:.2f} seconds)")
        
        # Step 2: Extract pose landmarks
        print("Step 2: Extracting pose landmarks from videos...")
        landmarks1 = self.extract_pose_landmarks(video1_path)
        landmarks2 = self.extract_pose_landmarks(video2_path)
        
        if not landmarks1 or not landmarks2:
            print("Error: Could not extract pose landmarks")
            return None
        
        # Step 3: Calculate synchronization scores
        print("Step 3: Calculating synchronization scores...")
        sync_scores = []
        frame_angles1 = []
        frame_angles2 = []
        
        min_frames = min(len(landmarks1), len(landmarks2))
        
        for i in range(min_frames):
            if landmarks1[i] and landmarks2[i]:
                angles1 = self.calculate_limb_angles(landmarks1[i])
                angles2 = self.calculate_limb_angles(landmarks2[i])
                
                frame_angles1.append(angles1)
                frame_angles2.append(angles2)
                
                sync_score = self.calculate_sync_score(angles1, angles2)
                sync_scores.append(sync_score)
            else:
                sync_scores.append(0.0)
                frame_angles1.append([None] * len(self.limb_connections))
                frame_angles2.append([None] * len(self.limb_connections))
        
        # Calculate overall statistics
        valid_scores = [s for s in sync_scores if s > 0]
        avg_sync_score = np.mean(valid_scores) if valid_scores else 0.0
        max_sync_score = np.max(valid_scores) if valid_scores else 0.0
        min_sync_score = np.min(valid_scores) if valid_scores else 0.0
        
        print(f"Average sync score: {avg_sync_score:.2f}%")
        print(f"Max sync score: {max_sync_score:.2f}%")
        print(f"Min sync score: {min_sync_score:.2f}%")
        
        # Step 4: Create comparison video
        print("Step 4: Creating comparison video...")
        comparison_video = self.create_comparison_video(
            video1_path, video2_path, landmarks1, landmarks2
        )
        
        # Step 5: Create detailed analysis report
        analysis_results = {
            'video1_path': video1_path,
            'video2_path': video2_path,
            'audio_offset_seconds': offset / sr1,
            'total_frames': min_frames,
            'valid_frames': len(valid_scores),
            'average_sync_score': avg_sync_score,
            'max_sync_score': max_sync_score,
            'min_sync_score': min_sync_score,
            'frame_scores': sync_scores,
            'comparison_video': comparison_video,
            'limb_analysis': self.analyze_limb_performance(frame_angles1, frame_angles2)
        }
        
        return analysis_results

    def analyze_limb_performance(self, angles1: List, angles2: List) -> Dict:
        """Analyze performance for each limb separately."""
        limb_scores = {}
        
        for i, limb_name in enumerate(self.limb_names):
            limb_angles1 = [frame[i] for frame in angles1 if frame[i] is not None]
            limb_angles2 = [frame[i] for frame in angles2 if frame[i] is not None]
            
            if limb_angles1 and limb_angles2:
                min_len = min(len(limb_angles1), len(limb_angles2))
                limb_angles1 = limb_angles1[:min_len]
                limb_angles2 = limb_angles2[:min_len]
                
                differences = [abs(a1 - a2) for a1, a2 in zip(limb_angles1, limb_angles2)]
                avg_diff = np.mean(differences)
                limb_score = max(0, 100 - (avg_diff / 180 * 100))
                
                limb_scores[limb_name] = {
                    'average_score': limb_score,
                    'average_difference': avg_diff,
                    'frames_analyzed': min_len
                }
            else:
                limb_scores[limb_name] = {
                    'average_score': 0,
                    'average_difference': 180,
                    'frames_analyzed': 0
                }
        
        return limb_scores

    def save_analysis_report(self, results: Dict, output_file: str = "dance_analysis_report.json"):
        """Save detailed analysis results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Analysis report saved as: {output_file}")

    def create_score_visualization(self, results: Dict, output_file: str = "sync_scores.png"):
        """Create visualization of sync scores over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(results['frame_scores'])
        plt.title('Dance Synchronization Scores Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Sync Score (%)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        # Add average line
        avg_score = results['average_sync_score']
        plt.axhline(y=avg_score, color='r', linestyle='--', 
                   label=f'Average: {avg_score:.1f}%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Score visualization saved as: {output_file}")


def main():
    """Main function to run the dance sync analysis."""
    parser = argparse.ArgumentParser(description='Dance Sync Analysis Tool')
    parser.add_argument('video1', help='Path to first video (your dance)')
    parser.add_argument('video2', help='Path to second video (model dance)')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--no-video', action='store_true', help='Skip video generation')
    
    args = parser.parse_args()
    
    # Check if video files exist
    if not os.path.exists(args.video1):
        print(f"Error: Video file {args.video1} not found")
        return
    
    if not os.path.exists(args.video2):
        print(f"Error: Video file {args.video2} not found")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DanceSyncAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_dance_sync(args.video1, args.video2)
    
    if results:
        # Save results
        report_file = os.path.join(args.output_dir, "dance_analysis_report.json")
        analyzer.save_analysis_report(results, report_file)
        
        # Create visualization
        viz_file = os.path.join(args.output_dir, "sync_scores.png")
        analyzer.create_score_visualization(results, viz_file)
        
        # Print summary
        print("\n" + "="*50)
        print("DANCE SYNC ANALYSIS RESULTS")
        print("="*50)
        print(f"Video 1: {results['video1_path']}")
        print(f"Video 2: {results['video2_path']}")
        print(f"Audio offset: {results['audio_offset_seconds']:.2f} seconds")
        print(f"Frames analyzed: {results['valid_frames']}/{results['total_frames']}")
        print(f"Average sync score: {results['average_sync_score']:.2f}%")
        print(f"Best sync score: {results['max_sync_score']:.2f}%")
        print(f"Worst sync score: {results['min_sync_score']:.2f}%")
        
        print("\nLimb Performance Analysis:")
        for limb, data in results['limb_analysis'].items():
            print(f"  {limb}: {data['average_score']:.1f}% "
                  f"(avg diff: {data['average_difference']:.1f}°)")
        
        if results['comparison_video']:
            print(f"\nComparison video: {results['comparison_video']}")
        
        print(f"\nDetailed report: {report_file}")
        print(f"Score visualization: {viz_file}")
    else:
        print("Analysis failed. Please check your video files and try again.")


if __name__ == "__main__":
    main()
