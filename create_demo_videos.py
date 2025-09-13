#!/usr/bin/env python3
"""
Create demo videos for testing the dance sync analysis
This script creates simple test videos with basic movements
"""

import cv2
import numpy as np
import os

def create_demo_video(filename, width=640, height=480, duration=5, fps=30):
    """Create a simple demo video with basic movements."""
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate time-based movement
        t = frame_num / fps
        
        # Create a simple "dancer" (colored circles representing joints)
        center_x = width // 2
        center_y = height // 2
        
        # Head (bobbing up and down)
        head_y = center_y - 100 + int(20 * np.sin(t * 2))
        cv2.circle(frame, (center_x, head_y), 15, (255, 255, 255), -1)
        
        # Shoulders (slight sway)
        shoulder_offset = int(10 * np.sin(t * 1.5))
        left_shoulder = (center_x - 40 + shoulder_offset, center_y - 50)
        right_shoulder = (center_x + 40 + shoulder_offset, center_y - 50)
        cv2.circle(frame, left_shoulder, 10, (0, 255, 0), -1)
        cv2.circle(frame, right_shoulder, 10, (0, 255, 0), -1)
        
        # Arms (swinging)
        arm_angle = t * 3  # Rotating arms
        left_elbow_x = center_x - 40 + int(30 * np.cos(arm_angle))
        left_elbow_y = center_y - 20 + int(30 * np.sin(arm_angle))
        right_elbow_x = center_x + 40 + int(30 * np.cos(arm_angle + np.pi))
        right_elbow_y = center_y - 20 + int(30 * np.sin(arm_angle + np.pi))
        
        cv2.circle(frame, (left_elbow_x, left_elbow_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (right_elbow_x, right_elbow_y), 8, (255, 0, 0), -1)
        
        # Wrists
        left_wrist_x = left_elbow_x + int(25 * np.cos(arm_angle + 0.5))
        left_wrist_y = left_elbow_y + int(25 * np.sin(arm_angle + 0.5))
        right_wrist_x = right_elbow_x + int(25 * np.cos(arm_angle + np.pi + 0.5))
        right_wrist_y = right_elbow_y + int(25 * np.sin(arm_angle + np.pi + 0.5))
        
        cv2.circle(frame, (left_wrist_x, left_wrist_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (right_wrist_x, right_wrist_y), 6, (0, 0, 255), -1)
        
        # Torso
        cv2.circle(frame, (center_x, center_y), 20, (128, 128, 128), -1)
        
        # Hips (swaying)
        hip_offset = int(15 * np.sin(t * 1.2))
        left_hip = (center_x - 30 + hip_offset, center_y + 50)
        right_hip = (center_x + 30 + hip_offset, center_y + 50)
        cv2.circle(frame, left_hip, 12, (255, 255, 0), -1)
        cv2.circle(frame, right_hip, 12, (255, 255, 0), -1)
        
        # Legs (stepping)
        leg_angle = t * 2
        left_knee_x = center_x - 30 + int(20 * np.cos(leg_angle))
        left_knee_y = center_y + 80 + int(20 * np.sin(leg_angle))
        right_knee_x = center_x + 30 + int(20 * np.cos(leg_angle + np.pi))
        right_knee_y = center_y + 80 + int(20 * np.sin(leg_angle + np.pi))
        
        cv2.circle(frame, (left_knee_x, left_knee_y), 10, (255, 0, 255), -1)
        cv2.circle(frame, (right_knee_x, right_knee_y), 10, (255, 0, 255), -1)
        
        # Ankles
        left_ankle_x = left_knee_x + int(15 * np.cos(leg_angle + 0.3))
        left_ankle_y = left_knee_y + int(15 * np.sin(leg_angle + 0.3))
        right_ankle_x = right_knee_x + int(15 * np.cos(leg_angle + np.pi + 0.3))
        right_ankle_y = right_knee_y + int(15 * np.sin(leg_angle + np.pi + 0.3))
        
        cv2.circle(frame, (left_ankle_x, left_ankle_y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (right_ankle_x, right_ankle_y), 8, (0, 255, 255), -1)
        
        # Draw connections between joints
        # Head to shoulders
        cv2.line(frame, (center_x, head_y), left_shoulder, (255, 255, 255), 2)
        cv2.line(frame, (center_x, head_y), right_shoulder, (255, 255, 255), 2)
        
        # Arms
        cv2.line(frame, left_shoulder, (left_elbow_x, left_elbow_y), (0, 255, 0), 2)
        cv2.line(frame, (left_elbow_x, left_elbow_y), (left_wrist_x, left_wrist_y), (255, 0, 0), 2)
        cv2.line(frame, right_shoulder, (right_elbow_x, right_elbow_y), (0, 255, 0), 2)
        cv2.line(frame, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), (255, 0, 0), 2)
        
        # Torso
        cv2.line(frame, left_shoulder, (center_x, center_y), (128, 128, 128), 2)
        cv2.line(frame, right_shoulder, (center_x, center_y), (128, 128, 128), 2)
        cv2.line(frame, (center_x, center_y), left_hip, (128, 128, 128), 2)
        cv2.line(frame, (center_x, center_y), right_hip, (128, 128, 128), 2)
        
        # Legs
        cv2.line(frame, left_hip, (left_knee_x, left_knee_y), (255, 255, 0), 2)
        cv2.line(frame, (left_knee_x, left_knee_y), (left_ankle_x, left_ankle_y), (255, 0, 255), 2)
        cv2.line(frame, right_hip, (right_knee_x, right_knee_y), (255, 255, 0), 2)
        cv2.line(frame, (right_knee_x, right_knee_y), (right_ankle_x, right_ankle_y), (255, 0, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created demo video: {filename}")

def create_variation_video(filename, width=640, height=480, duration=5, fps=30, phase_offset=0.5):
    """Create a variation of the demo video with slightly different timing."""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        t = frame_num / fps + phase_offset  # Add phase offset
        
        # Same drawing code as create_demo_video but with t+phase_offset
        center_x = width // 2
        center_y = height // 2
        
        # Head
        head_y = center_y - 100 + int(20 * np.sin(t * 2))
        cv2.circle(frame, (center_x, head_y), 15, (255, 255, 255), -1)
        
        # Shoulders
        shoulder_offset = int(10 * np.sin(t * 1.5))
        left_shoulder = (center_x - 40 + shoulder_offset, center_y - 50)
        right_shoulder = (center_x + 40 + shoulder_offset, center_y - 50)
        cv2.circle(frame, left_shoulder, 10, (0, 255, 0), -1)
        cv2.circle(frame, right_shoulder, 10, (0, 255, 0), -1)
        
        # Arms (with slight variation)
        arm_angle = t * 3.1  # Slightly different speed
        left_elbow_x = center_x - 40 + int(30 * np.cos(arm_angle))
        left_elbow_y = center_y - 20 + int(30 * np.sin(arm_angle))
        right_elbow_x = center_x + 40 + int(30 * np.cos(arm_angle + np.pi))
        right_elbow_y = center_y - 20 + int(30 * np.sin(arm_angle + np.pi))
        
        cv2.circle(frame, (left_elbow_x, left_elbow_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (right_elbow_x, right_elbow_y), 8, (255, 0, 0), -1)
        
        # Wrists
        left_wrist_x = left_elbow_x + int(25 * np.cos(arm_angle + 0.5))
        left_wrist_y = left_elbow_y + int(25 * np.sin(arm_angle + 0.5))
        right_wrist_x = right_elbow_x + int(25 * np.cos(arm_angle + np.pi + 0.5))
        right_wrist_y = right_elbow_y + int(25 * np.sin(arm_angle + np.pi + 0.5))
        
        cv2.circle(frame, (left_wrist_x, left_wrist_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (right_wrist_x, right_wrist_y), 6, (0, 0, 255), -1)
        
        # Torso
        cv2.circle(frame, (center_x, center_y), 20, (128, 128, 128), -1)
        
        # Hips
        hip_offset = int(15 * np.sin(t * 1.2))
        left_hip = (center_x - 30 + hip_offset, center_y + 50)
        right_hip = (center_x + 30 + hip_offset, center_y + 50)
        cv2.circle(frame, left_hip, 12, (255, 255, 0), -1)
        cv2.circle(frame, right_hip, 12, (255, 255, 0), -1)
        
        # Legs
        leg_angle = t * 2.1  # Slightly different speed
        left_knee_x = center_x - 30 + int(20 * np.cos(leg_angle))
        left_knee_y = center_y + 80 + int(20 * np.sin(leg_angle))
        right_knee_x = center_x + 30 + int(20 * np.cos(leg_angle + np.pi))
        right_knee_y = center_y + 80 + int(20 * np.sin(leg_angle + np.pi))
        
        cv2.circle(frame, (left_knee_x, left_knee_y), 10, (255, 0, 255), -1)
        cv2.circle(frame, (right_knee_x, right_knee_y), 10, (255, 0, 255), -1)
        
        # Ankles
        left_ankle_x = left_knee_x + int(15 * np.cos(leg_angle + 0.3))
        left_ankle_y = left_knee_y + int(15 * np.sin(leg_angle + 0.3))
        right_ankle_x = right_knee_x + int(15 * np.cos(leg_angle + np.pi + 0.3))
        right_ankle_y = right_knee_y + int(15 * np.sin(leg_angle + np.pi + 0.3))
        
        cv2.circle(frame, (left_ankle_x, left_ankle_y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (right_ankle_x, right_ankle_y), 8, (0, 255, 255), -1)
        
        # Draw connections
        cv2.line(frame, (center_x, head_y), left_shoulder, (255, 255, 255), 2)
        cv2.line(frame, (center_x, head_y), right_shoulder, (255, 255, 255), 2)
        cv2.line(frame, left_shoulder, (left_elbow_x, left_elbow_y), (0, 255, 0), 2)
        cv2.line(frame, (left_elbow_x, left_elbow_y), (left_wrist_x, left_wrist_y), (255, 0, 0), 2)
        cv2.line(frame, right_shoulder, (right_elbow_x, right_elbow_y), (0, 255, 0), 2)
        cv2.line(frame, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), (255, 0, 0), 2)
        cv2.line(frame, left_shoulder, (center_x, center_y), (128, 128, 128), 2)
        cv2.line(frame, right_shoulder, (center_x, center_y), (128, 128, 128), 2)
        cv2.line(frame, (center_x, center_y), left_hip, (128, 128, 128), 2)
        cv2.line(frame, (center_x, center_y), right_hip, (128, 128, 128), 2)
        cv2.line(frame, left_hip, (left_knee_x, left_knee_y), (255, 255, 0), 2)
        cv2.line(frame, (left_knee_x, left_knee_y), (left_ankle_x, left_ankle_y), (255, 0, 255), 2)
        cv2.line(frame, right_hip, (right_knee_x, right_knee_y), (255, 255, 0), 2)
        cv2.line(frame, (right_knee_x, right_knee_y), (right_ankle_x, right_ankle_y), (255, 0, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_num} (Variation)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created variation video: {filename}")

def main():
    """Create demo videos for testing."""
    print("Creating demo videos for dance sync analysis...")
    
    # Ensure video directory exists
    os.makedirs("video", exist_ok=True)
    
    # Create two demo videos
    video1_path = "video/demo_dance_1.mov"
    video2_path = "video/demo_dance_2.mov"
    
    print("Creating first demo video...")
    create_demo_video(video1_path)
    
    print("Creating second demo video (with variations)...")
    create_variation_video(video2_path)
    
    print("\nDemo videos created successfully!")
    print(f"Video 1: {video1_path}")
    print(f"Video 2: {video2_path}")
    print("\nNow you can run the dance sync analysis:")
    print("./run_with_mediapipe.sh video/demo_dance_1.mov video/demo_dance_2.mov")

if __name__ == "__main__":
    main()
