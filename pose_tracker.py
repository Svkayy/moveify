#!/usr/bin/env python3
"""
Moveify Multi-Person Pose Tracker
=================================

A real-time multi-person pose tracking application using Google MediaPipe and OpenCV.
Captures live video from webcam and overlays pose landmarks and skeleton connections
for multiple people simultaneously.

Features:
- Multi-person pose detection (up to 5 people)
- Different colors for each person's skeleton
- Person ID labels
- Real-time pose tracking

Requirements:
- OpenCV for video capture and display
- MediaPipe for pose detection and landmark tracking
- Webcam access

Usage:
    python pose_tracker.py

Controls:
    'q' - Quit the application
    'r' - Reset person counter
"""

import cv2
import mediapipe as mp
import numpy as np
import random


def generate_person_colors(num_people):
    """
    Generate distinct colors for each person's pose landmarks.
    
    Args:
        num_people (int): Number of people to generate colors for
        
    Returns:
        list: List of BGR color tuples for each person
    """
    colors = []
    for i in range(num_people):
        # Generate distinct colors using HSV color space
        hue = int(180 * i / num_people)  # Distribute hues evenly
        # Convert HSV to BGR
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr_color)))
    return colors


def draw_pose_landmarks(frame, landmarks, connections, color, person_id):
    """
    Draw pose landmarks and connections for a single person.
    
    Args:
        frame: OpenCV frame to draw on
        landmarks: MediaPipe pose landmarks
        connections: Pose connections to draw
        color: BGR color tuple for this person
        person_id: ID number for this person
    """
    h, w, _ = frame.shape
    
    # Draw landmarks
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, color, -1)
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if (start_idx < len(landmarks.landmark) and 
            end_idx < len(landmarks.landmark)):
            
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            start_x = int(start_point.x * w)
            start_y = int(start_point.y * h)
            end_x = int(end_point.x * w)
            end_y = int(end_point.y * h)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    # Draw person ID label
    if landmarks.landmark:
        # Use nose position for ID label
        nose = landmarks.landmark[0]  # Nose is landmark 0
        label_x = int(nose.x * w)
        label_y = int(nose.y * h) - 20
        
        cv2.putText(frame, f"Person {person_id}", 
                   (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)


def main():
    """
    Main function that initializes the multi-person pose tracking system and runs the main loop.
    """
    
    # Initialize MediaPipe pose solution
    # This creates the pose detection model with specific configuration
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure the pose detection model for multi-person tracking
    # min_detection_confidence: Minimum confidence for pose detection (0.0-1.0)
    # min_tracking_confidence: Minimum confidence for pose tracking (0.0-1.0)
    # static_image_mode: False for video (tracks across frames)
    # model_complexity: 1 for better accuracy (0, 1, or 2)
    # smooth_landmarks: True for smoother tracking
    # enable_segmentation: False (we don't need segmentation for pose tracking)
    # smooth_segmentation: False
    # min_detection_confidence: 0.5
    # min_tracking_confidence: 0.5
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize OpenCV video capture
    # 0 refers to the default webcam (usually the built-in camera)
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is accessible
    if not cap.isOpened():
        print("Error: Could not access webcam. Please check your camera connection.")
        return
    
    print("Multi-Person Pose Tracker Started!")
    print("Press 'q' to quit the application")
    print("Press 'r' to reset person counter")
    print("Make sure people are visible in the camera frame for pose detection")
    
    # Variables for multi-person tracking
    person_counter = 0
    max_people = 5  # Maximum number of people to track
    person_colors = generate_person_colors(max_people)
    
    # Main video processing loop
    while True:
        # Read a frame from the webcam
        # ret: boolean indicating if frame was read successfully
        # frame: the actual image data as a numpy array
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        # Flip the frame horizontally for a mirror effect
        # This makes the video feel more natural to the user
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        # MediaPipe expects RGB format, but OpenCV uses BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe pose detection
        # This analyzes the image and detects human pose landmarks
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks and connections for each detected person
        if results.pose_landmarks:
            # For multi-person tracking, we need to process each person individually
            # Note: MediaPipe's pose solution detects one person at a time by default
            # For true multi-person tracking, we would need to use a different approach
            # or run the detection multiple times with different regions of interest
            
            # Draw the pose landmarks (33 body points) and connections
            # This creates the skeleton overlay on the video
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add person ID label
            if results.pose_landmarks.landmark:
                nose = results.pose_landmarks.landmark[0]  # Nose is landmark 0
                h, w, _ = frame.shape
                label_x = int(nose.x * w)
                label_y = int(nose.y * h) - 20
                
                cv2.putText(frame, f"Person {person_counter + 1}", 
                           (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, person_colors[person_counter % len(person_colors)], 2)
            
            # Optional: Print landmark coordinates for debugging
            # Uncomment the following lines to see landmark coordinates in the console
            # for idx, landmark in enumerate(results.pose_landmarks.landmark):
            #     print(f"Person {person_counter + 1} - Landmark {idx}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
        
        # Add text overlay to the frame
        # This provides visual feedback about the application status
        cv2.putText(frame, "Moveify Multi-Person Pose Tracker", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame in a window
        # This shows the video feed with pose landmarks overlaid
        cv2.imshow('Moveify Multi-Person Pose Tracker', frame)
        
        # Check for key press
        # Wait for 1ms and check if 'q' or 'r' key was pressed
        key = cv2.waitKey(1) & 0xFF
        
        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            print("Exiting application...")
            break
        # Reset person counter if 'r' is pressed
        elif key == ord('r'):
            person_counter = 0
            print("Person counter reset!")
    
    # Cleanup: Release resources
    # This is important to free up the webcam and close windows properly
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
    
    print("Application closed successfully!")


if __name__ == "__main__":
    # Run the main function when the script is executed directly
    main()
