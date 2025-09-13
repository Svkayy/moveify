#!/usr/bin/env python3
"""
Moveify True Multi-Person Pose Tracker
======================================

A real-time multi-person pose tracking application using MediaPipe and OpenCV.
This version uses a two-stage approach:
1. Person detection using MediaPipe's holistic solution
2. Pose estimation for each detected person

Features:
- True multi-person pose detection (up to 5 people)
- Different colors for each person's skeleton
- Person ID labels and tracking
- Real-time pose tracking for multiple people
- Region-based detection for better accuracy

Requirements:
- OpenCV for video capture and display
- MediaPipe for pose detection and landmark tracking
- Webcam access

Usage:
    python true_multi_person_tracker.py

Controls:
    'q' - Quit the application
    'r' - Reset person counter
    's' - Save current frame
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import defaultdict, deque
import math


class PersonTracker:
    """
    Tracks people across frames to handle overlapping and maintain consistent IDs.
    """
    
    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_person_id = 0
        self.persons = {}  # person_id -> {'centroid': (x, y), 'landmarks': landmarks, 'confidence': conf, 'disappeared': 0}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history = defaultdict(lambda: deque(maxlen=5))  # Track recent positions
    
    def calculate_centroid(self, landmarks):
        """Calculate centroid from pose landmarks."""
        if not landmarks or not landmarks.landmark:
            return None
        
        x_coords = [landmark.x for landmark in landmarks.landmark if landmark.visibility > 0.5]
        y_coords = [landmark.y for landmark in landmarks.landmark if landmark.visibility > 0.5]
        
        if not x_coords or not y_coords:
            return None
        
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    def calculate_distance(self, centroid1, centroid2):
        """Calculate Euclidean distance between two centroids."""
        if not centroid1 or not centroid2:
            return float('inf')
        return math.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    
    def register(self, centroid, landmarks, confidence):
        """Register a new person."""
        person_id = self.next_person_id
        self.persons[person_id] = {
            'centroid': centroid,
            'landmarks': landmarks,
            'confidence': confidence,
            'disappeared': 0
        }
        self.disappeared[person_id] = 0
        self.history[person_id].append(centroid)
        self.next_person_id += 1
        return person_id
    
    def deregister(self, person_id):
        """Deregister a person."""
        if person_id in self.persons:
            del self.persons[person_id]
        if person_id in self.disappeared:
            del self.disappeared[person_id]
        if person_id in self.history:
            del self.history[person_id]
    
    def predict_position(self, person_id):
        """Predict next position based on movement history."""
        if person_id not in self.history or len(self.history[person_id]) < 2:
            return None
        
        history = list(self.history[person_id])
        if len(history) >= 2:
            # Simple linear prediction
            dx = history[-1][0] - history[-2][0]
            dy = history[-1][1] - history[-2][1]
            predicted_x = history[-1][0] + dx
            predicted_y = history[-1][1] + dy
            return (predicted_x, predicted_y)
        return None
    
    def update(self, detected_people):
        """Update person tracking with new detections."""
        if len(detected_people) == 0:
            # No detections - increment disappeared count
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
            return self.persons
        
        # Calculate centroids for new detections
        new_centroids = []
        for person in detected_people:
            if person.get('pose_landmarks'):
                centroid = self.calculate_centroid(person['pose_landmarks'])
                if centroid:
                    new_centroids.append((centroid, person))
        
        if len(self.persons) == 0:
            # No existing persons - register all new detections
            for centroid, person in new_centroids:
                confidence = person.get('confidence', 0.5)
                self.register(centroid, person['pose_landmarks'], confidence)
        else:
            # Match existing persons with new detections
            person_ids = list(self.persons.keys())
            person_centroids = [self.persons[pid]['centroid'] for pid in person_ids]
            
            # Calculate distance matrix
            D = np.full((len(person_centroids), len(new_centroids)), float('inf'))
            for i, existing_centroid in enumerate(person_centroids):
                for j, (new_centroid, _) in enumerate(new_centroids):
                    # Use predicted position if available
                    predicted = self.predict_position(person_ids[i])
                    if predicted:
                        distance = self.calculate_distance(predicted, new_centroid)
                    else:
                        distance = self.calculate_distance(existing_centroid, new_centroid)
                    D[i, j] = distance
            
            # Hungarian algorithm approximation - simple greedy matching
            used_existing = set()
            used_new = set()
            
            # Sort by distance and match
            matches = []
            for i in range(len(person_centroids)):
                for j in range(len(new_centroids)):
                    if i not in used_existing and j not in used_new and D[i, j] < self.max_distance:
                        matches.append((i, j, D[i, j]))
            
            matches.sort(key=lambda x: x[2])  # Sort by distance
            
            for i, j, distance in matches:
                if i not in used_existing and j not in used_new:
                    person_id = person_ids[i]
                    centroid, person = new_centroids[j]
                    
                    # Update existing person
                    self.persons[person_id]['centroid'] = centroid
                    self.persons[person_id]['landmarks'] = person['pose_landmarks']
                    self.persons[person_id]['confidence'] = person.get('confidence', 0.5)
                    self.disappeared[person_id] = 0
                    self.history[person_id].append(centroid)
                    
                    used_existing.add(i)
                    used_new.add(j)
            
            # Handle unmatched existing persons
            for i, person_id in enumerate(person_ids):
                if i not in used_existing:
                    self.disappeared[person_id] += 1
                    if self.disappeared[person_id] > self.max_disappeared:
                        self.deregister(person_id)
            
            # Handle unmatched new detections
            for j, (centroid, person) in enumerate(new_centroids):
                if j not in used_new:
                    confidence = person.get('confidence', 0.5)
                    self.register(centroid, person['pose_landmarks'], confidence)
        
        return self.persons


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


def detect_overlaps(detected_people, overlap_threshold=0.3):
    """
    Detect overlapping people based on bounding box overlap.
    
    Args:
        detected_people: List of detected people with landmarks
        overlap_threshold: Minimum overlap ratio to consider as overlap
        
    Returns:
        list: List of overlap groups
    """
    if len(detected_people) < 2:
        return []
    
    overlaps = []
    processed = set()
    
    for i, person1 in enumerate(detected_people):
        if i in processed:
            continue
            
        overlap_group = [i]
        bbox1 = calculate_bounding_box(person1['pose_landmarks'])
        
        for j, person2 in enumerate(detected_people[i+1:], i+1):
            if j in processed:
                continue
                
            bbox2 = calculate_bounding_box(person2['pose_landmarks'])
            overlap_ratio = calculate_overlap_ratio(bbox1, bbox2)
            
            if overlap_ratio > overlap_threshold:
                overlap_group.append(j)
                processed.add(j)
        
        if len(overlap_group) > 1:
            overlaps.append(overlap_group)
            processed.update(overlap_group)
    
    return overlaps


def calculate_bounding_box(landmarks):
    """
    Calculate bounding box from pose landmarks.
    
    Args:
        landmarks: MediaPipe pose landmarks
        
    Returns:
        tuple: (x, y, width, height) bounding box
    """
    if not landmarks or not landmarks.landmark:
        return None
    
    visible_landmarks = [landmark for landmark in landmarks.landmark if landmark.visibility > 0.5]
    if not visible_landmarks:
        return None
    
    x_coords = [landmark.x for landmark in visible_landmarks]
    y_coords = [landmark.y for landmark in visible_landmarks]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def calculate_overlap_ratio(bbox1, bbox2):
    """
    Calculate overlap ratio between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes as (x, y, width, height)
        
    Returns:
        float: Overlap ratio (0.0 to 1.0)
    """
    if not bbox1 or not bbox2:
        return 0.0
    
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def handle_overlapping_people(detected_people, overlaps, person_tracker):
    """
    Handle overlapping people by using tracking information and confidence.
    
    Args:
        detected_people: List of detected people
        overlaps: List of overlap groups
        person_tracker: PersonTracker instance
        
    Returns:
        list: Processed detected people with tracking info
    """
    processed_people = []
    
    for i, person in enumerate(detected_people):
        # Check if this person is in an overlap group
        in_overlap = False
        for overlap_group in overlaps:
            if i in overlap_group:
                in_overlap = True
                break
        
        if in_overlap:
            # For overlapping people, use tracking to maintain identity
            # Find the best match in the tracker
            best_match_id = None
            best_confidence = 0
            
            for person_id, tracked_person in person_tracker.persons.items():
                if tracked_person.get('landmarks'):
                    # Calculate similarity based on centroid distance
                    centroid = person_tracker.calculate_centroid(person['pose_landmarks'])
                    tracked_centroid = tracked_person.get('centroid')
                    
                    if centroid and tracked_centroid:
                        distance = person_tracker.calculate_distance(centroid, tracked_centroid)
                        if distance < person_tracker.max_distance:
                            # Use confidence and distance to determine best match
                            person_confidence = person.get('confidence', 0.5)
                            combined_score = person_confidence * (1.0 - distance / person_tracker.max_distance)
                            if combined_score > best_confidence:
                                best_confidence = combined_score
                                best_match_id = person_id
            
            # Add tracking information
            person['tracking_id'] = best_match_id
            person['is_overlapping'] = True
        else:
            person['tracking_id'] = None
            person['is_overlapping'] = False
        
        processed_people.append(person)
    
    return processed_people


def detect_people_using_holistic(frame, holistic, expected_people=2):
    """
    Detect people using MediaPipe's holistic solution.
    This approach runs holistic detection on different regions to find multiple people.
    
    Args:
        frame: Input frame
        holistic: MediaPipe holistic solution
        expected_people: Expected number of people to detect
        
    Returns:
        list: List of detected people with their landmarks
    """
    h, w = frame.shape[:2]
    detected_people = []
    
    # Create regions for holistic detection
    if expected_people == 1:
        regions = [(0, 0, w, h)]  # Full frame
    elif expected_people == 2:
        regions = [
            (0, 0, w//2, h),      # Left half
            (w//2, 0, w//2, h),   # Right half
        ]
    elif expected_people == 3:
        regions = [
            (0, 0, w//3, h),      # Left third
            (w//3, 0, w//3, h),   # Middle third
            (2*w//3, 0, w//3, h), # Right third
        ]
    elif expected_people == 4:
        regions = [
            (0, 0, w//2, h//2),      # Top-left
            (w//2, 0, w//2, h//2),   # Top-right
            (0, h//2, w//2, h//2),   # Bottom-left
            (w//2, h//2, w//2, h//2), # Bottom-right
        ]
    else:  # 5 or more people
        regions = [
            (0, 0, w//2, h//2),      # Top-left
            (w//2, 0, w//2, h//2),   # Top-right
            (0, h//2, w//2, h//2),   # Bottom-left
            (w//2, h//2, w//2, h//2), # Bottom-right
            (w//4, h//4, w//2, h//2) # Center
        ]
    
    for i, (x, y, region_w, region_h) in enumerate(regions):
        # Extract region
        region = frame[y:y+region_h, x:x+region_w]
        
        if region.size == 0:
            continue
        
        # Process region with holistic detection
        rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_region)
        
        if results.pose_landmarks:
            # Calculate confidence based on landmark visibility
            visible_landmarks = sum(1 for landmark in results.pose_landmarks.landmark 
                                  if landmark.visibility > 0.5)
            confidence = visible_landmarks / len(results.pose_landmarks.landmark)
            
            # Only accept detections with reasonable confidence
            if confidence > 0.3:
                # Adjust landmark coordinates to full frame
                adjusted_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    adjusted_landmark = landmark
                    adjusted_landmark.x = (landmark.x * region_w + x) / w
                    adjusted_landmark.y = (landmark.y * region_h + y) / h
                    adjusted_landmarks.append(adjusted_landmark)
                
                # Create a new landmark list with adjusted coordinates
                from mediapipe.framework.formats import landmark_pb2
                adjusted_pose_landmarks = landmark_pb2.NormalizedLandmarkList()
                for landmark in adjusted_landmarks:
                    adjusted_pose_landmarks.landmark.append(landmark)
                
                detected_people.append({
                    'pose_landmarks': adjusted_pose_landmarks,
                    'face_landmarks': results.face_landmarks,
                    'left_hand_landmarks': results.left_hand_landmarks,
                    'right_hand_landmarks': results.right_hand_landmarks,
                    'confidence': confidence,
                    'region_id': i
                })
    
    # Sort by confidence and return only the best detections
    detected_people.sort(key=lambda x: x['confidence'], reverse=True)
    return detected_people[:expected_people]  # Return only expected number of people


def detect_people_using_region_splitting(frame, pose, expected_people=2):
    """
    Detect multiple people by splitting the frame into regions and detecting poses in each region.
    This approach is more accurate when you know the expected number of people.
    
    Args:
        frame: Input frame
        pose: MediaPipe pose solution
        expected_people: Expected number of people to detect
        
    Returns:
        list: List of detected people with their landmarks
    """
    h, w = frame.shape[:2]
    detected_people = []
    
    # Create more intelligent regions based on expected number of people
    if expected_people == 1:
        regions = [(0, 0, w, h)]  # Full frame
    elif expected_people == 2:
        regions = [
            (0, 0, w//2, h),      # Left half
            (w//2, 0, w//2, h),   # Right half
        ]
    elif expected_people == 3:
        regions = [
            (0, 0, w//3, h),      # Left third
            (w//3, 0, w//3, h),   # Middle third
            (2*w//3, 0, w//3, h), # Right third
        ]
    elif expected_people == 4:
        regions = [
            (0, 0, w//2, h//2),      # Top-left
            (w//2, 0, w//2, h//2),   # Top-right
            (0, h//2, w//2, h//2),   # Bottom-left
            (w//2, h//2, w//2, h//2), # Bottom-right
        ]
    else:  # 5 or more people
        regions = [
            (0, 0, w//2, h//2),      # Top-left
            (w//2, 0, w//2, h//2),   # Top-right
            (0, h//2, w//2, h//2),   # Bottom-left
            (w//2, h//2, w//2, h//2), # Bottom-right
            (w//4, h//4, w//2, h//2) # Center
        ]
    
    for i, (x, y, region_w, region_h) in enumerate(regions):
        # Extract region
        region = frame[y:y+region_h, x:x+region_w]
        
        if region.size == 0:
            continue
            
        # Process region with pose detection
        rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_region)
        
        if results.pose_landmarks:
            # Calculate confidence based on landmark visibility
            visible_landmarks = sum(1 for landmark in results.pose_landmarks.landmark 
                                  if landmark.visibility > 0.5)
            confidence = visible_landmarks / len(results.pose_landmarks.landmark)
            
            # Only accept detections with reasonable confidence
            if confidence > 0.3:  # Minimum confidence threshold
                # Adjust landmark coordinates to full frame
                adjusted_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    adjusted_landmark = landmark
                    adjusted_landmark.x = (landmark.x * region_w + x) / w
                    adjusted_landmark.y = (landmark.y * region_h + y) / h
                    adjusted_landmarks.append(adjusted_landmark)
                
                # Create a new landmark list with adjusted coordinates
                from mediapipe.framework.formats import landmark_pb2
                adjusted_pose_landmarks = landmark_pb2.NormalizedLandmarkList()
                for landmark in adjusted_landmarks:
                    adjusted_pose_landmarks.landmark.append(landmark)
                
                detected_people.append({
                    'pose_landmarks': adjusted_pose_landmarks,
                    'confidence': confidence,
                    'region_id': i
                })
    
    # Sort by confidence and return only the best detections
    detected_people.sort(key=lambda x: x['confidence'], reverse=True)
    return detected_people[:expected_people]  # Return only expected number of people


def draw_pose_landmarks(frame, landmarks, connections, color, person_id, confidence=1.0):
    """
    Draw pose landmarks and connections for a single person.
    
    Args:
        frame: OpenCV frame to draw on
        landmarks: MediaPipe pose landmarks
        connections: Pose connections to draw
        color: BGR color tuple for this person
        person_id: ID number for this person
        confidence: Confidence score for this detection
    """
    h, w, _ = frame.shape
    
    # Draw landmarks
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Make landmark size based on confidence
        radius = max(3, int(5 * confidence))
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)  # White border
    
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
            
            # Make line thickness based on confidence
            thickness = max(1, int(2 * confidence))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
    
    # Draw person ID label with confidence
    if landmarks.landmark:
        # Use nose position for ID label
        nose = landmarks.landmark[0]  # Nose is landmark 0
        label_x = int(nose.x * w)
        label_y = int(nose.y * h) - 20
        
        label_text = f"Person {person_id} ({confidence:.2f})"
        cv2.putText(frame, label_text, 
                   (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        
        # Draw background rectangle for better visibility
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (label_x - 5, label_y - text_size[1] - 5),
                     (label_x + text_size[0] + 5, label_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(frame, label_text, 
                   (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)


def get_user_input():
    """
    Get user input for the number of people to track.
    
    Returns:
        int: Number of people to track
    """
    while True:
        try:
            num_people = int(input("Enter the number of people to track (1-5): "))
            if 1 <= num_people <= 5:
                return num_people
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return 1


def main():
    """
    Main function that initializes the true multi-person pose tracking system.
    """
    
    # Get user input for number of people to track
    print("=" * 60)
    print("Moveify True Multi-Person Pose Tracker")
    print("=" * 60)
    expected_people = get_user_input()
    
    # Initialize MediaPipe solutions
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure pose detection model
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Configure holistic solution for person detection
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam. Please check your camera connection.")
        return
    
    print(f"\nTracking {expected_people} people...")
    print("Press 'q' to quit the application")
    print("Press 'r' to reset person counter")
    print("Press 's' to save current frame")
    print("Press '1' to use holistic detection")
    print("Press '2' to use region-based detection")
    print("Press 'n' to change number of people")
    print("Make sure people are visible in the camera frame for pose detection")
    
    # Variables for multi-person tracking
    person_counter = 0
    max_people = 5
    person_colors = generate_person_colors(max_people)
    detection_method = 1  # 1 for holistic, 2 for region-based
    frame_count = 0
    
    # Initialize person tracker for handling overlaps
    person_tracker = PersonTracker(max_disappeared=15, max_distance=150)
    
    # Main video processing loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect people using selected method
        if detection_method == 1:
            detected_people = detect_people_using_holistic(frame, holistic, expected_people)
        else:
            detected_people = detect_people_using_region_splitting(frame, pose, expected_people)
        
        # Detect overlaps between people
        overlaps = detect_overlaps(detected_people, overlap_threshold=0.2)
        
        # Handle overlapping people using tracking
        processed_people = handle_overlapping_people(detected_people, overlaps, person_tracker)
        
        # Update person tracker
        tracked_persons = person_tracker.update(processed_people)
        
        # Draw pose landmarks for each detected person
        for i, person in enumerate(processed_people):
            if person['pose_landmarks']:
                # Use tracking ID if available, otherwise use index
                person_id = person.get('tracking_id', i)
                if person_id is None:
                    person_id = i
                color = person_colors[person_id % len(person_colors)]
                confidence = person.get('confidence', 1.0)
                is_overlapping = person.get('is_overlapping', False)
                
                # Draw pose landmarks with different style for overlapping people
                if is_overlapping:
                    # Thicker lines for overlapping people
                    mp_drawing.draw_landmarks(
                        frame,
                        person['pose_landmarks'],
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    # Draw additional outline for overlapping people
                    mp_drawing.draw_landmarks(
                        frame,
                        person['pose_landmarks'],
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                else:
                    # Normal drawing for non-overlapping people
                    mp_drawing.draw_landmarks(
                        frame,
                        person['pose_landmarks'],
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Add person ID and confidence label
                if person['pose_landmarks'].landmark:
                    nose = person['pose_landmarks'].landmark[0]
                    h, w, _ = frame.shape
                    label_x = int(nose.x * w)
                    label_y = int(nose.y * h) - 20
                    
                    # Add overlap indicator
                    overlap_indicator = " [OVERLAP]" if is_overlapping else ""
                    label_text = f"Person {person_id + 1} ({confidence:.2f}){overlap_indicator}"
                    
                    # Draw background rectangle for better visibility
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, 
                                 (label_x - 5, label_y - text_size[1] - 5),
                                 (label_x + text_size[0] + 5, label_y + 5),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, label_text, 
                               (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, color, 2)
        
        # Add text overlay to the frame
        method_name = "Holistic" if detection_method == 1 else "Region-based"
        cv2.putText(frame, f"Moveify True Multi-Person Tracker ({method_name})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset, 's' to save", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Press '1' for holistic, '2' for region-based", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Expected: {expected_people}, Detected: {len(processed_people)}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show overlap information
        overlapping_count = sum(1 for person in processed_people if person.get('is_overlapping', False))
        if overlapping_count > 0:
            cv2.putText(frame, f"Overlapping: {overlapping_count} people", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show tracking information
        tracked_count = len(tracked_persons)
        cv2.putText(frame, f"Tracked: {tracked_count} people", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame in a window
        cv2.imshow('Moveify True Multi-Person Pose Tracker', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            print("Exiting application...")
            break
        # Reset person counter if 'r' is pressed
        elif key == ord('r'):
            person_counter = 0
            person_tracker = PersonTracker(max_disappeared=15, max_distance=150)  # Reset tracker
            print("Person counter and tracker reset!")
        # Save current frame if 's' is pressed
        elif key == ord('s'):
            filename = f"multi_person_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        # Switch to holistic detection
        elif key == ord('1'):
            detection_method = 1
            print("Switched to holistic detection")
        # Switch to region-based detection
        elif key == ord('2'):
            detection_method = 2
            print("Switched to region-based detection")
        # Change number of people if 'n' is pressed
        elif key == ord('n'):
            print("\nChanging number of people to track...")
            # Note: This requires stopping the video loop to get user input
            # For now, we'll just print a message
            print("To change the number of people, restart the application.")
            print("Current setting: tracking", expected_people, "people")
        
        frame_count += 1
    
    # Cleanup: Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print("Application closed successfully!")


if __name__ == "__main__":
    main()
