import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class HandGestureDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Hand raise detection state
        self.hand_raised = False
        self.hand_raise_start_time = None
        self.hand_raise_duration_threshold = 1.0  # seconds
        self.hand_raise_count = 0
        self.last_hand_raise_time = None
        self.cooldown_period = 3.0  # seconds between hand raises
    
    def detect_hands(self, frame):
        """Detect hands in frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        hand_data = {
            'hands_detected': False,
            'hand_raised': False,
            'hand_landmarks': []
        }
        
        if results.multi_hand_landmarks:
            hand_data['hands_detected'] = True
            hand_data['hand_landmarks'] = results.multi_hand_landmarks
            
            # Check if hand is raised
            hand_data['hand_raised'] = self._is_hand_raised(results.multi_hand_landmarks[0])
        
        return hand_data, results
    
    def _is_hand_raised(self, hand_landmarks):
        """Check if hand is raised (above head level)"""
        # Get wrist and middle finger tip positions
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Hand is raised if wrist is significantly above middle finger
        # and hand is in upper portion of frame
        hand_raised = (wrist.y < 0.4 and  # Hand in upper 40% of frame
                      middle_finger_tip.y < wrist.y)  # Fingers pointing up
        
        return hand_raised
    
    def update_hand_raise_count(self, hand_raised):
        """Update hand raise count with debouncing"""
        current_time = datetime.now()
        
        if hand_raised:
            if not self.hand_raised:  # New hand raise
                self.hand_raised = True
                self.hand_raise_start_time = current_time
            elif self.hand_raise_start_time:
                # Check if hand has been raised long enough
                duration = (current_time - self.hand_raise_start_time).total_seconds()
                
                if duration >= self.hand_raise_duration_threshold:
                    # Check cooldown period
                    if (self.last_hand_raise_time is None or 
                        (current_time - self.last_hand_raise_time).total_seconds() >= self.cooldown_period):
                        self.hand_raise_count += 1
                        self.last_hand_raise_time = current_time
                        self.hand_raise_start_time = None  # Reset
                        return True  # New hand raise detected
        else:
            self.hand_raised = False
            self.hand_raise_start_time = None
        
        return False
    
    def draw_hand_landmarks(self, frame, results):
        """Draw hand landmarks on frame"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        return frame
    
    def get_hand_raise_count(self):
        """Get total hand raise count"""
        return self.hand_raise_count
    
    def reset_count(self):
        """Reset hand raise count"""
        self.hand_raise_count = 0
        self.hand_raised = False
        self.hand_raise_start_time = None
        self.last_hand_raise_time = None
    
    def release(self):
        """Release resources"""
        self.hands.close()


class FaceMovementDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
        
        # Attention tracking
        self.looking_away_count = 0
        self.is_looking_away = False
        self.looking_away_start_time = None
        self.looking_away_threshold = 2.0  # seconds
        self.last_looking_away_time = None
        self.cooldown_period = 5.0  # seconds
        
        # Head pose thresholds
        self.yaw_threshold = 20  # degrees
        self.pitch_threshold = 15  # degrees
    
    def detect_face_movement(self, frame):
        """Detect face and head pose"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        face_data = {
            'face_detected': False,
            'head_pose': None,
            'looking_away': False,
            'attention_score': 0
        }
        
        if results.multi_face_landmarks:
            face_data['face_detected'] = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate head pose
            head_pose = self._calculate_head_pose(face_landmarks, frame.shape)
            face_data['head_pose'] = head_pose
            
            # Determine if looking away
            if head_pose:
                yaw, pitch = head_pose['yaw'], head_pose['pitch']
                looking_away = (abs(yaw) > self.yaw_threshold or 
                              abs(pitch) > self.pitch_threshold)
                face_data['looking_away'] = looking_away
                
                # Calculate attention score (0-100)
                yaw_score = max(0, 100 - abs(yaw) * 2)
                pitch_score = max(0, 100 - abs(pitch) * 3)
                face_data['attention_score'] = (yaw_score + pitch_score) / 2
        
        return face_data, results
    
    def _calculate_head_pose(self, face_landmarks, image_shape):
        """Calculate head pose angles (yaw, pitch, roll)"""
        h, w = image_shape[:2]
        
        # Key facial landmarks for head pose estimation
        # Nose tip, chin, left eye, right eye, left mouth, right mouth
        landmarks_2d = []
        landmarks_3d = []
        
        # Define 3D model points (approximate)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Get corresponding 2D points
        landmark_indices = [1, 152, 33, 263, 61, 291]  # Nose, Chin, Left eye, Right eye, Left mouth, Right mouth
        
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append([x, y])
            landmarks_3d.append([landmark.x * w, landmark.y * h, landmark.z * w])
        
        landmarks_2d = np.array(landmarks_2d, dtype=np.float64)
        
        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles
            pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch = euler_angles[0][0]
            yaw = euler_angles[1][0]
            roll = euler_angles[2][0]
            
            return {
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll
            }
        
        return None
    
    def update_looking_away_count(self, looking_away):
        """Update looking away count with debouncing"""
        current_time = datetime.now()
        
        if looking_away:
            if not self.is_looking_away:
                self.is_looking_away = True
                self.looking_away_start_time = current_time
            elif self.looking_away_start_time:
                duration = (current_time - self.looking_away_start_time).total_seconds()
                
                if duration >= self.looking_away_threshold:
                    if (self.last_looking_away_time is None or
                        (current_time - self.last_looking_away_time).total_seconds() >= self.cooldown_period):
                        self.looking_away_count += 1
                        self.last_looking_away_time = current_time
                        self.looking_away_start_time = None
                        return True
        else:
            self.is_looking_away = False
            self.looking_away_start_time = None
        
        return False
    
    def draw_face_mesh(self, frame, results):
        """Draw face mesh on frame"""
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0, 255, 0), thickness=1
                    )
                )
        return frame
    
    def get_looking_away_count(self):
        """Get total looking away count"""
        return self.looking_away_count
    
    def reset_count(self):
        """Reset looking away count"""
        self.looking_away_count = 0
        self.is_looking_away = False
        self.looking_away_start_time = None
        self.last_looking_away_time = None
    
    def release(self):
        """Release resources"""
        self.face_mesh.close()

if __name__ == "__main__":
    print("Hand and Face Movement Detector initialized")