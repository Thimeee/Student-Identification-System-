import cv2
import numpy as np
from datetime import datetime, date
import time
from database import DatabaseManager
from face_recognition_module import FaceRecognitionSystem
from gesture_detection import HandGestureDetector, FaceMovementDetector

class BehaviorTrackingSystem:
    def __init__(self):
        print("=" * 60)
        print("Initializing Student Behavior Tracking System...")
        print("=" * 60)
        
        # Initialize components
        self.db = DatabaseManager()
        self.face_system = FaceRecognitionSystem()
        self.hand_detector = HandGestureDetector()
        self.face_movement_detector = FaceMovementDetector()
        
        # Session tracking
        self.session_start_time = None
        self.active_students = {}  # student_id: {data}
        self.attendance_marked = set()
        
        # Statistics
        self.total_frames = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        print("✓ All systems initialized successfully!")
        print()
    
    def start_monitoring_session(self):
        """Start a new monitoring session"""
        print("\n" + "=" * 60)
        print("Starting Monitoring Session")
        print("=" * 60)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  'r'   - Reset statistics")
        print("  's'   - Save session report")
        print("  'q'   - Quit")
        print("=" * 60 + "\n")
        
        self.session_start_time = datetime.now()
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        break
                    
                    # Process frame
                    processed_frame = self._process_frame(frame)
                    
                    # Display frame
                    cv2.imshow('Student Behavior Tracking System', processed_frame)
                    
                    # Update FPS
                    self.total_frames += 1
                    if time.time() - self.last_fps_update >= 1.0:
                        self.fps = self.total_frames / (time.time() - self.last_fps_update)
                        self.total_frames = 0
                        self.last_fps_update = time.time()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nEnding session...")
                    break
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n{status}")
                elif key == ord('r'):
                    self._reset_statistics()
                    print("\nStatistics reset!")
                elif key == ord('s'):
                    self._save_session_report()
                    print("\nSession report saved!")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self._save_session_report()
            print("\n✓ Session ended and report saved")
    
    def _process_frame(self, frame):
        """Process a single frame"""
        display_frame = frame.copy()
        current_time = datetime.now()
        current_date = date.today().isoformat()
        
        # 1. Face Recognition
        recognized_students = self.face_system.recognize_faces(frame)
        display_frame = self.face_system.draw_face_boxes(display_frame, recognized_students)
        
        # Update active students and mark attendance
        for student in recognized_students:
            if student['student_id']:
                student_id = student['student_id']
                
                # Initialize student data if new
                if student_id not in self.active_students:
                    self.active_students[student_id] = {
                        'name': student['name'],
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'hand_raises': 0,
                        'looking_away_count': 0,
                        'attention_scores': [],
                        'total_frames': 0
                    }
                
                # Update last seen
                self.active_students[student_id]['last_seen'] = current_time
                self.active_students[student_id]['total_frames'] += 1
                
                # Mark attendance (once per day)
                if student_id not in self.attendance_marked:
                    time_in = current_time.strftime("%H:%M:%S")
                    self.db.mark_attendance(student_id, current_date, time_in)
                    self.attendance_marked.add(student_id)
                    print(f"✓ Attendance marked for {student['name']} at {time_in}")
        
        # 2. Hand Gesture Detection
        hand_data, hand_results = self.hand_detector.detect_hands(frame)
        if hand_data['hands_detected']:
            display_frame = self.hand_detector.draw_hand_landmarks(display_frame, hand_results)
            
            # Check for hand raise
            if hand_data['hand_raised']:
                new_raise = self.hand_detector.update_hand_raise_count(True)
                if new_raise:
                    # Attribute hand raise to visible students
                    for student_id in self.active_students.keys():
                        self.active_students[student_id]['hand_raises'] += 1
                    print(f"✋ Hand raised detected!")
            else:
                self.hand_detector.update_hand_raise_count(False)
        
        # 3. Face Movement Detection
        face_data, face_results = self.face_movement_detector.detect_face_movement(frame)
        
        if face_data['face_detected']:
            # Update attention scores for active students
            for student_id in self.active_students.keys():
                self.active_students[student_id]['attention_scores'].append(
                    face_data['attention_score']
                )
            
            # Check if looking away
            if face_data['looking_away']:
                new_looking_away = self.face_movement_detector.update_looking_away_count(True)
                if new_looking_away:
                    for student_id in self.active_students.keys():
                        self.active_students[student_id]['looking_away_count'] += 1
                    print(f"👀 Student looking away detected!")
            else:
                self.face_movement_detector.update_looking_away_count(False)
        
        # 4. Draw UI overlay
        display_frame = self._draw_ui_overlay(display_frame, face_data, hand_data)
        
        return display_frame
    
    def _draw_ui_overlay(self, frame, face_data, hand_data):
        """Draw UI overlay with statistics"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for stats panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Session info
        y_offset = 40
        line_height = 30
        
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            duration_str = str(duration).split('.')[0]
            cv2.putText(frame, f"Session Duration: {duration_str}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # Active students
        active_count = len(self.active_students)
        cv2.putText(frame, f"Active Students: {active_count}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Hand raises
        total_hand_raises = self.hand_detector.get_hand_raise_count()
        color = (0, 255, 255) if hand_data.get('hand_raised', False) else (255, 255, 255)
        cv2.putText(frame, f"Hand Raises: {total_hand_raises}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        # Looking away count
        total_looking_away = self.face_movement_detector.get_looking_away_count()
        color = (0, 0, 255) if face_data.get('looking_away', False) else (255, 255, 255)
        cv2.putText(frame, f"Looking Away: {total_looking_away}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        # Attention score
        if face_data.get('attention_score'):
            attention = face_data['attention_score']
            color = (0, 255, 0) if attention > 70 else (0, 165, 255) if attention > 40 else (0, 0, 255)
            cv2.putText(frame, f"Attention: {attention:.1f}%", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height
        
        # Head pose
        if face_data.get('head_pose'):
            pose = face_data['head_pose']
            cv2.putText(frame, f"Yaw: {pose['yaw']:.1f}° Pitch: {pose['pitch']:.1f}°", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Student list
        if self.active_students:
            y_offset = 330
            cv2.putText(frame, "Active Students:", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            
            for student_id, data in list(self.active_students.items())[:5]:  # Show max 5
                avg_attention = np.mean(data['attention_scores']) if data['attention_scores'] else 0
                text = f"{data['name']}: Att={avg_attention:.0f}% HR={data['hand_raises']}"
                cv2.putText(frame, text, 
                           (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 22
        
        # Status indicators
        status_y = h - 30
        if hand_data.get('hand_raised'):
            cv2.putText(frame, "✋ HAND RAISED", 
                       (w - 250, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if face_data.get('looking_away'):
            cv2.putText(frame, "👀 LOOKING AWAY", 
                       (w - 250, status_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def _reset_statistics(self):
        """Reset session statistics"""
        self.hand_detector.reset_count()
        self.face_movement_detector.reset_count()
        for student_id in self.active_students:
            self.active_students[student_id]['hand_raises'] = 0
            self.active_students[student_id]['looking_away_count'] = 0
            self.active_students[student_id]['attention_scores'] = []
    
    def _save_session_report(self):
        """Save session report to database"""
        if not self.session_start_time:
            return
        
        session_date = date.today().isoformat()
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("SESSION REPORT")
        print("=" * 60)
        print(f"Date: {session_date}")
        print(f"Duration: {session_duration / 60:.1f} minutes")
        print(f"Total Students: {len(self.active_students)}")
        print("-" * 60)
        
        for student_id, data in self.active_students.items():
            # Calculate average attention score
            avg_attention = np.mean(data['attention_scores']) if data['attention_scores'] else 0
            
            # Calculate student's session duration
            student_duration = (data['last_seen'] - data['first_seen']).total_seconds()
            
            print(f"\nStudent: {data['name']} (ID: {student_id})")
            print(f"  Attention Score: {avg_attention:.1f}%")
            print(f"  Hand Raises: {data['hand_raises']}")
            print(f"  Looking Away Count: {data['looking_away_count']}")
            print(f"  Time Present: {student_duration / 60:.1f} minutes")
            
            # Save to database
            self.db.log_behavior(
                student_id=student_id,
                session_date=session_date,
                attention_score=avg_attention,
                hand_raises=data['hand_raises'],
                looking_away_count=data['looking_away_count'],
                total_duration=int(student_duration)
            )
        
        print("\n" + "=" * 60)
        print("✓ Report saved to database")
        print("=" * 60)
    
    def cleanup(self):
        """Cleanup resources"""
        self.hand_detector.release()
        self.face_movement_detector.release()

if __name__ == "__main__":
    try:
        system = BehaviorTrackingSystem()
        system.start_monitoring_session()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        if 'system' in locals():
            system.cleanup()
        print("\nThank you for using Student Behavior Tracking System!")