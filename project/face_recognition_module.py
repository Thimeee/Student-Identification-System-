import face_recognition
import cv2
import numpy as np
import pickle
import os
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, encodings_file="face_encodings.pkl"):
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_student_ids = []
        self.load_encodings()
    
    def load_encodings(self):
        """Load face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    self.known_student_ids = data['student_ids']
                print(f"Loaded {len(self.known_face_names)} face encodings")
            except Exception as e:
                print(f"Error loading encodings: {str(e)}")
        else:
            print("No existing encodings found. Starting fresh.")
    
    def save_encodings(self):
        """Save face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'student_ids': self.known_student_ids
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print("Encodings saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving encodings: {str(e)}")
            return False
    
    def register_student(self, image_path, student_id, name):
        """Register a new student's face"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                return False, "No face detected in the image!"
            
            if len(face_encodings) > 1:
                return False, "Multiple faces detected! Please use an image with only one face."
            
            # Add to known faces
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            self.known_student_ids.append(student_id)
            
            # Save encodings
            self.save_encodings()
            
            return True, f"Student {name} registered successfully!"
        
        except Exception as e:
            return False, f"Error registering student: {str(e)}"
    
    def register_student_from_camera(self, student_id, name):
        """Register a student using webcam"""
        print(f"\n=== Registering {name} (ID: {student_id}) ===")
        print("Press 'SPACE' to capture image")
        print("Press 'q' to cancel")
        
        cap = cv2.VideoCapture(0)
        captured = False
        face_encoding = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            cv2.putText(frame, "Press SPACE to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Registering: {name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Detect faces in frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Draw rectangle around faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            cv2.imshow('Register Student - Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture on SPACE
            if key == ord(' '):
                if len(face_locations) == 0:
                    print("No face detected! Try again.")
                    continue
                elif len(face_locations) > 1:
                    print("Multiple faces detected! Please ensure only one person is visible.")
                    continue
                else:
                    # Get face encoding
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        captured = True
                        print("Face captured successfully!")
                        break
            
            # Cancel on 'q'
            if key == ord('q'):
                print("Registration cancelled.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured and face_encoding is not None:
            # Add to known faces
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.known_student_ids.append(student_id)
            
            # Save encodings
            self.save_encodings()
            return True, f"Student {name} registered successfully!"
        else:
            return False, "Registration failed or cancelled."
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_students = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if face matches known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=0.6
            )
            
            name = "Unknown"
            student_id = None
            
            # Find best match
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    student_id = self.known_student_ids[best_match_index]
            
            recognized_students.append({
                'name': name,
                'student_id': student_id,
                'location': face_location,
                'confidence': 1 - face_distances[best_match_index] if len(face_distances) > 0 and matches[best_match_index] else 0
            })
        
        return recognized_students
    
    def draw_face_boxes(self, frame, recognized_students):
        """Draw boxes around recognized faces"""
        for student in recognized_students:
            top, right, bottom, left = student['location']
            
            # Choose color based on recognition
            color = (0, 255, 0) if student['student_id'] else (0, 0, 255)
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw name
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{student['name']}"
            if student['confidence'] > 0:
                text += f" ({student['confidence']:.2%})"
            
            cv2.putText(frame, text, (left + 6, bottom - 6),
                       font, 0.5, (255, 255, 255), 1)
        
        return frame

if __name__ == "__main__":
    # Test face recognition system
    face_system = FaceRecognitionSystem()
    print(f"Face Recognition System initialized with {len(face_system.known_face_names)} students")