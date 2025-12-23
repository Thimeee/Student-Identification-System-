import cv2
from database import DatabaseManager
from face_recognition_module import FaceRecognitionSystem

class StudentManagement:
    def __init__(self):
        self.db = DatabaseManager()
        self.face_system = FaceRecognitionSystem()
    
    def register_new_student(self):
        """Register a new student with face capture"""
        print("\n" + "=" * 60)
        print("STUDENT REGISTRATION")
        print("=" * 60)
        
        # Get student details
        student_id = input("Enter Student ID: ").strip()
        if not student_id:
            print("❌ Student ID cannot be empty!")
            return
        
        name = input("Enter Student Name: ").strip()
        if not name:
            print("❌ Name cannot be empty!")
            return
        
        email = input("Enter Email (optional): ").strip()
        class_name = input("Enter Class/Section: ").strip()
        
        # Add to database
        success, message = self.db.add_student(student_id, name, email, class_name)
        
        if not success:
            print(f"❌ {message}")
            return
        
        print(f"✓ {message}")
        
        # Capture face
        print("\n📸 Now capturing face for recognition...")
        print("Please look at the camera and press SPACE when ready")
        
        success, message = self.face_system.register_student_from_camera(student_id, name)
        
        if success:
            print(f"✓ {message}")
            print(f"\n✓ Student {name} registered successfully!")
        else:
            print(f"❌ {message}")
            print("⚠️  Student added to database but face registration failed.")
            print("    You can retry face registration later.")
    
    def register_student_from_image(self):
        """Register student from an image file"""
        print("\n" + "=" * 60)
        print("STUDENT REGISTRATION FROM IMAGE")
        print("=" * 60)
        
        student_id = input("Enter Student ID: ").strip()
        if not student_id:
            print("❌ Student ID cannot be empty!")
            return
        
        name = input("Enter Student Name: ").strip()
        if not name:
            print("❌ Name cannot be empty!")
            return
        
        email = input("Enter Email (optional): ").strip()
        class_name = input("Enter Class/Section: ").strip()
        image_path = input("Enter image file path: ").strip()
        
        # Add to database
        success, message = self.db.add_student(student_id, name, email, class_name)
        
        if not success:
            print(f"❌ {message}")
            return
        
        print(f"✓ {message}")
        
        # Register face from image
        success, message = self.face_system.register_student(image_path, student_id, name)
        
        if success:
            print(f"✓ {message}")
        else:
            print(f"❌ {message}")
    
    def view_all_students(self):
        """View all registered students"""
        print("\n" + "=" * 60)
        print("REGISTERED STUDENTS")
        print("=" * 60)
        
        students = self.db.get_all_students()
        
        if not students:
            print("No students registered yet.")
            return
        
        print(f"\nTotal Students: {len(students)}\n")
        print(f"{'ID':<15} {'Name':<25} {'Class':<15} {'Email':<30}")
        print("-" * 85)
        
        for student in students:
            student_id = student[1]
            name = student[2]
            email = student[3] or "N/A"
            class_name = student[4] or "N/A"
            
            print(f"{student_id:<15} {name:<25} {class_name:<15} {email:<30}")
        
        print("-" * 85)
        
        # Show face recognition status
        print(f"\nFace encodings registered: {len(self.face_system.known_face_names)}")
    
    def view_student_report(self):
        """View behavior report for a student"""
        print("\n" + "=" * 60)
        print("STUDENT BEHAVIOR REPORT")
        print("=" * 60)
        
        student_id = input("Enter Student ID: ").strip()
        
        if not student_id:
            print("❌ Student ID cannot be empty!")
            return
        
        # Get date range
        print("\nDate range (leave empty for all records):")
        start_date = input("Start date (YYYY-MM-DD): ").strip() or None
        end_date = input("End date (YYYY-MM-DD): ").strip() or None
        
        # Get report
        report = self.db.get_student_report(student_id, start_date, end_date)
        
        if not report:
            print(f"\n❌ No records found for student ID: {student_id}")
            return
        
        print(f"\n{'Date':<12} {'Attention':<12} {'Hand Raises':<15} {'Looking Away':<15} {'Duration (min)':<15}")
        print("-" * 70)
        
        total_attention = 0
        total_hand_raises = 0
        total_looking_away = 0
        total_duration = 0
        
        for record in report:
            date = record[0]
            attention = record[1]
            hand_raises = record[2]
            looking_away = record[3]
            duration = record[4] / 60  # Convert to minutes
            
            print(f"{date:<12} {attention:<12.1f} {hand_raises:<15} {looking_away:<15} {duration:<15.1f}")
            
            total_attention += attention
            total_hand_raises += hand_raises
            total_looking_away += looking_away
            total_duration += duration
        
        print("-" * 70)
        
        # Calculate averages
        num_sessions = len(report)
        avg_attention = total_attention / num_sessions if num_sessions > 0 else 0
        
        print(f"\nSummary for {num_sessions} session(s):")
        print(f"  Average Attention Score: {avg_attention:.1f}%")
        print(f"  Total Hand Raises: {total_hand_raises}")
        print(f"  Total Looking Away Incidents: {total_looking_away}")
        print(f"  Total Time Tracked: {total_duration:.1f} minutes")
    
    def test_face_recognition(self):
        """Test face recognition with webcam"""
        print("\n" + "=" * 60)
        print("TESTING FACE RECOGNITION")
        print("=" * 60)
        print("Press 'q' to quit")
        print()
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recognize faces
            recognized_students = self.face_system.recognize_faces(frame)
            
            # Draw boxes and labels
            frame = self.face_system.draw_face_boxes(frame, recognized_students)
            
            # Display
            cv2.imshow('Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Test completed")
    
    def main_menu(self):
        """Display main menu"""
        while True:
            print("\n" + "=" * 60)
            print("STUDENT MANAGEMENT SYSTEM")
            print("=" * 60)
            print("1. Register New Student (with camera)")
            print("2. Register Student from Image File")
            print("3. View All Students")
            print("4. View Student Behavior Report")
            print("5. Test Face Recognition")
            print("6. Exit")
            print("=" * 60)
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self.register_new_student()
            elif choice == '2':
                self.register_student_from_image()
            elif choice == '3':
                self.view_all_students()
            elif choice == '4':
                self.view_student_report()
            elif choice == '5':
                self.test_face_recognition()
            elif choice == '6':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please try again.")

if __name__ == "__main__":
    try:
        management = StudentManagement()
        management.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")