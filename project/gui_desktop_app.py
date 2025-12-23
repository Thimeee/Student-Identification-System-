#!/usr/bin/env python3
"""
Student Behavior Tracking System - Desktop GUI Application
Modern graphical interface using Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
from datetime import datetime
from database import DatabaseManager
from face_recognition_module import FaceRecognitionSystem
from gesture_detection import HandGestureDetector, FaceMovementDetector
import numpy as np

class BehaviorTrackingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Behavior Tracking System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2C3E50')
        
        # Initialize systems
        self.db = DatabaseManager()
        self.face_system = FaceRecognitionSystem()
        self.hand_detector = HandGestureDetector()
        self.face_detector = FaceMovementDetector()
        
        # Monitoring state
        self.is_monitoring = False
        self.cap = None
        self.session_start_time = None
        self.active_students = {}
        self.attendance_marked = set()
        
        # Create GUI
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
    def create_header(self):
        """Create header section"""
        header = tk.Frame(self.root, bg='#953e3f', height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        
        title = tk.Label(
            header,
            text="🎓 STUDENT BEHAVIOR TRACKING SYSTEM",
            font=("Arial", 24, "bold"),
            bg='#953e3f',
            fg='white'
        )
        title.pack(pady=20)
    
    def create_main_content(self):
        """Create main content area"""
        # Main container
        main_container = tk.Frame(self.root, bg='#2C3E50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video feed
        left_panel = tk.Frame(main_container, bg='#34495E', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video label
        video_label_frame = tk.Frame(left_panel, bg='#34495E')
        video_label_frame.pack(pady=10)
        
        tk.Label(
            video_label_frame,
            text="📹 LIVE CAMERA FEED",
            font=("Arial", 14, "bold"),
            bg='#34495E',
            fg='white'
        ).pack()
        
        self.video_label = tk.Label(left_panel, bg='black')
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = tk.Frame(left_panel, bg='#34495E')
        control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶ START MONITORING",
            command=self.start_monitoring,
            bg='#27AE60',
            fg='white',
            font=("Arial", 12, "bold"),
            width=20,
            height=2,
            cursor='hand2'
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹ STOP MONITORING",
            command=self.stop_monitoring,
            bg='#E74C3C',
            fg='white',
            font=("Arial", 12, "bold"),
            width=20,
            height=2,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Statistics and controls
        right_panel = tk.Frame(main_container, bg='#34495E', width=400, relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Statistics section
        stats_label = tk.Label(
            right_panel,
            text="📊 LIVE STATISTICS",
            font=("Arial", 14, "bold"),
            bg='#34495E',
            fg='white'
        )
        stats_label.pack(pady=10)
        
        # Stats display
        stats_frame = tk.Frame(right_panel, bg='#2C3E50', relief=tk.SUNKEN, bd=2)
        stats_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.session_time_label = self.create_stat_label(stats_frame, "⏱ Session Duration:", "00:00:00")
        self.fps_label = self.create_stat_label(stats_frame, "⚡ FPS:", "0")
        self.students_label = self.create_stat_label(stats_frame, "👥 Active Students:", "0")
        self.hand_raises_label = self.create_stat_label(stats_frame, "✋ Hand Raises:", "0")
        self.looking_away_label = self.create_stat_label(stats_frame, "👀 Looking Away:", "0")
        self.attention_label = self.create_stat_label(stats_frame, "🎯 Avg Attention:", "0%")
        
        # Student management section
        management_label = tk.Label(
            right_panel,
            text="👨‍🎓 STUDENT MANAGEMENT",
            font=("Arial", 14, "bold"),
            bg='#34495E',
            fg='white'
        )
        management_label.pack(pady=(20, 10))
        
        # Management buttons
        mgmt_frame = tk.Frame(right_panel, bg='#34495E')
        mgmt_frame.pack(pady=5)
        
        tk.Button(
            mgmt_frame,
            text="➕ Register New Student",
            command=self.open_register_window,
            bg='#3498DB',
            fg='white',
            font=("Arial", 10, "bold"),
            width=25,
            cursor='hand2'
        ).pack(pady=5)
        
        tk.Button(
            mgmt_frame,
            text="📋 View All Students",
            command=self.view_students,
            bg='#3498DB',
            fg='white',
            font=("Arial", 10, "bold"),
            width=25,
            cursor='hand2'
        ).pack(pady=5)
        
        tk.Button(
            mgmt_frame,
            text="📈 View Reports",
            command=self.view_reports,
            bg='#3498DB',
            fg='white',
            font=("Arial", 10, "bold"),
            width=25,
            cursor='hand2'
        ).pack(pady=5)
        
        tk.Button(
            mgmt_frame,
            text="🧪 Test Recognition",
            command=self.test_recognition,
            bg='#9B59B6',
            fg='white',
            font=("Arial", 10, "bold"),
            width=25,
            cursor='hand2'
        ).pack(pady=5)
        
        # Active students list
        students_list_label = tk.Label(
            right_panel,
            text="📝 ACTIVE STUDENTS",
            font=("Arial", 12, "bold"),
            bg='#34495E',
            fg='white'
        )
        students_list_label.pack(pady=(20, 10))
        
        self.students_text = scrolledtext.ScrolledText(
            right_panel,
            height=10,
            bg='#2C3E50',
            fg='white',
            font=("Courier", 10),
            relief=tk.SUNKEN,
            bd=2
        )
        self.students_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
    
    def create_stat_label(self, parent, label_text, value_text):
        """Create a statistic label"""
        frame = tk.Frame(parent, bg='#2C3E50')
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            frame,
            text=label_text,
            font=("Arial", 10, "bold"),
            bg='#2C3E50',
            fg='#ECF0F1',
            anchor='w'
        ).pack(side=tk.LEFT)
        
        value_label = tk.Label(
            frame,
            text=value_text,
            font=("Arial", 10, "bold"),
            bg='#2C3E50',
            fg='#27AE60',
            anchor='e'
        )
        value_label.pack(side=tk.RIGHT)
        
        return value_label
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Label(
            self.root,
            text="Ready | Camera: Not Connected",
            bg='#1C2833',
            fg='white',
            font=("Arial", 10),
            anchor='w',
            relief=tk.SUNKEN
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def start_monitoring(self):
        """Start monitoring session"""
        # Check if students are registered
        students = self.db.get_all_students()
        if not students:
            messagebox.showwarning(
                "No Students",
                "Please register at least one student before starting monitoring!"
            )
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera!")
            return
        
        self.is_monitoring = True
        self.session_start_time = datetime.now()
        self.active_students = {}
        self.attendance_marked = set()
        
        # Update buttons
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Update status
        self.status_bar.config(text="Monitoring Active | Camera: Connected")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        messagebox.showinfo("Monitoring Started", "Behavior monitoring session has started!")
    
    def stop_monitoring(self):
        """Stop monitoring session"""
        self.is_monitoring = False
        
        if self.cap:
            self.cap.release()
        
        # Update buttons
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Update status
        self.status_bar.config(text="Ready | Camera: Disconnected")
        
        # Clear video
        self.video_label.config(image='')
        
        # Save session report
        self.save_session_report()
        
        messagebox.showinfo("Monitoring Stopped", "Session ended. Report saved!")
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        frame_count = 0
        
        while self.is_monitoring:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Convert to PhotoImage
            cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize((800, 600), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update video label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Update statistics
            if frame_count % 10 == 0:  # Update every 10 frames
                self.update_statistics()
        
    def process_frame(self, frame):
        """Process video frame"""
        current_time = datetime.now()
        current_date = current_time.date().isoformat()
        
        # 1. Face Recognition
        recognized_students = self.face_system.recognize_faces(frame)
        frame = self.face_system.draw_face_boxes(frame, recognized_students)
        
        # Update active students
        for student in recognized_students:
            if student['student_id']:
                student_id = student['student_id']
                
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
                
                self.active_students[student_id]['last_seen'] = current_time
                self.active_students[student_id]['total_frames'] += 1
                
                # Mark attendance
                if student_id not in self.attendance_marked:
                    time_in = current_time.strftime("%H:%M:%S")
                    self.db.mark_attendance(student_id, current_date, time_in)
                    self.attendance_marked.add(student_id)
        
        # 2. Hand Detection
        hand_data, hand_results = self.hand_detector.detect_hands(frame)
        if hand_data['hands_detected']:
            frame = self.hand_detector.draw_hand_landmarks(frame, hand_results)
            
            if hand_data['hand_raised']:
                new_raise = self.hand_detector.update_hand_raise_count(True)
                if new_raise:
                    for student_id in self.active_students.keys():
                        self.active_students[student_id]['hand_raises'] += 1
            else:
                self.hand_detector.update_hand_raise_count(False)
        
        # 3. Face Movement
        face_data, _ = self.face_detector.detect_face_movement(frame)
        
        if face_data['face_detected']:
            for student_id in self.active_students.keys():
                self.active_students[student_id]['attention_scores'].append(
                    face_data['attention_score']
                )
            
            if face_data['looking_away']:
                new_looking_away = self.face_detector.update_looking_away_count(True)
                if new_looking_away:
                    for student_id in self.active_students.keys():
                        self.active_students[student_id]['looking_away_count'] += 1
            else:
                self.face_detector.update_looking_away_count(False)
        
        return frame
    
    def update_statistics(self):
        """Update statistics display"""
        # Session time
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            duration_str = str(duration).split('.')[0]
            self.session_time_label.config(text=duration_str)
        
        # FPS (approximate)
        self.fps_label.config(text="25-30")
        
        # Active students
        self.students_label.config(text=str(len(self.active_students)))
        
        # Hand raises
        hand_raises = self.hand_detector.get_hand_raise_count()
        self.hand_raises_label.config(text=str(hand_raises))
        
        # Looking away
        looking_away = self.face_detector.get_looking_away_count()
        self.looking_away_label.config(text=str(looking_away))
        
        # Average attention
        if self.active_students:
            total_attention = 0
            count = 0
            for data in self.active_students.values():
                if data['attention_scores']:
                    total_attention += np.mean(data['attention_scores'])
                    count += 1
            
            if count > 0:
                avg_attention = total_attention / count
                self.attention_label.config(text=f"{avg_attention:.1f}%")
        
        # Update students list
        self.update_students_list()
    
    def update_students_list(self):
        """Update active students list"""
        self.students_text.delete(1.0, tk.END)
        
        if not self.active_students:
            self.students_text.insert(tk.END, "No active students detected")
            return
        
        for student_id, data in self.active_students.items():
            avg_attention = np.mean(data['attention_scores']) if data['attention_scores'] else 0
            
            text = f"{data['name']}\n"
            text += f"  ID: {student_id}\n"
            text += f"  Attention: {avg_attention:.1f}%\n"
            text += f"  Hand Raises: {data['hand_raises']}\n"
            text += f"  Looking Away: {data['looking_away_count']}\n"
            text += "-" * 30 + "\n"
            
            self.students_text.insert(tk.END, text)
    
    def save_session_report(self):
        """Save session report"""
        if not self.session_start_time:
            return
        
        session_date = datetime.now().date().isoformat()
        
        for student_id, data in self.active_students.items():
            avg_attention = np.mean(data['attention_scores']) if data['attention_scores'] else 0
            student_duration = (data['last_seen'] - data['first_seen']).total_seconds()
            
            self.db.log_behavior(
                student_id=student_id,
                session_date=session_date,
                attention_score=avg_attention,
                hand_raises=data['hand_raises'],
                looking_away_count=data['looking_away_count'],
                total_duration=int(student_duration)
            )
    
    def open_register_window(self):
        """Open student registration window"""
        RegisterWindow(self.root, self.db, self.face_system)
    
    def view_students(self):
        """View all registered students"""
        students = self.db.get_all_students()
        
        if not students:
            messagebox.showinfo("No Students", "No students registered yet.")
            return
        
        # Create new window
        window = tk.Toplevel(self.root)
        window.title("Registered Students")
        window.geometry("800x600")
        window.configure(bg='#2C3E50')
        
        # Header
        tk.Label(
            window,
            text="📋 REGISTERED STUDENTS",
            font=("Arial", 16, "bold"),
            bg='#953e3f',
            fg='white'
        ).pack(fill=tk.X, pady=10)
        
        # Students list
        text = scrolledtext.ScrolledText(
            window,
            bg='#34495E',
            fg='white',
            font=("Courier", 10)
        )
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text.insert(tk.END, f"{'ID':<15} {'Name':<25} {'Class':<15} {'Email':<30}\n")
        text.insert(tk.END, "-" * 85 + "\n")
        
        for student in students:
            student_id = student[1]
            name = student[2]
            email = student[3] or "N/A"
            class_name = student[4] or "N/A"
            
            text.insert(tk.END, f"{student_id:<15} {name:<25} {class_name:<15} {email:<30}\n")
        
        text.config(state=tk.DISABLED)
    
    def view_reports(self):
        """View student reports"""
        ReportWindow(self.root, self.db)
    
    def test_recognition(self):
        """Test face recognition"""
        if self.is_monitoring:
            messagebox.showwarning("Already Monitoring", "Please stop monitoring first!")
            return
        
        messagebox.showinfo(
            "Test Mode",
            "Camera will open for testing.\nPress 'q' to close."
        )
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            recognized_students = self.face_system.recognize_faces(frame)
            frame = self.face_system.draw_face_boxes(frame, recognized_students)
            
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

class RegisterWindow:
    def __init__(self, parent, db, face_system):
        self.db = db
        self.face_system = face_system
        
        self.window = tk.Toplevel(parent)
        self.window.title("Register New Student")
        self.window.geometry("500x400")
        self.window.configure(bg='#2C3E50')
        
        # Header
        tk.Label(
            self.window,
            text="➕ REGISTER NEW STUDENT",
            font=("Arial", 16, "bold"),
            bg='#953e3f',
            fg='white'
        ).pack(fill=tk.X, pady=10)
        
        # Form
        form_frame = tk.Frame(self.window, bg='#34495E', relief=tk.RAISED, bd=2)
        form_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Student ID
        tk.Label(form_frame, text="Student ID:", bg='#34495E', fg='white',
                font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.id_entry = tk.Entry(form_frame, font=("Arial", 10), width=30)
        self.id_entry.grid(row=0, column=1, padx=10, pady=10)
        
        # Name
        tk.Label(form_frame, text="Name:", bg='#34495E', fg='white',
                font=("Arial", 10, "bold")).grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.name_entry = tk.Entry(form_frame, font=("Arial", 10), width=30)
        self.name_entry.grid(row=1, column=1, padx=10, pady=10)
        
        # Email
        tk.Label(form_frame, text="Email:", bg='#34495E', fg='white',
                font=("Arial", 10, "bold")).grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.email_entry = tk.Entry(form_frame, font=("Arial", 10), width=30)
        self.email_entry.grid(row=2, column=1, padx=10, pady=10)
        
        # Class
        tk.Label(form_frame, text="Class:", bg='#34495E', fg='white',
                font=("Arial", 10, "bold")).grid(row=3, column=0, padx=10, pady=10, sticky='w')
        self.class_entry = tk.Entry(form_frame, font=("Arial", 10), width=30)
        self.class_entry.grid(row=3, column=1, padx=10, pady=10)
        
        # Register button
        tk.Button(
            self.window,
            text="📸 Capture Face & Register",
            command=self.register_student,
            bg='#27AE60',
            fg='white',
            font=("Arial", 12, "bold"),
            cursor='hand2'
        ).pack(pady=10)
    
    def register_student(self):
        """Register student with face capture"""
        student_id = self.id_entry.get().strip()
        name = self.name_entry.get().strip()
        email = self.email_entry.get().strip()
        class_name = self.class_entry.get().strip()
        
        if not student_id or not name:
            messagebox.showerror("Error", "Student ID and Name are required!")
            return
        
        # Add to database
        success, message = self.db.add_student(student_id, name, email, class_name)
        
        if not success:
            messagebox.showerror("Error", message)
            return
        
        # Capture face
        messagebox.showinfo(
            "Capture Face",
            "Camera will open.\nPress SPACE to capture your face."
        )
        
        success, message = self.face_system.register_student_from_camera(student_id, name)
        
        if success:
            messagebox.showinfo("Success", f"Student {name} registered successfully!")
            self.window.destroy()
        else:
            messagebox.showerror("Error", message)

class ReportWindow:
    def __init__(self, parent, db):
        self.db = db
        
        self.window = tk.Toplevel(parent)
        self.window.title("Student Reports")
        self.window.geometry("900x700")
        self.window.configure(bg='#2C3E50')
        
        # Header
        tk.Label(
            self.window,
            text="📈 STUDENT BEHAVIOR REPORTS",
            font=("Arial", 16, "bold"),
            bg='#953e3f',
            fg='white'
        ).pack(fill=tk.X, pady=10)
        
        # Student ID entry
        input_frame = tk.Frame(self.window, bg='#34495E')
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Student ID:", bg='#34495E', fg='white',
                font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.id_entry = tk.Entry(input_frame, font=("Arial", 10), width=20)
        self.id_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            input_frame,
            text="📊 View Report",
            command=self.view_report,
            bg='#3498DB',
            fg='white',
            font=("Arial", 10, "bold"),
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)
        
        # Report display
        self.report_text = scrolledtext.ScrolledText(
            self.window,
            bg='#34495E',
            fg='white',
            font=("Courier", 10)
        )
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def view_report(self):
        """View report for student"""
        student_id = self.id_entry.get().strip()
        
        if not student_id:
            messagebox.showerror("Error", "Please enter Student ID!")
            return
        
        report = self.db.get_student_report(student_id)
        
        if not report:
            messagebox.showinfo("No Data", f"No records found for student ID: {student_id}")
            return
        
        self.report_text.delete(1.0, tk.END)
        
        self.report_text.insert(tk.END, f"{'Date':<12} {'Attention':<12} {'Hand Raises':<15} {'Looking Away':<15} {'Duration (min)':<15}\n")
        self.report_text.insert(tk.END, "-" * 70 + "\n")
        
        total_attention = 0
        total_hand_raises = 0
        total_looking_away = 0
        
        for record in report:
            date = record[0]
            attention = record[1]
            hand_raises = record[2]
            looking_away = record[3]
            duration = record[4] / 60
            
            self.report_text.insert(tk.END, 
                f"{date:<12} {attention:<12.1f} {hand_raises:<15} {looking_away:<15} {duration:<15.1f}\n")
            
            total_attention += attention
            total_hand_raises += hand_raises
            total_looking_away += looking_away
        
        # Summary
        num_sessions = len(report)
        avg_attention = total_attention / num_sessions if num_sessions > 0 else 0
        
        self.report_text.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.report_text.insert(tk.END, f"SUMMARY ({num_sessions} sessions):\n")
        self.report_text.insert(tk.END, f"  Average Attention: {avg_attention:.1f}%\n")
        self.report_text.insert(tk.END, f"  Total Hand Raises: {total_hand_raises}\n")
        self.report_text.insert(tk.END, f"  Total Looking Away: {total_looking_away}\n")

def main():
    """Main entry point"""
    root = tk.Tk()
    app = BehaviorTrackingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()