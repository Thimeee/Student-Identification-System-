"""
Student Behavior Tracking System - Web Application
Flask-based web interface
"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
import cv2
import json
from datetime import datetime
from database import DatabaseManager
from face_recognition_module import FaceRecognitionSystem
from gesture_detection import HandGestureDetector, FaceMovementDetector
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Initialize systems
db = DatabaseManager()
face_system = FaceRecognitionSystem()
hand_detector = HandGestureDetector()
face_detector = FaceMovementDetector()

# Global state
monitoring_state = {
    'is_monitoring': False,
    'session_start_time': None,
    'active_students': {},
    'attendance_marked': set(),
    'stats': {
        'hand_raises': 0,
        'looking_away': 0,
        'active_count': 0,
        'avg_attention': 0
    }
}

camera = None

@app.route('/')
def index():
    """Home page"""
    students = db.get_all_students()
    return render_template('index.html', 
                         student_count=len(students),
                         monitoring=monitoring_state['is_monitoring'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Student registration page"""
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        email = request.form.get('email')
        class_name = request.form.get('class_name')
        
        success, message = db.add_student(student_id, name, email, class_name)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('capture_face', student_id=student_id, name=name))
        else:
            flash(message, 'error')
    
    return render_template('register.html')

@app.route('/capture_face/<student_id>/<name>')
def capture_face(student_id, name):
    """Face capture page"""
    return render_template('capture_face.html', student_id=student_id, name=name)

@app.route('/students')
def students():
    """View all students"""
    all_students = db.get_all_students()
    return render_template('students.html', students=all_students)

@app.route('/monitor')
def monitor():
    """Monitoring page"""
    return render_template('monitor.html')

@app.route('/reports')
def reports():
    """Reports page"""
    all_students = db.get_all_students()
    return render_template('reports.html', students=all_students)

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring session"""
    global camera, monitoring_state
    
    students = db.get_all_students()
    if not students:
        return jsonify({'success': False, 'message': 'No students registered!'})
    
    if not monitoring_state['is_monitoring']:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'success': False, 'message': 'Could not open camera!'})
        
        monitoring_state['is_monitoring'] = True
        monitoring_state['session_start_time'] = datetime.now()
        monitoring_state['active_students'] = {}
        monitoring_state['attendance_marked'] = set()
        
        hand_detector.reset_count()
        face_detector.reset_count()
        
        return jsonify({'success': True, 'message': 'Monitoring started!'})
    
    return jsonify({'success': False, 'message': 'Already monitoring!'})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring session"""
    global camera, monitoring_state
    
    if monitoring_state['is_monitoring']:
        monitoring_state['is_monitoring'] = False
        
        if camera:
            camera.release()
            camera = None
        
        # Save session report
        save_session_report()
        
        return jsonify({'success': True, 'message': 'Monitoring stopped!'})
    
    return jsonify({'success': False, 'message': 'Not currently monitoring!'})

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    if monitoring_state['session_start_time']:
        duration = datetime.now() - monitoring_state['session_start_time']
        duration_str = str(duration).split('.')[0]
    else:
        duration_str = "00:00:00"
    
    # Calculate average attention
    if monitoring_state['active_students']:
        total_attention = 0
        count = 0
        for data in monitoring_state['active_students'].values():
            if data['attention_scores']:
                total_attention += np.mean(data['attention_scores'])
                count += 1
        
        avg_attention = (total_attention / count) if count > 0 else 0
    else:
        avg_attention = 0
    
    return jsonify({
        'duration': duration_str,
        'active_students': len(monitoring_state['active_students']),
        'hand_raises': hand_detector.get_hand_raise_count(),
        'looking_away': face_detector.get_looking_away_count(),
        'avg_attention': f"{avg_attention:.1f}",
        'students_list': [
            {
                'name': data['name'],
                'id': sid,
                'attention': f"{np.mean(data['attention_scores']):.1f}" if data['attention_scores'] else "0",
                'hand_raises': data['hand_raises']
            }
            for sid, data in monitoring_state['active_students'].items()
        ]
    })

@app.route('/api/report/<student_id>')
def get_report(student_id):
    """Get student report"""
    report = db.get_student_report(student_id)
    
    if not report:
        return jsonify({'success': False, 'message': 'No records found'})
    
    report_data = []
    total_attention = 0
    total_hand_raises = 0
    total_looking_away = 0
    
    for record in report:
        report_data.append({
            'date': record[0],
            'attention': record[1],
            'hand_raises': record[2],
            'looking_away': record[3],
            'duration': record[4] / 60
        })
        
        total_attention += record[1]
        total_hand_raises += record[2]
        total_looking_away += record[3]
    
    num_sessions = len(report)
    
    return jsonify({
        'success': True,
        'report': report_data,
        'summary': {
            'sessions': num_sessions,
            'avg_attention': total_attention / num_sessions if num_sessions > 0 else 0,
            'total_hand_raises': total_hand_raises,
            'total_looking_away': total_looking_away
        }
    })

def generate_frames():
    """Generate video frames"""
    global camera, monitoring_state
    
    while monitoring_state['is_monitoring']:
        if camera is None or not camera.isOpened():
            break
        
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame
        frame = process_frame(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frame(frame):
    """Process video frame"""
    current_time = datetime.now()
    current_date = current_time.date().isoformat()
    
    # Face Recognition
    recognized_students = face_system.recognize_faces(frame)
    frame = face_system.draw_face_boxes(frame, recognized_students)
    
    # Update active students
    for student in recognized_students:
        if student['student_id']:
            student_id = student['student_id']
            
            if student_id not in monitoring_state['active_students']:
                monitoring_state['active_students'][student_id] = {
                    'name': student['name'],
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'hand_raises': 0,
                    'looking_away_count': 0,
                    'attention_scores': [],
                    'total_frames': 0
                }
            
            monitoring_state['active_students'][student_id]['last_seen'] = current_time
            monitoring_state['active_students'][student_id]['total_frames'] += 1
            
            # Mark attendance
            if student_id not in monitoring_state['attendance_marked']:
                time_in = current_time.strftime("%H:%M:%S")
                db.mark_attendance(student_id, current_date, time_in)
                monitoring_state['attendance_marked'].add(student_id)
    
    # Hand Detection
    hand_data, hand_results = hand_detector.detect_hands(frame)
    if hand_data['hands_detected']:
        frame = hand_detector.draw_hand_landmarks(frame, hand_results)
        
        if hand_data['hand_raised']:
            new_raise = hand_detector.update_hand_raise_count(True)
            if new_raise:
                for student_id in monitoring_state['active_students'].keys():
                    monitoring_state['active_students'][student_id]['hand_raises'] += 1
        else:
            hand_detector.update_hand_raise_count(False)
    
    # Face Movement
    face_data, _ = face_detector.detect_face_movement(frame)
    
    if face_data['face_detected']:
        for student_id in monitoring_state['active_students'].keys():
            monitoring_state['active_students'][student_id]['attention_scores'].append(
                face_data['attention_score']
            )
        
        if face_data['looking_away']:
            new_looking_away = face_detector.update_looking_away_count(True)
            if new_looking_away:
                for student_id in monitoring_state['active_students'].keys():
                    monitoring_state['active_students'][student_id]['looking_away_count'] += 1
        else:
            face_detector.update_looking_away_count(False)
    
    return frame

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def save_session_report():
    """Save session report to database"""
    if not monitoring_state['session_start_time']:
        return
    
    session_date = datetime.now().date().isoformat()
    
    for student_id, data in monitoring_state['active_students'].items():
        avg_attention = np.mean(data['attention_scores']) if data['attention_scores'] else 0
        student_duration = (data['last_seen'] - data['first_seen']).total_seconds()
        
        db.log_behavior(
            student_id=student_id,
            session_date=session_date,
            attention_score=avg_attention,
            hand_raises=data['hand_raises'],
            looking_away_count=data['looking_away_count'],
            total_duration=int(student_duration)
        )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)