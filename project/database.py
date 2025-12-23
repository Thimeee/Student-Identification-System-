import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="student_behavior.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                class_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date DATE NOT NULL,
                time_in TIME,
                time_out TIME,
                status TEXT DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                UNIQUE(student_id, date)
            )
        ''')
        
        # Behavior logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                session_date DATE NOT NULL,
                attention_score REAL,
                hand_raises INTEGER DEFAULT 0,
                looking_away_count INTEGER DEFAULT 0,
                total_duration INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully!")
    
    def add_student(self, student_id, name, email, class_name):
        """Add a new student to the database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO students (student_id, name, email, class_name)
                VALUES (?, ?, ?, ?)
            ''', (student_id, name, email, class_name))
            conn.commit()
            conn.close()
            return True, "Student added successfully!"
        except sqlite3.IntegrityError:
            return False, "Student ID already exists!"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_all_students(self):
        """Get all students from database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM students')
        students = cursor.fetchall()
        conn.close()
        return students
    
    def mark_attendance(self, student_id, date, time_in, status='Present'):
        """Mark attendance for a student"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO attendance (student_id, date, time_in, status)
                VALUES (?, ?, ?, ?)
            ''', (student_id, date, time_in, status))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error marking attendance: {str(e)}")
            return False
    
    def log_behavior(self, student_id, session_date, attention_score, 
                     hand_raises, looking_away_count, total_duration):
        """Log student behavior data"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO behavior_logs 
                (student_id, session_date, attention_score, hand_raises, 
                 looking_away_count, total_duration)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, session_date, attention_score, hand_raises, 
                  looking_away_count, total_duration))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging behavior: {str(e)}")
            return False
    
    def get_student_report(self, student_id, start_date=None, end_date=None):
        """Get behavior report for a student"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        query = '''
            SELECT session_date, attention_score, hand_raises, 
                   looking_away_count, total_duration
            FROM behavior_logs
            WHERE student_id = ?
        '''
        
        params = [student_id]
        
        if start_date and end_date:
            query += ' AND session_date BETWEEN ? AND ?'
            params.extend([start_date, end_date])
        
        query += ' ORDER BY session_date DESC'
        
        cursor.execute(query, params)
        report = cursor.fetchall()
        conn.close()
        return report

if __name__ == "__main__":
    # Test database
    db = DatabaseManager()
    print("Database setup completed!")