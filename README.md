# Facial Attendance System

This project is a facial recognition-based attendance system designed to streamline attendance tracking. It uses a webcam to detect faces, capture user information, and log attendance in real time. A pre-trained model enhances the accuracy of face recognition, and attendance data is stored in a CSV file for easy access.

## Features

- **Capture User Images**: Captures approximately 50 images per user and stores them in a named folder with corresponding names and IDs.
- **Real-time Face Recognition**: Utilizes the webcam for real-time face detection and recognition.
- **Attendance Logging**: Automatically logs attendance with timestamps in an `attendance.csv` file. If the file does not exist, the system creates it.
- **Streamlit Integration**: Provides a user-friendly interface through Streamlit for ease of use and better interaction.

