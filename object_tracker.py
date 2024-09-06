import cv2
import numpy as np

# Kalman filter initialization
def create_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic parameters (x, y, dx, dy), 2 measured parameters (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
    return kalman

# Function to detect and track the ball
def track_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Kalman filter for object tracking
    kalman = create_kalman_filter()
    
    trajectory = []  # To store the trajectory of the ball
    first_detection = False  # To track if the first detection was made
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for detecting the ball (adjust to the specific ball's color)
        lower_color = np.array([29, 86, 6])  # Example for greenish-yellow ball
        upper_color = np.array([64, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        # Apply morphological transformations to clean the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            
            if radius > 10:
                # Update Kalman filter with detected position
                kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))

                # Update trajectory
                trajectory.append((int(x), int(y)))

                # Draw the circle and bounding box
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x - radius), int(y - radius)), 
                              (int(x + radius), int(y + radius)), (255, 0, 0), 2)

                first_detection = True
            else:
                # Predict ball location if detection is too small
                predicted = kalman.predict()
                x, y = predicted[0][0], predicted[1][0]
                cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 0), 2)

        else:
            # Use prediction if ball is not detected in the current frame
            predicted = kalman.predict()
            x, y = predicted[0][0], predicted[1][0]
            cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 0), 2)
            if first_detection:
                trajectory.append((int(x), int(y)))

        # Display the frame
        cv2.imshow('Ball Tracking', frame)
        
        # Exit loop on 'q' press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return trajectory

if __name__ == '__main__':
    video_path = './ball_tracker.mp4'  # Provide your video file path here
    trajectory = track_ball(video_path)
    print("Trajectory of the ball:", trajectory)