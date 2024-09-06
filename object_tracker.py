import cv2

# Load video file
cap = cv2.VideoCapture('./ball_tracker.mp4')

# Define the tracker
tracker = cv2.TrackerCSRT_create()

# Read the first frame
ret, frame = cap.read()

# Select the object to track (e.g., the ball)
bbox = cv2.selectROI(frame, False)

# Initialize the tracker
tracker.init(frame, bbox)

# Create a list to store the trajectory
trajectory = []

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    # Draw the bounding box
    if success:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Append the center coordinates to the trajectory
        trajectory.append((x+w//2, y+h//2))

    # Display the frame
    cv2.imshow('Object Tracker', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the trajectory
print(trajectory)






