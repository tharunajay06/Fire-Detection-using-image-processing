import cv2
import numpy as np

def detect_fire(frame):
    """
    Detects fire in the given frame using color detection in the HSV space.
    
    Parameters:
    frame (numpy.ndarray): Input image frame.
    
    Returns:
    bool: True if fire is detected, else False.
    numpy.ndarray: Output image with fire regions highlighted.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for fire-like colors
    lower_bound = np.array([0, 50, 50])  # Lower bound for reddish hues
    upper_bound = np.array([35, 255, 255])  # Upper bound for yellowish hues
    
    # Create a mask for fire-like colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Check if any fire-like region exists
    detected = cv2.countNonZero(mask) > 500  # Threshold for detection
    
    # Highlight the fire region in the original frame
    output_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return detected, output_frame

# Capture video from webcam or a video file
video_source = 0  # Change to video file path if needed
cap = cv2.VideoCapture(video_source)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect fire in the frame
    fire_detected, output_frame = detect_fire(frame)
    
    # Display the result
    if fire_detected:
        cv2.putText(frame, "Fire Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Fire Detection", frame)
    cv2.imshow("Fire Mask", output_frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
