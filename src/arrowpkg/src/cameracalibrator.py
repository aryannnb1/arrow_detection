# #!/usr/bin/env python3

# import cv2
# import numpy as np
# import yaml

# def load_camera_parameters(config_file):
#     with open(config_file, 'r') as file:
#         config = yaml.safe_load(file)

#     return config

# def detect_arrow(image, config):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply GaussianBlur to reduce noise and help contour detection
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Use Canny edge detection
#     edges = cv2.Canny(blurred, 50, 150)

#     # Find contours in the edged image
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Loop over the contours
#     for contour in contours:
#         # Approximate the contour
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # Check if the contour has 7 points (representing an arrow)
#         if len(approx) == 7:
#             cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)  # Draw the arrow contour

#     return image

# def main():
#     config = load_camera_parameters('/home/aryan/ros2_arrow_detection_ws/src/arrowpkg/arrowconfig/cam.yaml')

#     # Open a connection to the camera (assuming video_device is set correctly in camera.yaml)
#     cap = cv2.VideoCapture(config['video_device'])

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         # Perform arrow detection
#         result_frame = detect_arrow(frame, config)

#         # Display the resulting frame
#         cv2.imshow('Arrow Detection', result_frame)

#         # Break the loop if 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the camera and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())


if args["image"]:
    img = cv2.imread(args["image"])
else:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()


while True:
    if not args["image"]:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
    else:
        frame = img.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)


    left = [0, 0]
    right = [0, 0]
    up = [0, 0]
    down = [0, 0]

    if lines is not None:  
        for obj in lines:
            theta = obj[0][1]
            rho = obj[0][0]

         
            if 1.0 <= np.round(theta, 2) <= 1.1 or 2.0 <= np.round(theta, 2) <= 2.1:
                if 20 <= rho <= 30:
                    left[0] += 1
                elif 60 <= rho <= 65:
                    left[1] += 1
                elif -73 <= rho <= -57:
                    right[0] += 1
                elif 148 <= rho <= 176:
                    right[1] += 1

          
            elif 0.4 <= np.round(theta, 2) <= 0.6 or 2.6 <= np.round(theta, 2) <= 2.7:
                if -63 <= rho <= -15:
                    up[0] += 1
                elif 67 <= rho <= 74:
                    down[1] += 1
                    up[1] += 1
                elif 160 <= rho <= 171:
                    down[0] += 1


    if left[0] >= 1 and left[1] >= 1:
        direction = "left"
    elif right[0] >= 1 and right[1] >= 1:
        direction = "right"
    elif up[0] >= 1 and up[1] >= 1:
        direction = "up"
    elif down[0] >= 1 and down[1] >= 1:
        direction = "down"
    else:
        direction = "no arrow detected"


    print(direction)
    print("Up:", up, "Down:", down, "Left:", left, "Right:", right)


    cv2.imshow("Arrow Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if not args["image"]:
    cap.release()
else:
    cv2.destroyAllWindows()