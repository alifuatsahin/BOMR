import cv2
import time
from pyzbar.pyzbar import decode
import numpy as np
from matplotlib import pyplot as plt

def get_qr_data(input_frame):
    try:
        return decode(input_frame)
    except:
        return []

def draw_polygon(frame_in, qrobj):
    points = np.zeros((4,2),np.int32)
    position = np.zeros((1,2),np.int32)
    corners = 0
    detected = False
    if len(qrobj) == 0:
        return frame_in
    else:
        for obj in qrobj:
            text = obj.data.decode('utf-8')
            pts = obj.polygon
            pts = np.array([pts], np.int32)
            if text == 'Robot':
                cv2.polylines(frame_in, [pts], True, (0, 0, 255), 2)
                robot_x_center = obj.rect[2]/2 + obj.rect[0]
                robot_y_center = obj.rect[3]/2 + obj.rect[1]
                position[0] = (int(robot_x_center), int(robot_y_center))
                detected = True
            else:
                cv2.polylines(frame_in, [pts], True, (255, 55, 5), 2)
                x_center = obj.rect[2]/2 + obj.rect[0]
                y_center = obj.rect[3]/2 + obj.rect[1]
                if text == 'Top Left':
                    points[0] = (int(x_center), int(y_center))
                    corners = corners + 1
                elif text == 'Top Right':
                    points[1] = (int(x_center), int(y_center))
                    corners = corners + 1
                elif text == 'Bottom Right':
                    points[2] = (int(x_center), int(y_center))
                    corners = corners + 1
                elif text == 'Bottom Left':
                    points[3] = (int(x_center), int(y_center))
                    corners = corners + 1
            x_org = obj.rect[0]
            y_org = obj.rect[1]
            if text == 'Robot':
                cv2.circle(frame, (int(robot_x_center), int(robot_y_center)), 1, (0, 0, 255), 2)
            else:
                cv2.circle(frame, (int(x_center), int(y_center)), 1, (0, 0, 255), 2)
            cv2.putText(frame_in, text, (x_org, y_org - 20  ), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 200, 1), 2)
        if corners == 4:
            points_arr = np.array([points], np.int32)
            cv2.polylines(frame_in, [points_arr], True, (0, 255, 0), 2)
            if detected:
                for point in points:
                    distance = np.zeros((2,2), np.int32)
                    distance[0] = point
                    distance[1] = position
                    cv2.polylines(frame_in, [distance], True, (0, 255, 0), 2)
            width_AD = np.linalg.norm(points[0] - points[1])
            width_BC = np.linalg.norm(points[3] - points[2])
            max_width = max(int(width_AD), int(width_BC))
            height_AB = np.linalg.norm(points[0] - points[3])
            height_CD = np.linalg.norm(points[2] - points[1])
            max_height = max(int(height_AB), int(height_CD))

            input_pts = np.float32([points[0], points[1], points[2], points[3]])
            output_pts = np.float32([[0,0],[max_width - 1, 0],[max_width - 1, max_height - 1],[0, max_height - 1]])

            M = cv2.getPerspectiveTransform(input_pts, output_pts)
            warped = cv2.warpPerspective(frame_in,M,(max_width, max_height),flags=cv2.INTER_LINEAR)
            cv2.imshow('warp', warped)

        return frame_in

# Connect to webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Set resolution of image
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



# Loop through every frame until we close our webcam
while cap.isOpened():
    ret, frame = cap.read()

    # Grayscale image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    start = time.perf_counter()

    qr_obj = get_qr_data(frame)
    cv2.putText(frame, f'Connected QR: {int(len(qr_obj))}', (30,200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    frame = draw_polygon(frame, qr_obj)

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(frame, f'FPS: {int(fps)}', (30,70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Show image
    cv2.imshow('Webcam', frame)

    # If Q is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
        break

# Release webcam
cap.release()
# Closes the frame
cv2.destroyAllWindows()
