import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def detect_posture():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Camera not detected")
                break

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Back to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # LEFT SIDE
                l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

                left_angle = calculate_angle(
                    [l_ear.x, l_ear.y],
                    [l_shoulder.x, l_shoulder.y],
                    [l_hip.x, l_hip.y]
                )

                # RIGHT SIDE
                r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

                right_angle = calculate_angle(
                    [r_ear.x, r_ear.y],
                    [r_shoulder.x, r_shoulder.y],
                    [r_hip.x, r_hip.y]
                )

                avg_angle = (left_angle + right_angle) / 2

                # Posture classification
                if avg_angle > 165:
                    posture = "Excellent Posture"
                    color = (0, 255, 0)
                elif avg_angle > 150:
                    posture = "Slightly Slouched"
                    color = (0, 255, 255)
                else:
                    posture = "Bad Posture - Sit Straight"
                    color = (0, 0, 255)

                # Display text
                cv2.putText(image, posture, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                cv2.putText(image, f"Angle: {int(avg_angle)}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except:
                pass

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Posture Detection", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_posture()