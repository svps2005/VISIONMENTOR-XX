import cv2
import mediapipe as mp
import numpy as np

# Init modules
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

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


def run_system():
    cap = cv2.VideoCapture(0)

    prev_x = None
    movement = 0

    with mp_pose.Pose() as pose, \
         mp_face.FaceMesh(refine_landmarks=True) as face_mesh, \
         mp_hands.Hands(max_num_hands=2) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(image)
            face_res = face_mesh.process(image)
            hand_res = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            posture_text = "Detecting..."
            eye_text = "Detecting..."
            gesture_text = "Detecting..."

            # ---------------- POSTURE ----------------
            try:
                lm = pose_res.pose_landmarks.landmark

                l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]

                angle = calculate_angle(
                    [l_ear.x, l_ear.y],
                    [l_sh.x, l_sh.y],
                    [l_hip.x, l_hip.y]
                )

                if angle > 155:
                    posture_text = "Posture: Good"
                    p_color = (0, 255, 0)
                elif angle > 140:
                    posture_text = "Posture: Slight Slouch"
                    p_color = (0, 255, 255)
                else:
                    posture_text = "Posture: Bad"
                    p_color = (0, 0, 255)

                cv2.putText(image, posture_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, p_color, 2)

                mp_drawing.draw_landmarks(
                    image, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except:
                pass

            # ---------------- EYE CONTACT ----------------
            if face_res.multi_face_landmarks:
                for face_landmarks in face_res.multi_face_landmarks:

                    left_corner = face_landmarks.landmark[33]
                    right_corner = face_landmarks.landmark[133]
                    r_left = face_landmarks.landmark[362]
                    r_right = face_landmarks.landmark[263]

                    eye_center = ((left_corner.x + right_corner.x) +
                                  (r_left.x + r_right.x)) / 4

                    nose = face_landmarks.landmark[1].x

                    if abs(eye_center - nose) < 0.03:
                        eye_text = "Eye: Good"
                        e_color = (0, 255, 0)
                    else:
                        eye_text = "Eye: Look at Camera"
                        e_color = (0, 0, 255)

                    cv2.putText(image, eye_text, (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, e_color, 2)

                    mp_drawing.draw_landmarks(
                        image, face_landmarks, mp_face.FACEMESH_TESSELATION)

            # ---------------- GESTURE ----------------
            if hand_res.multi_hand_landmarks:
                for hand_landmarks in hand_res.multi_hand_landmarks:

                    wrist = hand_landmarks.landmark[0]
                    curr_x = wrist.x

                    if prev_x is not None:
                        movement += abs(curr_x - prev_x)

                    prev_x = curr_x

                    if movement > 0.5:
                        gesture_text = "Gesture: Too Much"
                        g_color = (0, 0, 255)
                    else:
                        gesture_text = "Gesture: Stable"
                        g_color = (0, 255, 0)

                    cv2.putText(image, gesture_text, (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show
            cv2.imshow("AI Interview Analyzer", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()