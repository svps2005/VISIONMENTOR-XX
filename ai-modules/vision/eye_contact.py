import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def detect_eye_contact():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Camera not detected")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # LEFT EYE CORNERS
                    left_corner = face_landmarks.landmark[33]
                    right_corner = face_landmarks.landmark[133]

                    # RIGHT EYE CORNERS (optional refinement)
                    r_left = face_landmarks.landmark[362]
                    r_right = face_landmarks.landmark[263]

                    # Compute eye center (average of both eyes)
                    left_eye_center = (left_corner.x + right_corner.x) / 2
                    right_eye_center = (r_left.x + r_right.x) / 2

                    eye_center_x = (left_eye_center + right_eye_center) / 2

                    # Nose as reference
                    nose_x = face_landmarks.landmark[1].x

                    # Decision threshold
                    if abs(eye_center_x - nose_x) < 0.03:
                        status = "Good Eye Contact"
                        color = (0, 255, 0)
                    else:
                        status = "Maintain Eye Contact"
                        color = (0, 0, 255)

                    # Display text
                    cv2.putText(image, status, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION
                    )

            cv2.imshow("Eye Contact Detection", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_eye_contact()