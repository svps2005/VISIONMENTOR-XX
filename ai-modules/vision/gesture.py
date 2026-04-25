import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_gesture():
    cap = cv2.VideoCapture(0)

    prev_x = None
    movement = 0

    with mp_hands.Hands(max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    wrist = hand_landmarks.landmark[0]
                    current_x = wrist.x

                    if prev_x is not None:
                        movement += abs(current_x - prev_x)

                    prev_x = current_x

                    if movement > 0.5:
                        status = "Too Much Movement"
                        color = (0, 0, 255)
                    else:
                        status = "Stable Hands"
                        color = (0, 255, 0)

                    cv2.putText(image, status, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.imshow("Gesture Detection", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_gesture()