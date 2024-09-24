#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, Bool
import mediapipe as mp
import cv2
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def draw_wrist_line(image, left_hand_landmark, right_hand_landmark):
    if left_hand_landmark is not None and right_hand_landmark is not None:
        left_wrist_x = int(left_hand_landmark.x * image.shape[1])
        left_wrist_y = int(left_hand_landmark.y * image.shape[0])
        right_wrist_x = int(right_hand_landmark.x * image.shape[1])
        right_wrist_y = int(right_hand_landmark.y * image.shape[0])

        # Calculate the rotation of the wrist line
        wrist_line_angle = math.atan2(right_wrist_y - left_wrist_y, right_wrist_x - left_wrist_x)
        wrist_line_rotation = wrist_line_angle / (math.pi / 2)

        # Adjust wrist line rotation
        if wrist_line_rotation < -1:
            wrist_line_rotation += 1
        elif wrist_line_rotation > 1:
            wrist_line_rotation -= 1

        cv2.line(image, (left_wrist_x, left_wrist_y), (right_wrist_x, right_wrist_y), (0, 255, 0), 2)
        
        center_x = (left_wrist_x + right_wrist_x) // 2
        center_y = (left_wrist_y + right_wrist_y) // 2

        radius = int(math.sqrt((left_wrist_x - right_wrist_x)**2 + (left_wrist_y - right_wrist_y)**2) / 2)
        cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)

        return wrist_line_rotation
    else:
        return 0

def calculate_average_distance(hand_landmark):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    distances = []
    for finger_tip in finger_tips:
        x1, y1 = hand_landmark.landmark[finger_tip].x, hand_landmark.landmark[finger_tip].y
        x2, y2 = hand_landmark.landmark[0].x, hand_landmark.landmark[0].y
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        distances.append(distance)
    return sum(distances) / len(distances)

def main():
    rospy.init_node('hand_tracking_node', anonymous=True)
    pub = rospy.Publisher('hand_data', Float32MultiArray, queue_size=10)
    hands_detected_pub = rospy.Publisher('hands_detected', Bool, queue_size=10)
    both_hands_detected_pub = rospy.Publisher('both_hands_detected', Bool, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    with mp_hands.Hands(min_detection_confidence=0.42, min_tracking_confidence=0.42) as hands:
        while not rospy.is_shutdown() and cap.isOpened():
            re, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            wrist_line_rotation = 0
            left_hand_distance = 0
            right_hand_distance = 0

            hands_detected = Bool()
            hands_detected.data = False
            both_hands_detected = Bool()
            both_hands_detected.data = False

            if results.multi_hand_landmarks:
                hands_detected.data = True
                if len(results.multi_hand_landmarks) >= 2:
                    both_hands_detected.data = True
                for num, hand in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[num].classification[0].label
                    if handedness == "Right":
                        hand_color = (0, 255, 0)  # Green
                    else:
                        hand_color = (255, 0, 0)  # Blue

                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

                    # Display landmark coordinates
                    for id, lm in enumerate(hand.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.putText(image, str(id), (cx - 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

                    # Calculate average distance for each hand
                    avg_distance = calculate_average_distance(hand)
                    if handedness == "Right":
                        right_hand_distance = avg_distance
                    else:
                        left_hand_distance = avg_distance

                # Calculate and draw the wrist line rotation
                if len(results.multi_hand_landmarks) >= 2:
                    wrist_line_rotation = draw_wrist_line(image, results.multi_hand_landmarks[0].landmark[0], results.multi_hand_landmarks[1].landmark[0])

                # Publish data
                hand_data = Float32MultiArray()
                hand_data.data = [wrist_line_rotation, left_hand_distance, right_hand_distance]
                pub.publish(hand_data)

                rospy.loginfo(f"Published: Rotation: {wrist_line_rotation:.2f}, Left: {left_hand_distance:.2f}, Right: {right_hand_distance:.2f}")

            # Publish whether hands are detected or not
            
            both_hands_detected_pub.publish(both_hands_detected)
            rospy.loginfo(f"Both hands detected: {both_hands_detected.data}")

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass