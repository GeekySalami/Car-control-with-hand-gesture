#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Float32MultiArray, Bool

class DifferentialDriveCar:
    def __init__(self):
        rospy.init_node('differential_drive_car', anonymous=True)

        # Subscriber for obstacle distance
        self.distance_sub = rospy.Subscriber('/obstacle_distance', Float32, self.distance_callback)

        # Subscriber for hand gestures
        self.gesture_sub = rospy.Subscriber('/hand_data', Float32MultiArray, self.gesture_callback)

        # Subscriber for both hands detected
        self.both_hands_detected_sub = rospy.Subscriber('/both_hands_detected', Bool, self.both_hands_detected_callback)

        # Publisher for controlling the car's movement
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

        # Initialize parameters
        self.obstacle_distance = None
        self.min_safe_distance = 0.65  # Minimum safe distance from obstacles in meters
        self.max_speed = 0.7  # Maximum speed
        self.max_turn_rate = 0.4  # Maximum turn rate (reduced from 0.8 to lower turning sensitivity)

        # Hand gesture parameters
        self.wrist_line_rotation = 0.0
        self.left_hand_distance = 0.0
        self.right_hand_distance = 0.0
        self.both_hands_detected = False

    def distance_callback(self, msg):
        self.obstacle_distance = msg.data

    def gesture_callback(self, msg):
        if len(msg.data) == 3:
            self.wrist_line_rotation = msg.data[0]
            self.left_hand_distance = msg.data[1]
            self.right_hand_distance = msg.data[2]
            self.move()  # Call move() immediately when new data is received

    def both_hands_detected_callback(self, msg):
        self.both_hands_detected = msg.data
        if not self.both_hands_detected:
            self.stop()  # Stop the car if both hands are not detected

    def move(self):
        if not self.both_hands_detected:
            return  # Don't move if both hands are not detected

        # Check for obstacles
        if self.obstacle_distance is not None and self.obstacle_distance < self.min_safe_distance:
            rospy.loginfo("Obstacle detected! Slowing down.")
            speed_factor = 0.01  # Slow down to 1% of normal speed when obstacle is detected
        else:
            speed_factor = 1.0  # Normal speed when no obstacle is detected

        # Differential drive control
        if self.right_hand_distance < 0.470:
            # Accelerate proportionally to right hand closure
            self.twist.linear.x = (0.470 - self.right_hand_distance) / 0.470 * self.max_speed * speed_factor
        elif self.left_hand_distance < 0.470:
            # Decelerate/brake proportionally to left hand closure
            self.twist.linear.x = -((0.470 - self.left_hand_distance) / 0.470 * self.max_speed * speed_factor)
        else:
            # No acceleration or braking
            self.twist.linear.x = 0.0

        # Turning control (with reduced sensitivity)
        self.twist.angular.z = self.wrist_line_rotation * self.max_turn_rate

        rospy.loginfo(f"Speed: {self.twist.linear.x:.2f} m/s, Turn rate: {self.twist.angular.z:.2f} rad/s")

        # Publish the twist message
        self.cmd_vel_pub.publish(self.twist)

    def stop(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("Both hands not detected. Stopping the car.")

    def run(self):
        rospy.spin()  # Keep the node running and waiting for callbacks

if __name__ == '__main__':
    try:
        car = DifferentialDriveCar()
        car.run()
    except rospy.ROSInterruptException:
        pass