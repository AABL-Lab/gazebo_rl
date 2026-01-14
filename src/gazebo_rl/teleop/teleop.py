#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
import numpy as np

# Import the UIInterface and the specific UI class you want to use
from ui_interface import UIInterface
from mouse_keyboard_expert import MouseKeyboardExpert
from spacemouse_expert import SpaceMouseExpert  

class RobotControlNode:
    def __init__(self, ui: UIInterface):
        self.ui = ui
        self.pub = rospy.Publisher('joy', Joy, queue_size=10)
        rospy.init_node('robot_control_node', anonymous=True)
        self.rate = rospy.Rate(30)  # 10 Hz

    def run(self):
        while not rospy.is_shutdown():
            action, buttons = self.ui.get_action()
            # Process the action and buttons as needed
            joy_msg = self.process_action(action, buttons)
            # Publish the Joy message
            self.pub.publish(joy_msg)
            self.rate.sleep()

    def process_action(self, action: np.ndarray, buttons: list) -> Joy:
        # Convert action and buttons to Joy message
        joy = Joy()
        # Map the action array to the axes field
        joy.axes = action.tolist()

        joy.buttons = buttons

        return joy

def main():
    # Instantiate the UI object (can be swapped with other UI implementations)
    # ui = MouseKeyboardExpert()
    ui = SpaceMouseExpert()
    robot_control_node = RobotControlNode(ui)
    try:
        robot_control_node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
