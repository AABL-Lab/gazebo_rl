import threading
import pyspacemouse
import numpy as np
from typing import Tuple


# pulled from https://github.com/rail-berkeley/serl/blob/main/serl_robot_infra/franka_env/spacemouse/spacemouse_expert.py
# this coder relies on pyspacemouse which requires a linux library and changes to udev rules https://spacemouse.kubaandrysek.cz/#easyhid-is-hidapi-interface-for-python-required-on-all-platforms
class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=pyspacemouse.print_buttons)
        if not success:
            raise Exception("Failed to open SpaceMouse")

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]


if __name__ == "__main__":
    expert = SpaceMouseExpert()
    while True:
        action, buttons = expert.get_action()
        print(action, buttons)