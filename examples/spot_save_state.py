#!/usr/bin/env python3
"""Undock Spot, reset arm pose, print state for 5 seconds, then save final state to JSON."""

import argparse
import json
import os
import time
from pathlib import Path

from google.protobuf.json_format import MessageToDict

from spot_teleop.spot_controller import SpotRobotController


def main() -> int:
    parser = argparse.ArgumentParser(description="Undock, reset arm, and save Spot state to JSON.")
    parser.add_argument("--hostname", default=os.environ.get("SPOT_ROBOT_IP", "192.168.1.138"),
                        help="Robot IP/hostname (env SPOT_ROBOT_IP).")
    parser.add_argument("--username", default=os.environ.get("BOSDYN_CLIENT_USERNAME", "user"),
                        help="Robot username (env BOSDYN_CLIENT_USERNAME).")
    parser.add_argument("--password", default=os.environ.get("BOSDYN_CLIENT_PASSWORD", "password"),
                        help="Robot password (env BOSDYN_CLIENT_PASSWORD).")
    parser.add_argument("--outfile", default="state_example_new.json", help="Path to write final state JSON.")
    args = parser.parse_args()

    controller = SpotRobotController(args.hostname, args.username, args.password)

    controller.undock()
    time.sleep(3.0)  # wait for the robot to undock
    controller.reset_pose()
    time.sleep(3.0)  # wait for the robot to get to the reset pose
    # Print state for 5 seconds (approx 1 Hz).
    for _ in range(5):
        state = controller.current_state()
        print(state)
        time.sleep(1.0)

    final_state = controller.current_state()
    state_dict = MessageToDict(final_state, preserving_proto_field_name=True)
    out_path = Path(args.outfile)
    out_path.write_text(json.dumps(state_dict, indent=2))
    print(f"Wrote final state to {out_path.resolve()}")

    controller.stow_arm()

    time.sleep(3.0)
    controller.dock()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
