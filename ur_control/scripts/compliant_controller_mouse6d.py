#!/usr/bin/env python
"""
CompliantController + SpaceMouse teleoperation for UR5e.

Verifies the same control path used by UR5ePickCubeGymEnv (hil-serl-sim):
  - CompliantController with FZI cartesian compliance controller
  - OSC_POSITION mode (parallel, high stiffness)
  - SpaceMouse (Mouse6D) for teleoperation
  - Robotiq gripper open/close via buttons
  - Camera images via ImageRecorder (optional)

Usage:
    # Basic test (no cameras):
    rosrun ur_control compliant_controller_mouse6d.py

    # With cameras:
    rosrun ur_control compliant_controller_mouse6d.py --cameras

    # With gripper:
    rosrun ur_control compliant_controller_mouse6d.py --gripper robotiq

    # Full test (cameras + gripper + observation printout):
    rosrun ur_control compliant_controller_mouse6d.py --cameras --gripper robotiq --print-obs
"""
import sys
import argparse
import signal
import time
from pathlib import Path
import getch
import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(suppress=True, precision=4, linewidth=200)

import rospy
from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control.constants import GripperType
from ur_control.mouse_6d import Mouse6D


def signal_handler(sig, frame):
    print("\nCtrl+C pressed, shutting down.")
    rospy.signal_shutdown("User interrupt")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def print_robot_state(arm):
    ee = arm.end_effector(rot_type="quaternion")
    euler = R.from_quat(ee[3:]).as_euler("xyz")
    wrench = arm.get_wrench()
    print(f"  TCP pos:    [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]")
    print(f"  TCP euler:  [{euler[0]:.4f}, {euler[1]:.4f}, {euler[2]:.4f}]")
    print(f"  TCP quat:   [{ee[3]:.4f}, {ee[4]:.4f}, {ee[5]:.4f}, {ee[6]:.4f}]")
    print(f"  Force:      [{wrench[0]:.2f}, {wrench[1]:.2f}, {wrench[2]:.2f}]")
    print(f"  Torque:     [{wrench[3]:.2f}, {wrench[4]:.2f}, {wrench[5]:.2f}]")
    print(f"  Joints:     {np.round(np.degrees(arm.joint_angles()), 2).tolist()}")
    if arm.gripper is not None:
        print(f"  Gripper:    {arm.gripper.get_opening_percentage():.2f} (1=open, 0=closed)")


def print_obs_format(arm, image_recorder=None):
    """Print observation in the same format as UR5ePickCubeGymEnv._get_obs()."""
    ee = arm.end_effector(rot_type="quaternion")
    pos = ee[:3].astype(np.float32)
    euler = R.from_quat(ee[3:]).as_euler("xyz").astype(np.float32)
    tcp_pose = np.concatenate([pos, euler])

    tcp_vel = arm.end_effector_velocity().astype(np.float32)

    wrench = arm.get_wrench()
    tcp_force = wrench[:3].astype(np.float32)
    tcp_torque = wrench[3:].astype(np.float32)

    gripper_pct = arm.gripper.get_opening_percentage() if arm.gripper else 0.0
    gripper_pose = np.array([gripper_pct], dtype=np.float32)

    print(f"  tcp_pose(6):    {tcp_pose}")
    print(f"  tcp_vel(6):     {tcp_vel}")
    print(f"  gripper_pose(1):{gripper_pose}")
    print(f"  tcp_force(3):   {tcp_force}")
    print(f"  tcp_torque(3):  {tcp_torque}")

    if image_recorder is not None:
        imgs = image_recorder.get_images()
        for name, img in imgs.items():
            if img is not None:
                print(f"  {name} image:  shape={img.shape} dtype={img.dtype} mean={img.mean():.1f}")
            else:
                print(f"  {name} image:  None (stale or missing)")


def start_control(args, arm, mouse6d, image_recorder=None):
    rate = rospy.Rate(20)
    delta_x = 0.005
    delta_q = np.deg2rad(1)
    done = False

    print("\nReady! Move the SpaceMouse. Ctrl+C to quit.\n")

    try:
        while not done and not rospy.is_shutdown():
            if mouse6d.twist is None:
                rate.sleep()
                continue

            x = arm.end_effector()
            xd = np.array(mouse6d.twist, dtype=float)

            c = getch.getch()
            if c:
                if c in ['\x1b', '\x03']:
                    done = True
                    rospy.signal_shutdown("Example finished.")

            # Deadzone + fixed step (same as joint_position_mouse6d.py)
            xd[:3] = [delta_x * np.sign(xd[i]) if abs(xd[i]) > 20.0 else 0.0 for i in range(3)]
            xd[3:] = [delta_q * np.sign(xd[3 + i]) if abs(xd[3 + i]) > 2.0 else 0.0 for i in range(3)]
            xd[0], xd[1] = xd[1], xd[0]

            # Gripper via buttons (same as joint_position_mouse6d.py)
            if mouse6d.joy_buttons and mouse6d.joy_buttons[0] == 1:
                if arm.gripper is not None:
                    arm.gripper.close()
            elif mouse6d.joy_buttons and mouse6d.joy_buttons[1] == 1:
                if arm.gripper is not None:
                    arm.gripper.open()

            # Apply delta to current pose
            target_pose = x.copy()
            target_pose[:3] += xd[:3]
            target_pose[3:] = (R.from_euler("xyz", xd[3:]) * R.from_quat(x[3:])).as_quat()

            arm.set_cartesian_target_pose(target_pose.tolist())

            if args.print_obs:
                rospy.loginfo_throttle(2, f"pos={np.round(x[:3], 4)} force={np.round(arm.get_wrench()[:3], 2)}")

            rate.sleep()

    finally:
        print("\nShutting down...")
        try:
            arm.activate_joint_trajectory_controller()
        except Exception:
            pass


def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=arg_fmt, description=__doc__
    )
    parser.add_argument(
        "--gripper", type=str, default=None, choices=["robotiq", "generic"],
        help="Gripper type (default: none)",
    )
    parser.add_argument(
        "--cameras", action="store_true",
        help="Enable camera image recording (front_camera, wrist_camera)",
    )
    parser.add_argument(
        "--camera-names", nargs="+", default=["front_camera", "wrist_camera"],
        help="ROS camera topic prefixes (default: front_camera wrist_camera)",
    )
    parser.add_argument(
        "--print-obs", action="store_true",
        help="Print observations in HIL-SERL format when right button pressed",
    )
    parser.add_argument(
        "--namespace", type=str, default=None,
        help="ROS namespace for the robot",
    )

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("compliant_controller_mouse6d", log_level=rospy.INFO)

    # --- Gripper ---
    if args.gripper == "robotiq":
        gripper_type = GripperType.ROBOTIQ
    elif args.gripper == "generic":
        gripper_type = GripperType.GENERIC
    else:
        gripper_type = None

    # --- Arm (CompliantController) ---
    print("Initializing CompliantController...")
    arm = CompliantController(
        namespace=args.namespace,
        gripper_type=gripper_type,
    )

    if not arm.dashboard_services.activate_ros_control_on_ur():
        rospy.logerr("Failed to activate ROS control on UR. Exiting.")
        sys.exit(1)

    # Configure OSC_POSITION mode (same as hil-serl-sim wrapper.py)
    arm.activate_cartesian_controller()
    arm.set_control_mode("parallel")
    arm.update_selection_matrix(np.ones(6))
    arm.update_stiffness(np.array([2000, 2000, 2000, 500, 500, 500]))
    arm.update_pd_gains(
        [0.05, 0.05, 0.05, 0.20, 0.20, 0.30],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    arm.zero_ft_sensor()

    # Set initial target to current pose
    arm.set_cartesian_target_pose(arm.end_effector().tolist())
    print("CompliantController initialized.")

    # --- SpaceMouse ---
    print("Initializing Mouse6D...")
    mouse6d = Mouse6D()
    print("Mouse6D initialized.")

    # --- Cameras (optional) ---
    image_recorder = None
    if args.cameras:
        print(f"Initializing cameras: {args.camera_names}")
        from osx_gym_env.utils import ImageRecorder
        image_recorder = ImageRecorder(
            init_node=False,
            camera_names=args.camera_names,
        )
        print("Cameras initialized.")

    # --- Print initial state ---
    print("\n--- Initial Robot State ---")
    print_robot_state(arm)
    if args.print_obs:
        print("--- HIL-SERL Observation Format ---")
        print_obs_format(arm, image_recorder)

    # --- Start control loop ---
    start_control(args, arm, mouse6d, image_recorder)
    print("Done.")


if __name__ == "__main__":
    main()
