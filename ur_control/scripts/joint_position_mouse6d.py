#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2018-2021 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

"""
UR Joint Position Example: 3Dconnexion mouse

requires ros-$ROS-VERSION-spacenav-node
and to launch roslaunch cartesian_controller_utilities spacenav.launch
#TODO launch automatically
"""
import argparse

import rospy

from ur_control.arm import Arm
from ur_control.constants import GripperType
from ur_control.exceptions import InverseKinematicsException
from ur_control.mouse_6d import Mouse6D
from ur_control import transformations

import numpy as np

np.set_printoptions(suppress=True)

SPACEMOUSE_STALE_TIMEOUT = 0.5
SPACEMOUSE_WAIT_TIMEOUT = 1.0
SPACEMOUSE_WARN_INTERVAL = 5.0


def print_robot_state():
    print("Joint angles:", np.round(arm.joint_angles(), 3).tolist())
    print("End Effector:", np.round(arm.end_effector(rot_type='euler'), 3).tolist())


def start_control():
    rate = rospy.Rate(125)
    target_time = 0.25
    delta_x = 0.01
    delta_q = np.deg2rad(1)

    while not rospy.is_shutdown():
        if mouse6d.twist is None:
            rate.sleep()
            continue

        x = arm.end_effector()
        xd = np.array(mouse6d.twist, dtype=float)

        xd[:3] = [delta_x*np.sign(xd[i]) if abs(xd[i]) > 0.15 else 0.0 for i in range(3)]
        xd[3:] = [delta_q*np.sign(xd[3+i]) if abs(xd[3+i]) > 0.15 else 0.0 for i in range(3)]

        if mouse6d.joy_buttons and mouse6d.joy_buttons[0] == 1:
            print_robot_state()

        if not np.any(xd):
            rate.sleep()
            continue

        pose_delta = xd * target_time
        target_pose = transformations.transform_pose(x, pose_delta, rotated_frame=relative_to_tcp)

        try:
            arm.set_target_pose(pose=target_pose, target_time=target_time)
        except InverseKinematicsException:
            rospy.logdebug("IK solver failed for requested mouse6d pose update")

        rate.sleep()


def main():
    """Joint Position Example: 3D mouse Control

    Use a 3Dconnexion mouse to control end-effector pose.
    """
    epilog = """
Run `roslaunch cartesian_controller_utilities spacenav.launch` before starting this script.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=arg_fmt, description=main.__doc__, epilog=epilog)
    parser.add_argument(
        '--relative', action='store_true', help='Motion Relative to ee')
    parser.add_argument(
        '--namespace', type=str, help='Namespace of arm (useful when having multiple arms)', default=None)
    parser.add_argument(
        '--gripper', type=str, help='gripper type', default=None)
    parser.add_argument(
        '--tcp', type=str, help='Tool Center Point or End-Effector frame for IK without joint prefix', default='tool0'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("joint_position_mouse6d", log_level=rospy.INFO)

    global relative_to_tcp
    relative_to_tcp = args.relative

    tcp_link = args.tcp
    joints_prefix = args.namespace + '_' if args.namespace else None
    if args.gripper == 'robotiq':
        gripper = GripperType.ROBOTIQ
    elif args.gripper == 'generic':
        gripper = GripperType.GENERIC
    else:
        gripper = None

    global arm
    arm = Arm(namespace=args.namespace,
              gripper_type=gripper,
              joint_names_prefix=joints_prefix,
              ee_link=tcp_link)

    arm.dashboard_services.activate_ros_control_on_ur()

    global mouse6d
    mouse6d = Mouse6D()

    start_control()
    print("Done.")


if __name__ == '__main__':
    main()
