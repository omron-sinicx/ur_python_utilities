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
import getch

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
    target_time = 5
    done = False
    while not done and not rospy.is_shutdown():

        c = getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True
                rospy.signal_shutdown("Example finished.")
        xd = np.zeros(6)
        # reset xd to zero 
        # xd[2] = 0.005
        # x = arm.end_effector()
        x= np.array([ 0.08099008, -0.45817897,  0.31013017,  0.00216159,  0.99999719,  0.00072622, -0.00064167])
        target_pose = transformations.transform_pose(x, xd, rotated_frame=relative_to_tcp)
        print(f"Target pose: {target_pose}")
        

        arm.set_target_pose(pose=target_pose, target_time=target_time)


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

    rospy.init_node("simple_move", log_level=rospy.INFO)

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

    start_control()
    print("Done.")


if __name__ == '__main__':
    main()
