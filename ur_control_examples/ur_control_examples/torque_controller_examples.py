#! /usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2026 Cristian Beltran
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

"""Basic ur_control motion examples (ROS 2)."""

import argparse
import random
import sys
import threading
import time
import timeit

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ur_control.arm import Arm

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def move_joints(arm):
    q = [1.8391, -1.5659, 1.4889, -1.6421, -1.6115, 0.2656]
    arm.set_joint_positions(positions=q, wait=True, target_time=1.0)



def move_torque(arm):
    arm.activate_joint_effort_controller()
    
    end = time.monotonic() + 0.5
    period = 1.0 / 500.0
    while time.monotonic() < end:
        arm.set_joint_efforts([0.0, 0.0, 0.0, 0.0, 0.0, -1.5])
        time.sleep(period)
    arm.stop_robot()


def move_to_joint_pose(arm):
    q0 = arm.joint_angles()
    q0[-1] += 0.5
    stiffness = np.array([30.0, 30.0, 30.0, 5.0, 5.0, 5.0])  # Nm/rad
    damping = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])       # Nm·s/rad

    arm.activate_joint_effort_controller()
    period = 1.0 / 500
    try:
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            q, qd = arm.joint_angles(), arm.joint_velocities()
            tau = stiffness * (q0 - q) - damping * qd
            arm.set_joint_efforts(tau)
            time.sleep(period)
    finally:
        arm.stop_robot()

def main(args=None):
    parser = argparse.ArgumentParser(description='ur_control motion examples')
    parser.add_argument('-m', '--move', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-t', '--move_torque', action='store_true',
                        help='simply apply torque to a joint')
    parser.add_argument('-q', '--move_q', action='store_true',
                        help='move to joint configuration with torque interface')
    parser.add_argument('--namespace', type=str, default=None)

    argv = remove_ros_args(args if args is not None else sys.argv)
    cli_args = parser.parse_args(argv[1:])

    rclpy.init(args=args)
    node = Node('ur_torque_control_examples')

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    real_start_time = timeit.default_timer()
    ros_start_time = node.get_clock().now().nanoseconds / 1e9

    try:
        gripper = None

        joints_prefix = cli_args.namespace + '_' if cli_args.namespace else None
        arm = Arm(node,
                  namespace=cli_args.namespace,
                  gripper_type=gripper,
                  joint_names_prefix=joints_prefix,
                  use_velocity_interface=True,
                  robot_version='UR5e')

        arm.activate_joint_trajectory_controller()

        if cli_args.move:
            move_joints(arm)
        if cli_args.move_torque:
            move_torque(arm)
        if cli_args.move_q:
            move_to_joint_pose(arm)

        arm.activate_joint_trajectory_controller()
    finally:
        ros_end = node.get_clock().now().nanoseconds / 1e9
        print("real time", round(timeit.default_timer() - real_start_time, 3))
        print("ros time", round(ros_end - ros_start_time, 3))
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
