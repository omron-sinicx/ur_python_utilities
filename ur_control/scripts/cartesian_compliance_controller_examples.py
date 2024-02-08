#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2023 Cristian Beltran
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

import sys
import signal
from ur_control import spalg, utils, traj_utils, constants
from ur_control.fzi_cartesian_compliance_controller import CompliantController
import argparse
import rospy
import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def move_joints():
    q = [1.3506, -1.6493, 1.9597, -1.8814, -1.5652, 1.3323]
    q = [1.4414, -1.7303, 2.145, -1.9863, -1.5656, 1.4397]
    q = [1.0497, -1.5863, 2.0175, -2.005, -1.5681, 1.0542]
    arm.set_joint_positions(positions=q, target_time=5, wait=True)


def move_cartesian():
    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(positions=q, target_time=3, wait=True)

    # arm.set_position_control_mode(False)
    # arm.set_control_mode(mode="spring-mass-damper")
    arm.set_position_control_mode(True)
    arm.set_control_mode(mode="parallel")
    arm.set_solver_parameters(error_scale=0.5, iterations=1)
    arm.update_stiffness([1500,1500,1500,100,100,100])

    # selection_matrix = [0.5, 0.5, 1, 0.5, 0.5, 0.5]
    selection_matrix = np.ones(6)
    arm.update_selection_matrix(selection_matrix)

    p_gains = [0.05, 0.05, 0.05, 1.5, 1.5, 1.5]
    d_gains = [0.005, 0.005, 0.005, 0, 0, 0]
    arm.update_pd_gains(p_gains, d_gains=d_gains)

    ee = arm.end_effector()

    p1 = ee.copy()
    p1[2] -= 0.03

    p2 = p1.copy()
    p2[2] += 0.005

    trajectory = p1
    # trajectory = np.stack((p1, p2))
    target_force = np.zeros(6)

    def f(x): return rospy.loginfo_throttle(0.25, f"error: {np.round(trajectory[:3] - x[:3], 4)}")
    arm.zero_ft_sensor()
    res = arm.execute_compliance_control(trajectory, target_wrench=target_force, max_force_torque=[50., 50., 50., 5., 5., 5.],
                                         duration=5, func=f, scale_up_error=True, max_scale_error=3.0, auto_stop=False)
    print("EE total displacement", np.round(ee - arm.end_effector(), 4))
    print("Pose error", np.round(trajectory[:3] - arm.end_effector()[:3], 4))


def move_force():
    """ Linear push. Move until the target force is felt and stop. """
    arm.zero_ft_sensor()

    arm.set_control_mode("parallel")
    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    # arm.set_control_mode("spring-mass-damper")

    arm.set_solver_parameters(error_scale=0.5, iterations=1)
    arm.update_stiffness([1500,1500,1500,100,100,100])

    p_gains = [0.05, 0.05, 0.1, 1.5, 1.5, 1.5]
    d_gains = [0.005, 0.005, 0.005, 0, 0, 0]
    arm.update_pd_gains(p_gains, d_gains)

    ee = arm.end_effector()

    target_force = [0, 0, -5, 0, 0, 0]  # express in the end_effector_link
    # account for the direction of the force?
    stop_at_wrench = np.copy(target_force)
    stop_at_wrench *= -1
    # transform = arm.end_effector(tip_link="b_bot_tool0")
    # tf = spalg.convert_wrench(target_force, transform)
    # print(transform)
    # print("TF", tf[:3])

    res = arm.execute_compliance_control(ee, target_wrench=target_force,
                                         max_force_torque=[30., 30., 30., 4., 4., 4.], duration=15,
                                         stop_at_wrench=stop_at_wrench,
                                         stop_on_target_force=True)
    print(res)
    print("EE change", ee - arm.end_effector())


def slicing():
    """ Push down while oscillating in X-axis or Y-axis """
    arm.zero_ft_sensor()

    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    pd_gains = [0.03, 0.03, 0.03, 1.0, 1.0, 1.0]
    arm.update_pd_gains(pd_gains)

    ee = arm.end_effector()

    trajectory = traj_utils.compute_sinusoidal_trajectory(ee, dimension=1, period=3, amplitude=0.02, num_of_points=100)
    target_force = [0, 0, -3, 0, 0, 0]  # express in the end_effector_link

    res = arm.execute_compliance_control(trajectory, target_wrench=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=20)
    print(res)
    print("EE change", ee - arm.end_effector())


def admittance_control():
    """ Spring-mass-damper force control """
    rospy.loginfo("START ADMITTANCE")

    arm.set_control_mode(mode="spring-mass-damper")

    ee = arm.end_effector()
    target_force = np.zeros(6)
    arm.execute_compliance_control(ee, target_wrench=target_force,
                                   max_force_torque=[50., 50., 50., 5., 5., 5.], duration=10,
                                   stop_on_target_force=False)

    rospy.loginfo("STOP ADMITTANCE")

def free_drive():
    rospy.loginfo("START FREE DRIVE")
    arm.zero_ft_sensor()
    arm.set_control_mode("parallel")
    selection_matrix = [0., 0., 0., 1., 1., 1.]
    arm.update_selection_matrix(selection_matrix)

    pd_gains = [0.03, 0.03, 0.03, 1.0, 1.0, 1.0]
    arm.update_pd_gains(pd_gains)

    ee = arm.end_effector()

    target_force = np.zeros(6)
    target_force[1] += 0

    res = arm.execute_compliance_control(ee, target_wrench=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=15,
                                         stop_on_target_force=False)
    print(res)
    print("EE change", ee - arm.end_effector())
    rospy.loginfo("STOP FREE DRIVE")


def test():
    # start here
    move_joints()

    for _ in range(3):
        # Move down (cut)
        arm.move_relative(transformation=[0, 0, -0.03, 0, 0, 0], relative_to_tcp=False, duration=0.5, wait=True)

        # Move back up and to the next initial pose
        arm.move_relative(transformation=[0, 0, 0.03, 0, 0, 0], relative_to_tcp=False, duration=0.25, wait=True)
        arm.move_relative(transformation=[0, 0.01, 0, 0, 0, 0], relative_to_tcp=False, duration=0.25, wait=True)

def enable_compliance_control():
    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(q, t=1, wait=True)

    arm.zero_ft_sensor()
    
    # arm.set_control_mode(mode="parallel")
    arm.set_control_mode(mode="spring-mass-damper")
    # arm.set_position_control_mode(True)
    # arm.update_selection_matrix(np.array([0.1]*6))

    arm.activate_cartesian_controller()
    arm.set_cartesian_target_pose(arm.end_effector(tip_link="b_bot_gripper_tip_link"))

    for _ in range(30):
        print("current target pose", arm.current_target_pose[:3])
        print("error", arm.current_target_pose[:3] - arm.end_effector(tip_link="b_bot_gripper_tip_link")[:3])
        rospy.sleep(1)

    arm.activate_joint_trajectory_controller()

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move_joints', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-mc', '--move_cartesian', action='store_true',
                        help='move to cartesian configuration')
    parser.add_argument('-mf', '--move_force', action='store_true',
                        help='move towards target force')
    parser.add_argument('-fd', '--free_drive', action='store_true',
                        help='move the robot freely')
    parser.add_argument('-a', '--admittance', action='store_true',
                        help='Spring-mass-damper force control demo')
    parser.add_argument('-s', '--slicing', action='store_true',
                        help='Push down while oscillating on X-axis')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Test')
    parser.add_argument('-teleop', '--teleoperation', action='store_true',
                        help='Enable cartesian controllers for teleoperation')
    parser.add_argument('--namespace', type=str,
                        help='Namespace of arm', default=None)
    args = parser.parse_args()

    rospy.init_node('ur3e_compliance_control')

    ns = "None"
    joints_prefix = None
    tcp_link = 'gripper_tip_link'

    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + '_'

    global arm
    arm = CompliantController(namespace=ns,
                              joint_names_prefix=joints_prefix,
                              ee_link=tcp_link,
                              ft_topic='wrench',
                              gripper_type=None)
    
    arm.dashboard_services.activate_ros_control_on_ur()

    if args.move_joints:
        move_joints()

    if args.move_cartesian:
        move_cartesian()
    if args.move_force:
        move_force()
    if args.admittance:
        admittance_control()
    if args.free_drive:
        free_drive()
    if args.slicing:
        slicing()
    if args.test:
        test()
    if args.teleoperation:
        enable_compliance_control()
    # if args.hand_frame_control:
    #     move_hand_frame_control()


if __name__ == "__main__":
    main()
