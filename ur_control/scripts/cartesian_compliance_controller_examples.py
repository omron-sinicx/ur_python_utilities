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
    q = [1.4817, -2.0874, 1.7722, -1.2554, -1.5669, 0.0189]
    arm.set_joint_positions(positions=q, target_time=3, wait=True)


def move_cartesian():
    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(positions=q, target_time=3, wait=True)

    # arm.set_position_control_mode(False)
    # arm.set_control_mode(mode="spring-mass-damper")
    arm.set_position_control_mode(True)
    arm.set_control_mode(mode="parallel")
    arm.set_solver_parameters(error_scale=0.5, iterations=1)
    arm.update_stiffness([1500, 1500, 1500, 100, 100, 100])
    arm.update_stiffness([1500, 1500, 1500, 100, 100, 100])

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

    def f(x):
        rospy.loginfo_throttle(0.25, f"x: {x[:3]}")
        rospy.loginfo_throttle(0.25, f"error: {np.round(trajectory[:3] - x[:3], 4)}")

    arm.zero_ft_sensor()
    res = arm.execute_compliance_control(
        trajectory,
        target_wrench=target_force,
        max_force_torque=[50., 50., 50., 5., 5., 5.],
        duration=30,
        func=f,
        scale_up_error=True,
        max_scale_error=3.0,
        auto_stop=False,
    )
    print("EE total displacement", np.round(ee - arm.end_effector(), 4))
    print("Pose error", np.round(trajectory[:3] - arm.end_effector()[:3], 4))


def recompute_trajectory(R, h, num_waypoints):
    # Compute reference trajectory and reference force profile

    # mortar surface function and derivatives
    def fx(x, y): return 4*x**3 * 11445.39 + y**2 * 2*x * 22890.7 + 2*x * 3.11558
    def fy(x, y): return 4*y**3 * 11445.39 + 2*y * x**2 * 22890.7 + 2*y * 3.11558

    def get_orientation_quaternion_smooth(n, prev_quat=None, prev_R=None):
        if prev_R is None:
            # If no previous rotation
            R = np.zeros((3, 3))
            R[:, 2] = n
            R[:, 0] = np.cross([0, 1, 0], n)
            R[:, 0] /= np.linalg.norm(R[:, 0])
            R[:, 1] = np.cross(n, R[:, 0])
        else:
            # If we have a previous rotation, try to minimize the change
            R = prev_R.copy()
            R[:, 2] = n  # Set the new normal
            R[:, 1] = np.cross(n, R[:, 0])  # Adjust the y-axis
            R[:, 1] /= np.linalg.norm(R[:, 1])
            R[:, 0] = np.cross(R[:, 1], n)  # Adjust the x-axis

        # quat = T.mat2quat(R)
        # quat = [0, 0, 0, 1]
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(R).as_quat()
        print(quat)

        # Ensure consistent quaternion sign
        if prev_quat is not None and np.dot(quat, prev_quat) < 0:
            quat = -quat

        return quat, R

    h = np.clip(h, 0.01, 0.04)  # make sure it's still inside the mortar as height
    # table_height = 0.8
    table_height = 0.0
    initial_pose = [0.0, R, table_height+h, 0.0, 0.9990052, 0.04459406, 0.0]

    # ref_traj = traj_utils.compute_trajectory(
    #     initial_pose,
    #     plane='XY',
    #     radius=R,
    #     radius_direction='-Y',
    #     steps=num_waypoints,
    #     revolutions=1,
    #     from_center=False,
    #     trajectory_type='circular',
    # )
    steps = num_waypoints
    revolutions = 1
    theta_offset = 0
    radius = R
    theta = np.linspace(0, 2*np.pi*revolutions, steps) + theta_offset
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    z = np.ones(steps) * (table_height + h)
    ref_traj_pos = np.array([x, y, z]).T
    # print(ref_traj_pos)

    ind = 0

    # recalculate orientation in every point to be perpendicular to surface;
    ref_traj = np.zeros((steps, 7))
    for point in ref_traj_pos:
        # point to evaluate in
        px, py = point[0], point[1]

        # calculate the normal and tangents vectors to the surface of the mortar
        n = np.array([fx(px, py), fy(px, py), -1])
        normal_vect_direction = n/np.linalg.norm(n)

        try:
            quat_ref, R = get_orientation_quaternion_smooth(normal_vect_direction, quat_ref, R)
        except:
            # first point won't have a rotation matrix to refer to
            quat_ref, R = get_orientation_quaternion_smooth(normal_vect_direction)

        ref_traj[ind, :3] = ref_traj_pos[ind]
        ref_traj[ind, 3:] = quat_ref
        ind += 1

    load_N = np.random.randint(low=1, high=20)  # take a random force reference
    ref_force = np.array([[0, 0, load_N, 0, 0, 0]]*num_waypoints)

    return ref_traj, ref_force


def powder_grounding():
    ref_traj, ref_force = recompute_trajectory(R=0.0085, h=0.004, num_waypoints=100)
    ref_traj += np.array([0, 0.5, 0, 0, 0, 0, 0])
    print(ref_traj)
    # print(ref_force)

    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(positions=q, target_time=3, wait=True)

    # arm.set_position_control_mode(False)
    # arm.set_control_mode(mode="spring-mass-damper")
    arm.set_position_control_mode(True)
    arm.set_control_mode(mode="parallel")
    arm.set_solver_parameters(error_scale=0.5, iterations=1)
    arm.update_stiffness([1500, 1500, 1500, 100, 100, 100])

    # selection_matrix = [0.5, 0.5, 1, 0.5, 0.5, 0.5]
    selection_matrix = np.ones(6)
    # selection_matrix = [1, 1, 0, 1, 1, 1]  # x, y, z, rx, ry, rz
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
    trajectory = np.stack((p1, p2))
    target_force = np.zeros(6)
    # target_force = np.ones(6)

    def R_base2surface(pos=[0., 0., 0.], center=[0., 0., 0.]):
        x, y, z = pos
        a, b, c = center
        x, y, z = x - a, y - b, z - c
        r = np.sqrt(x**2 + y**2 + z**2)
        # print(x, y, z)
        # print(1-(x/r)**2)
        # print((x*y)/r^4*(1-r^2)*(r^2-x^2))
        # print(x/r)
        u_z = 1/r * np.array([x, y, z])
        # print(u_z)
        x_basis = np.array([1, 0, 0])
        u_x = x_basis - np.sum(x_basis*u_z)*u_z
        u_x /= np.linalg.norm(u_x)
        y_basis = np.array([0, 1, 0])
        u_y = y_basis - np.sum(y_basis*u_z)*u_z - np.sum(y_basis*u_x)*u_x
        u_y /= np.linalg.norm(u_y)
        # u_y = y_basis - np.sum(y_basis*u_z)*u_z
        R = np.array([
            u_x, u_y, u_z,
        ]).T
        return R
        # return np.array([
        #     [1-(x/r)**2,  -x*y/r**2 -x*y/r**2,      x/r],
        #     [ -x*y/r**2, 1-(y/r)**2 -x*y/r**2*x*y/r**2, y/r],
        #     [ -x*z/r**2,  -y*z/r**2 -x*y/r**2*x*z/r**2,  z/r],
        # ])

    def f(x):
        rospy.loginfo_throttle(0.25, f"x: {x[:]}")
        rospy.loginfo_throttle(0.25, f"error: {np.round(trajectory[:] - x[:], 4)}")
        R = R_base2surface(pos=x[:3])
        # print(R)
        x, y, z = x[0], x[1], x[2]
        # print(x, y, z)
        px = np.array([
            [0, -z,  y],
            [z,  0, -x],
            [-y, x,  0],

        ])
        T_6x6 = np.block([  # transformation matrix from base to surface
            # [R, px @ R],
            # [np.zeros((3, 3)), R],
            [R, np.zeros((3, 3))],
            [px @ R, R],
        ])
        T_6x6_inv = np.block([  # transformation matrix from surface to base
            # [R.T, (px @ R).T],
            # [np.zeros((3, 3)), R.T],
            [R.T, np.zeros((3, 3))],
            [(px @ R).T, R.T],
        ])
        # print(T_6x6 @ T_6x6_inv)
        # print(np.sum(R[:, 0] * R[:, 2]))
        # print(np.sum(R[:, 1] * R[:, 2]))
        # print(np.sum(R[:, 0] * R[:, 1]))
        selection_matrix_transformed = T_6x6_inv@np.diag(selection_matrix)@T_6x6
        # print("selection_matrix_transformed:")
        # print(selection_matrix_transformed)
        # selection_matrix_transformed_inv = T_6x6_inv@(np.eye(6) - np.diag(selection_matrix))@T_6x6
        # print("selection_matrix_transformed_inv:")
        # print(selection_matrix_transformed_inv)
        # arm.update_selection_matrix(selection_matrix_transformed)

    arm.zero_ft_sensor()
    res = arm.execute_compliance_control(
        # trajectory,
        # target_wrench=target_force,
        ref_traj,
        target_wrench=ref_force[0],
        max_force_torque=[50., 50., 50., 5., 5., 5.],
        duration=30,
        func=f,
        scale_up_error=True,
        max_scale_error=3.0,
        auto_stop=False,
    )
    print("EE total displacement", np.round(ee - arm.end_effector(), 4))
    print("Pose error", np.round(trajectory[:, :3] - arm.end_effector()[:3], 4))


def move_force():
    """ Linear push. Move until the target force is felt and stop. """
    arm.zero_ft_sensor()

    arm.set_control_mode("parallel")
    selection_matrix = [1, 1, 0, 1, 1, 1]
    arm.update_selection_matrix(selection_matrix)

    # arm.set_control_mode("spring-mass-damper")

    arm.set_solver_parameters(error_scale=0.5, iterations=1)
    arm.update_stiffness([1500, 1500, 1500, 100, 100, 100])
    arm.update_stiffness([1500, 1500, 1500, 100, 100, 100])

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
    rospy.sleep(0.5)
    arm.zero_ft_sensor()
    # arm.set_control_mode("parallel")
    arm.set_control_mode("spring-mass-damper")
    # arm.set_hand_frame_control(False)
    # arm.set_end_effector_link("b_bot_tool0")
    # selection_matrix = [0., 0., 0., 1., 1., 1.]
    # selection_matrix = [1., 1., 1., 0., 0., 0.]
    # arm.update_selection_matrix(selection_matrix)
    # 0.8 is vibrating
    arm.set_solver_parameters(error_scale=1.0, iterations=1.0)
    # pd_gains = [0.03, 0.03, 0.03, 1.0, 1.0, 1.0]
    # pd_gains = [0.06, 0.06, 0.06, 1.5, 1.5, 1.5] # stiff
    pd_gains = [0.1, 0.1, 0.1, 2.0, 2.0, 3.0]  # flex
    d_gains = [0.0, 0.0, 0.0, 0, 0, 0]
    # d_gains = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    # d_gains = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]  # good with p0.1 e_s0.8 ite1.0
    # d_gains = [0.0001, 0.0001, 0.0001, 0, 0, 0]
    # pd_gains = [0.08, 0.08, 0.08, 1.5, 1.5, 1.5]
    arm.update_pd_gains(pd_gains, d_gains)
    # arm.update_stiffness([10, 10, 10, 10, 10, 10])
    # arm.update_stiffness([1000, 1000, 1000, 50, 50, 20])
    # arm.update_stiffness([250, 250, 250, 5, 5, 5])  # 10 for pos can make it drift
    arm.update_stiffness([100, 100, 100, 5, 5, 5])  # 10 for pos can make it drift


# mapping parameter space vs vibration
# how to reduce vibration...
# TODO: rotation is wrong!
# add wrench feedback
#

    ee = arm.end_effector()

    target_force = np.zeros(6)
    target_force[1] += 0

    res = arm.execute_compliance_control(ee, target_wrench=target_force,
                                         max_force_torque=[50., 50., 50., 5., 5., 5.], duration=60,
                                         stop_on_target_force=False)
    print(res)
    print("EE change", ee - arm.end_effector())
    rospy.loginfo("STOP FREE DRIVE")


def test():
    # start here
    move_joints()

    arm.move_relative(transformation=[0, 0, -0.03, 0, 0, 0], relative_to_tcp=False, target_time=0.5, wait=True)
    # Move down (cut)
    arm.move_relative(transformation=[0, 0, -0.03, 0, 0, 0], relative_to_tcp=False, target_time=0.5, wait=True)
    for _ in range(3):
        # Move down (cut)
        arm.move_relative(transformation=[0, 0, -0.03, 0, 0, 0], relative_to_tcp=False, target_time=0.5, wait=True)

    # Move back up and to the next initial pose
    arm.move_relative(transformation=[0, 0, 0.03, 0, 0, 0], relative_to_tcp=False, duration=0.25, wait=True)
    arm.move_relative(transformation=[0, 0.01, 0, 0, 0, 0], relative_to_tcp=False, duration=0.25, wait=True)

    arm.set_joint_positions(positions=q, target_time=1, wait=True)
    q = [1.3524, -1.5555, 1.7697, -1.7785, -1.5644, 1.3493]
    arm.set_joint_positions(positions=q, target_time=1, wait=True)

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
    parser.add_argument('-pd', '--powder_grounding', action='store_true',
                        help='powder_grounding')
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

    ns = ""
    joints_prefix = None
    tcp_link = 'gripper_tip_link'
    # tcp_link = 'wrist_3_link'
    # tcp_link = 'tool0'

    if args.namespace:
        ns = args.namespace
        joints_prefix = args.namespace + '_'

    global arm
    arm = CompliantController(namespace=ns,
                              joint_names_prefix=joints_prefix,
                              ee_link=tcp_link,

                              gripper_type=None)

    arm.dashboard_services.activate_ros_control_on_ur()

    if args.move_joints:
        move_joints()

    if args.move_cartesian:
        move_cartesian()
    if args.powder_grounding:
        powder_grounding()
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
