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

import datetime
import sys
import signal
import timeit
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
    # q = [1.4, -2.1, 1.57, -0.85, -1.57, 0]
    # arm.set_joint_positions(positions=q, target_time=3, wait=True)

    ee = [-0.16486, 0.50763, 0.03688, 1.0, 0.0, 0.0, 0.0]
    arm.set_target_pose(pose=ee, target_time=3, wait=True)


def plot_stuff(x_list, x_ref_list, ref_traj, w_list, w_ref_list, center, R_list, time_list, folder_name="test1"):
    # visualization
    import matplotlib.pyplot as plt
    x_list_np = np.array(x_list)
    x_ref_list_np = np.array(x_ref_list)
    fig_traj = plt.figure()
    ax = fig_traj.add_subplot(111)
    plt.axis("equal")
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ls="--")
    ax.plot(x_list_np[:, 0], x_list_np[:, 1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(["Reference trajectory", "Result"])

    w_list_np = np.array(w_list)
    w_ref_list_np = np.array(w_ref_list)
    fig_w = plt.figure()
    ax = fig_w.add_subplot(111)
    ax.plot(-w_list_np[:, :3])
    ax.plot(-w_ref_list_np[:, :3], ls="--")
    colors = ["C0", "C1", "C2", "C0", "C1", "C2"]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])
    ax.legend(["x", "y", "z", "x_tgt", "y_tgt", "z_tgt"])
    ax.set_ylabel("Force [N]")

    # plt.figure()
    fig_3d = plt.figure(figsize=(8, 8))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_zlabel("z", size=14)
    x = x_list_np[:, 0]
    y = x_list_np[:, 1]
    z = x_list_np[:, 2]
    ax.plot(x, y, z, color='tab:orange', marker='.', linestyle='-')
    x_ref = ref_traj[:, 0]
    y_ref = ref_traj[:, 1]
    z_ref = ref_traj[:, 2]
    ax.plot(x_ref, y_ref, z_ref, color='tab:blue', marker='.', linestyle='--')
    ax.scatter(center[0], center[1], center[2], color='tab:red', marker='o')

    # origin
    ax.quiver(center[0], center[1], center[2],
              1.0 / 200, 0.0, 0.0,
              color='r', label='X')
    ax.quiver(center[0], center[1], center[2],
              0.0, 1.0 / 200, 0.0,
              color='g', label='Y')
    ax.quiver(center[0], center[1], center[2],
              0.0, 0.0, 1.0 / 200,
              color='b', label='Z')

    for j in range(len(w_list_np) // 100):
        i = j * 100
        rotation_matrix = R_list[i]

        rot_vec_x = rotation_matrix[:, 0]
        rot_vec_y = rotation_matrix[:, 1]
        rot_vec_z = rotation_matrix[:, 2]

        rot_vec_x /= 200
        rot_vec_y /= 200
        rot_vec_z /= 200

        w_res = w_list_np[i] / 2000
        w_ref = w_ref_list_np[i] / 2000
        ax.quiver(x[i], y[i], z[i],
                  w_res[0], w_res[1], w_res[2],
                  color="tab:orange", label="Z")
        ax.quiver(x[i], y[i], z[i],
                  w_ref[0], w_ref[1], w_ref[2],
                  color="tab:blue", label="Z")

    # Make data
    X = np.arange(center[0] - 0.04, center[0] + 0.04, 0.001)
    Y = np.arange(center[1] - 0.04, center[1] + 0.04, 0.001)
    X, Y = np.meshgrid(X, Y)
    Z = -np.sqrt(np.clip(-(X - center[0])**2 + -(Y -
                 center[1])**2 + 0.04**2, 0, np.inf)) + center[2]
    # X = X + center[0]
    # Y = Y + center[1]
    # Z = Z + center[2]

    # Plot the surface
    from matplotlib import cm
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        alpha=0.25,
    )
    lim = 0.05
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(center[2] - lim, center[2] + lim)

    import os
    os.makedirs("./plot", exist_ok=True)
    os.makedirs(f"./plot/{folder_name}", exist_ok=True)
    fig_traj.savefig(f"./plot/{folder_name}/traj.png")
    fig_w.savefig(f"./plot/{folder_name}/wrench.png")
    fig_3d.savefig(f"./plot/{folder_name}/3d_plot.png")

    # save csv
    # import math
    # repeat_num = math.ceil(x_list_np.shape[0] / ref_traj.shape[0])
    # print("repeat_num:", repeat_num)
    # print(ref_traj.repeat(repeat_num, axis=0).shape)
    time_np = np.array(time_list)
    time_np = time_np.reshape((time_np.shape[0], 1))
    # print(time_np.shape)
    # print(x_list_np.shape)
    # print(x_ref_list_np.shape)
    # print(w_list_np.shape)
    # print(w_ref_list_np.shape)
    # time = np.arange(0, duration, duration / x_list_np.shape[0])
    # time = time.reshape((time.shape[0], 1))

    rows = np.concatenate(
        [
            time_np,
            x_list_np,
            x_ref_list_np,
            w_list_np,
            w_ref_list_np,
        ],
        axis=1,
    )
    # print(rows)
    import csv
    with open(f"./plot/{folder_name}/trajectory.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time"] +
            [f"x[{i}]" for i in range(7)] +
            [f"x_ref[{i}]" for i in range(7)] +
            [f"w[{i}]" for i in range(6)] +
            [f"w_ref[{i}]" for i in range(6)]
        )
        writer.writerows(rows)

    plt.show()


def gripper_frame_method():
    mortar_position = np.array([-0.16484, 0.50799, 0.03673])

    if arm.dashboard_services.use_real_robot:
        # Real robot
        fix_motion_duration = 3
        max_force_torque = [50., 50., 50., 5., 5., 5.]
        env = "real"
        param_scale = 1
    else:
        # Simulation (Gazebo)
        fix_motion_duration = 1
        max_force_torque = [250., 250., 250., 50., 50., 50.]
        env = "sim"
        param_scale = 0.1

    duration = 10
    frequency = 500
    num_waypoints = duration * frequency // 10
    mortar_diameter = 0.08
    desired_height = 0.0025
    fraction = 1.0
    initial_orientation = [0.707,  -0.707, 0.0,  0.0]
    target_force = 10

    reference_trajectory = traj_utils.generate_mortar_trajectory(
        mortar_diameter,
        desired_height,
        num_waypoints,
        initial_orientation,
        fraction,
    )
    # start trajectory at the center of the mortar
    reference_trajectory[:, :3] += mortar_position

    reference_trajectory[:, 2] += 0.00  # offset height if necessary

    reference_force = [[0, 0, -target_force, 0, 0, 0]] * \
        len(reference_trajectory)

    # move to home position
    home_config = [1.6363, -1.4535, 1.8073, -1.9241, -1.5649, -0.0005]
    arm.set_joint_positions(
        positions=home_config,
        target_time=fix_motion_duration,
        wait=True,
    )

    # controller config
    arm.set_position_control_mode(True)
    arm.set_control_mode(mode="parallel")
    arm.set_solver_parameters(error_scale=1.5*param_scale, iterations=3)
    arm.update_stiffness([3000, 3000, 3000, 100, 100, 100])
    p_gains = [0.01, 0.01, 0.01, 1.5, 1.5, 1.5]
    d_gains = [0.001, 0.001, 0.001, 0, 0, 0]
    # d_gains = [0.0, 0.0, 0.0, 0, 0, 0]
    arm.update_pd_gains(p_gains, d_gains=d_gains)

    selection_matrix = [1, 1, 1, 1, 1, 1]  # x, y, z, rx, ry, rz
    arm.update_selection_matrix(selection_matrix)

    # Move to first step in trajectory
    ee = arm.end_effector()
    arm.set_target_pose(
        pose=np.concatenate([np.array([
            reference_trajectory[0, 0],
            reference_trajectory[0, 1],
            reference_trajectory[0, 2] + 0.05
        ]), ee[3:]]),
        target_time=fix_motion_duration,
        wait=True,
    )

    arm.set_target_pose(
        pose=reference_trajectory[0, :],
        target_time=fix_motion_duration,
        wait=True,
    )

    x_list, x_ref_list, w_list, w_ref_list, R_list, time_list = [], [], [], [], [], []

    sphere_center = mortar_position + [0, 0, 0.04]

    def f(x, w, tp, tf):
        # current pose, current wrench, target pose, target force
        time_list.append(rospy.get_time())
        x_list.append(x)
        w_list.append(w)
        R = R_base2surface(pos=tp[:3], center=sphere_center)
        R_list.append(R)
        x_ref_list.append(tp)
        w_ref_list.append(tf)

    input("Press ENTER to start")

    arm.zero_ft_sensor()

    start_time = timeit.default_timer()
    arm.execute_compliance_control(
        reference_trajectory,
        target_wrench=reference_force,
        max_force_torque=max_force_torque,
        duration=duration,
        scale_up_error=False,
        max_scale_error=3.0,
        auto_stop=False,
        func=f,
        mode='TRACKING_ERROR'
    )
    print("compliance control duration "
          f"{timeit.default_timer()-start_time:0.02f}")

    # move to home position
    arm.set_joint_positions(positions=home_config,
                            target_time=fix_motion_duration, wait=True)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    folder_name = f"gripper_frame_{env}_{duration}s_" + \
        f"{target_force}N_{current_datetime}"
    plot_stuff(x_list, x_ref_list, reference_trajectory, w_list,
               w_ref_list, sphere_center, R_list, time_list,
               folder_name=folder_name)


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Test force control')
    parser.add_argument('-m', '--move_joints', action='store_true',
                        help='move to joint configuration')
    parser.add_argument('-mc', '--move_cartesian', action='store_true',
                        help='move to cartesian configuration')
    parser.add_argument('-pg', '--powder_grounding', action='store_true',
                        help='powder_grounding')
    parser.add_argument('-gfm', '--gripper_frame_method', action='store_true',
                        help='Selection matrix in gripper frame')
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
    arm = CompliantController(
        namespace=ns,
        joint_names_prefix=joints_prefix,
        ee_link=tcp_link,
        gripper_type=None,
    )

    if not arm.dashboard_services.activate_ros_control_on_ur():
        exit(0)

    if args.move_joints:
        move_joints()
    if args.gripper_frame_method:
        gripper_frame_method()
    if args.powder_grounding:
        powder_grounding()


if __name__ == "__main__":
    main()
