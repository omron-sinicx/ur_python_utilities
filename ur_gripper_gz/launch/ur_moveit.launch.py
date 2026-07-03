# MoveIt 2 bringup for the ur_gripper_gz gz-sim robots (ARM-ONLY planning).
#
# Run AFTER a gz bringup is up (it reads /robot_description from the running sim and
# executes on its scaled_joint_trajectory_controller):
#   ros2 launch ur_gripper_gz ur_gz_control.launch.py gui:=false        # Hand-E (ur3e)
#   ros2 launch ur_gripper_gz ur_moveit.launch.py ur_type:=ur3e gripper:=hande
#
#   ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py gui:=false   # 2F-85 (ur5e)
#   ros2 launch ur_gripper_gz ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85
#
# Reuses the apt ur_moveit_config planning pipelines / kinematics / controller mapping
# (moveit_controllers.yaml already targets scaled_joint_trajectory_controller). Only the
# SRDF is overridden with ur_gripper_gz's gripper-aware one, so the gripper links present in
# /robot_description don't trip start-state self-collision. The gripper is NOT a planning
# group here (arm-only); it rides along collision-exempt.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    gripper = LaunchConfiguration("gripper").perform(context)
    launch_rviz = LaunchConfiguration("launch_rviz")

    srdf_path = os.path.join(
        get_package_share_directory("ur_gripper_gz"), "srdf", "ur_gripper_gz.srdf.xacro")

    # robot_description comes from the running gz bringup's /robot_description topic (arm +
    # gripper); we only supply the semantic + planning config here.
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(file_path=srdf_path, mappings={"name": ur_type, "gripper": gripper})
        .to_moveit_configs()
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"])
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        condition=IfCondition(launch_rviz),
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
    )

    return [move_group, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur3e",
                              description="UR variant of the running bringup (ur3e for Hand-E, ur5e for 2F-85)"),
        DeclareLaunchArgument("gripper", default_value="hande",
                              description="Gripper collision set: hande | robotiq_2f85"),
        DeclareLaunchArgument("launch_rviz", default_value="true",
                              description="Launch RViz MotionPlanning"),
        OpaqueFunction(function=launch_setup),
    ])
