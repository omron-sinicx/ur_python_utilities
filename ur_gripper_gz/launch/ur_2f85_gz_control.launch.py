# Gazebo (gz-sim / Harmonic) bringup for UR + Robotiq 2F-85 via gz_ros2_control.
#
#   ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py            # GUI
#   ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py gui:=false # headless
#
# NOTE: forces the bullet-featherstone physics engine — gz Harmonic's default DART engine
# does NOT support mimic constraints, so the 2F-85 four-bar linkage would not articulate.
#
# The gripper exposes gripper_controller/gripper_cmd (GripperCommand), so drive it with
# ur_control.grippers.GripperController(gripper_type="85").

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # gz resolves package:// meshes (ur_description + robotiq_description) from these roots.
    share_root = os.path.dirname(get_package_share_directory("ur_description"))
    gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join([share_root, os.environ.get("GZ_SIM_RESOURCE_PATH", "")]))

    ur_type = LaunchConfiguration("ur_type")
    gui = LaunchConfiguration("gui")
    gripper = LaunchConfiguration("gripper")

    # Single source of truth for the gripper: latch its name on /active_gripper so any
    # ur_control client ('--gripper auto') resolves its config internally (no --params-file).
    active_gripper_pub = ExecuteProcess(
        cmd=["ros2", "topic", "pub", "/active_gripper", "std_msgs/msg/String",
             ["{data: ", gripper, "}"], "--qos-durability", "transient_local"],
        output="screen")

    pkg = FindPackageShare("ur_gripper_gz")
    controllers_file = PathJoinSubstitution([pkg, "config", "ur_gz_2f85_controllers.yaml"])
    xacro_file = PathJoinSubstitution([pkg, "urdf", "ur_gripper_2f85_gz.urdf.xacro"])

    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " name:=", ur_type, " ur_type:=", ur_type,
        " simulation_controllers:=", controllers_file,
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[robot_description, {"use_sim_time": True}])
    clock_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], output="screen")

    # bullet-featherstone is REQUIRED for the gripper mimic joints.
    gz_args = PythonExpression([
        "'--physics-engine gz-physics-bullet-featherstone-plugin -r -v3 empty.sdf' if '", gui,
        "' == 'true' else '--physics-engine gz-physics-bullet-featherstone-plugin -s -r -v3 empty.sdf'"])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": gz_args}.items())

    spawn_entity = Node(package="ros_gz_sim", executable="create", output="screen",
                        arguments=["-topic", "robot_description", "-name", ur_type, "-allow_renaming", "true"])
    spawn_entity_delayed = TimerAction(period=4.0, actions=[spawn_entity])

    def spawner(name, *extra):
        return Node(package="controller_manager", executable="spawner", output="screen",
                    arguments=[name, "-c", "/controller_manager", "--controller-manager-timeout", "120", *extra])

    jsb = spawner("joint_state_broadcaster")
    jtc = spawner("scaled_joint_trajectory_controller")
    gc = spawner("gripper_controller")

    return LaunchDescription([
        gz_resource_path,
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Run the Gazebo GUI (false = headless server)"),
        DeclareLaunchArgument("gripper", default_value="robotiq_2f85",
                              description="Gripper name published on /active_gripper for ur_control clients"),
        robot_state_publisher, clock_bridge, active_gripper_pub, gz_sim, spawn_entity_delayed,
        RegisterEventHandler(OnProcessExit(target_action=spawn_entity, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[jtc, gc])),
    ])
