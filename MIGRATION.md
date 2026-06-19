# `ur_python_utilities` — ROS 1 → ROS 2 (Jazzy) migration

Status of the ROS 2 **Jazzy** / **Gazebo Harmonic (gz-sim 8)** port of this repo.
Part of the wider OSX migration (see `docs/ros2-migration-plan.md` in the parent repo).

Branch: `jazzy`. Build: `colcon build --symlink-install` (two workspaces: `underlay_ws` then `catkin_ws`).

---

## Package status

| Package | Build | State |
|---|---|---|
| `ur_control` | `ament_python` | ✅ Ported & verified (control lib) |
| `ur_pykdl` | `ament_python` | ✅ Ported & verified (FK/IK via PyKDL) |
| `ur_control_examples` | `ament_python` | ✅ Ported (keyboard teleop example) |
| `ur_gripper_gz` | `ament_cmake` | ✅ New — gz-sim bringup (UR + Hand-E / 2F-85) |
| `ur_gripper_description` | catkin | ⏸️ `COLCON_IGNORE` — ROS 1; superseded for sim by `ur_gripper_gz` + apt `robotiq_description` |
| `ur_gripper_gazebo` | catkin | ⏸️ `COLCON_IGNORE` — Classic-Gazebo; replaced by `ur_gripper_gz` (gz Harmonic) |
| `ur_gripper_85_moveit_config` | catkin | ⏸️ `COLCON_IGNORE` — MoveIt 1; regenerate with MSA 2 |
| `ur_hande_moveit_config` | catkin | ⏸️ `COLCON_IGNORE` — MoveIt 1; regenerate with MSA 2 |

---

## What has been done

### `ur_control` (core control library)
Full `rospy` → `rclpy` port. Layout moved to standard `ament_python` (`ur_control/ur_control/…`).

**Architecture — shared `Node` + background executor.** Every class takes an injected
`rclpy.node.Node`; the application spins a `MultiThreadedExecutor` in a background thread.
Synchronous `client.call(...)` and the async-action poll-waits (`FollowJointTrajectory`,
`GripperCommand`) all rely on that background spinning.

Ported modules:
- **Math / helpers:** `transformations`, `spalg`, `math_utils`, `filters`, `conversions`
  (ROS 2 message constructors are keyword-only), `log`, `exceptions`, `constants`, `utils`
  (node-scoped `read_parameter`, monotonic `PID`/`Rate`, `topic_exist`, `solve_namespace`).
- **Controllers:** `controllers_connection` (controller_manager services — `activate/deactivate`,
  state `"active"`, `*Srv.Request()`), `controllers` (`JointTrajectoryController` /
  `JointVelocityController` over `rclpy.action`), `grippers`.
- **Robot driver glue:** `ur_services` (UR dashboard / set_io / payload via service clients).
- **`arm.py`** — the aggregator: wires the shared node + `ControllersConnection` + `URServices`
  + trajectory/velocity controllers + gripper + FT subscriber; FK/IK; controller switching.
- **Compliance / force control:** `hybrid_controller`, `impedance_control`, `admittance`
  (in `hybrid`), `compliance_controller` (RT loop), `fzi_cartesian_compliance_controller`
  + `fzi_utils`. `dynamic_reconfigure` → a single **`SetParameters`** call on the controller
  node; param names matched to `cartesian_controllers` (`stiffness.*`, `pd_gains.<axis>.{p,d}`,
  `solver.*`, `hand_frame_control`, `end_effector_link`).
- **Kinematics:** `eaik_kinematics` (EAIK analytical IK) + KDL (`ur_pykdl`). `constants` controller
  names updated to ROS 2 (`scaled_joint_trajectory_controller`, `forward_velocity_controller`).
- **Grippers:** `GripperController` (generic) supports a `GripperCommand` path **and** a
  `FollowJointTrajectory` ("trajectory") path; `RobotiqGripper` (CModelCommand, real HW).
  `gripper_configs` registry + `apply_gripper_config()` + `read_active_gripper()` (see grippers below).

**Verified:** against the mock UR driver (FK, KDL IK round-trip, JTC move) and in gz-sim
(`arm.py` drives a gz UR5e — "M3 in sim").

Off-path leftovers kept but not on the import path: `simple_controllers` (legacy, superseded by
`controllers`), `mouse_6d` (spacenav teleop). `PDRotation` quaternion-type mismatch is a known
TODO (only bites a Tier-D caller that isn't wired yet).

### `ur_pykdl`
`ament_python`. `URDF.from_parameter_server()` → fetch from the **`/robot_description`** topic
(transient-local), with an XML-prolog strip for `urdf_parser_py`/lxml. FK verified both from the
packaged URDF and from the topic.

### `ur_control_examples`
New `ament_python` package for the ported examples. `joint_position_keyboard` (keyboard teleop)
uses the shared-node model; `--gripper` accepts `auto` (default, reads `/active_gripper`), a
registry name (`robotiq_2f85`, `robotiq_hande`, `none`), or legacy `generic`/`robotiq`.

### `ur_gripper_gz` (new — gz Harmonic bringup)
Self-contained `gz_ros2_control` bringups for UR + gripper:
- **Hand-E:** `ur_gripper_hande_gz.urdf.xacro` + `ur_gz_controllers.yaml` + `ur_gz_control.launch.py`.
  Single actuated `finger_joint`; `hande_right_finger_joint` is a URDF `<mimic>`. Hand-E
  geometry/meshes come from the in-repo `robotiq-cri/robotiq_description`.
- **2F-85:** `ur_gripper_2f85_gz.urdf.xacro` + `ur_gz_2f85_controllers.yaml` +
  `ur_2f85_gz_control.launch.py`. Uses PickNik **`robotiq_description`** geometry
  (`include_ros2_control:=false`) + `ur_to_robotiq` adapter; one actuated knuckle + 5 mimic joints.
  `GripperActionController` (`gripper_cmd`) — matches `ur_control` `gripper_type="85"`.
- **Self-describing gripper:** each launch runs `active_gripper_publisher` (in `ur_control`),
  which latches the gripper name on **`/active_gripper`** (publish-once, transient-local). Clients
  use `--gripper auto` → no `--params-file` juggling.

**Verified in gz (headless):** UR arm JTC move + gripper open/close + all mimic joints articulate,
for both Hand-E and 2F-85, driven through `ur_control`.

---

## Key design notes / gotchas (read before extending)

1. **Shared `Node` + background `MultiThreadedExecutor` is mandatory.** Construct it and start the
   spin thread before building `Arm`/grippers. **Never call `rclpy.spin_once(node)` on a node an
   executor is already spinning** — it corrupts the wait set (this crashed the example until fixed;
   `grippers.py` now polls instead).
2. **gz Harmonic mimic joints require the bullet-featherstone physics engine.** The default DART
   engine silently doesn't support mimic constraints (logs `Physics.cc … does not support mimic
   constraints`), which breaks every mimic gripper. Both gripper launches pass
   `--physics-engine gz-physics-bullet-featherstone-plugin`. Mimic joints get **state interfaces
   only** in `<ros2_control>` (no command interface).
3. **`gz_ros2_control` ignores URDF *command* limits** ("Enforcing command limits is disabled").
   gz physics still enforces the joint *range of motion* (`<limit>`), so cap travel there (and/or
   via the client `gripper_finger_max_position`) to avoid over-close jamming.
4. **`trac_ik_python` has no ROS 2 Jazzy build** (only the C++ lib + MoveIt plugin). `IKSolverType.TRAC_IK`
   transparently falls back to KDL; analytical IK is available via **EAIK** (`eaik_kinematics`).
5. **Controller renames:** `scaled_pos_joint_traj_controller` → `scaled_joint_trajectory_controller`;
   `joint_group_vel_controller` → `forward_velocity_controller`.

---

## Run it

```bash
# UR5e + Robotiq 2F-85 in gz (bullet-featherstone forced by the launch)
ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py            # gui:=false for headless
# UR + Robotiq Hand-E
ros2 launch ur_gripper_gz ur_gz_control.launch.py

# Drive arm + gripper; --gripper auto reads /active_gripper from the bringup
ros2 run ur_control_examples joint_position_keyboard
```

---

## What's next

- **Spike 2, second half — FT + Cartesian compliance in gz:** add a force-torque sensor + the
  `force_torque_sensor_broadcaster` (publish on `/wrench`), load the
  `cartesian_compliance_controller`, and validate `fzi_cartesian_compliance_controller` driving a
  compliant motion in gz. (`SetParameters` path already verified vs. the mock driver.)
- **Robotiq real-hardware driver:** the `robotiq-cri` `robotiq_control` (pymodbus URCAP/RTU/TCP/
  URScript) is deferred (`COLCON_IGNORE`); re-port when real-robot gripper control is needed.
- **MoveIt 2 configs:** regenerate `ur_gripper_85_moveit_config` / `ur_hande_moveit_config` with
  MoveIt Setup Assistant 2 (the catkin ones are `COLCON_IGNORE`d).
- **Decide on `ur_gripper_description` / `ur_gripper_gazebo`:** port to gz Harmonic or drop in favor
  of `ur_gripper_gz` + apt `robotiq_description`.
- **Tidy-ups:** fix the `PDRotation` quaternion-type mismatch; port or remove `simple_controllers`
  and `mouse_6d` (teleop) when their consumers are migrated.
