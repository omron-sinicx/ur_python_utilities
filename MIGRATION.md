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
| `ur_gripper_gz` | `ament_cmake` | ✅ New — gz-sim bringup (UR + Hand-E / 2F-85) + FT + Cartesian compliance |
| `ur_gripper_gz_moveit_config` | `ament_cmake` | ✅ New — custom MoveIt 2 config (arm + gripper group), parameterized for Hand-E / 2F-85 |
| `ur_gripper_description` | catkin | ✅ Removed; superseded for sim by `ur_gripper_gz` + apt `robotiq_description` |
| `ur_gripper_gazebo` | catkin | ✅ Removed; replaced by `ur_gripper_gz` (gz Harmonic) |
| `ur_gripper_85_moveit_config` | catkin | ✅ removed; replaced by ur_gripper_gz_moveit_config |
| `ur_hande_moveit_config` | catkin | ✅ removed; replaced by ur_gripper_gz_moveit_config |

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

**Force-torque sensor (Hand-E bringup):** a gz `force_torque` sensor (`tcp_fts_sensor`) on
`wrist_3_joint` + the `gz-sim-forcetorque-system` plugin (stock `empty.sdf` doesn't load it),
read by the `force_torque_sensor_broadcaster` and **published on `/wrench`** (remapped from the
broadcaster's `~/wrench` via `--controller-ros-args`, matching `ur_control`'s `FT_SUBSCRIBER`).
Upstream `ur_simulation_gz` ships **no** FT sensor, so this is new. Attached to the revolute
`wrist_3_joint` (not the canonical `ft_frame` fixed joint, which bullet-featherstone lumps away);
reported in the `wrist_3_link` frame, so downstream consumers transform via TF.
**Verified in gz (headless):** `/wrench` at 500 Hz, frame `wrist_3_link`, ~13.4 N at the home
pose — consistent with the distal mass (wrist_3 + coupler + Hand-E ≈ 1.4 kg) under gravity.
The **same FT wiring is applied to the 2F-85 bringup** (`ur_gripper_2f85_gz.urdf.xacro` +
`ur_gz_2f85_controllers.yaml` + `ur_2f85_gz_control.launch.py`), verified likewise (~12.6 N).

**Cartesian compliance (Hand-E bringup):** the FZI `cartesian_compliance_controller` is loaded
**inactive** in `ur_gz_controllers.yaml` (`robot_base_link=base_link`, `end_effector_link` /
`compliance_ref_link=gripper_tip_link`, `ft_sensor_ref_link=wrist_3_link`, `command_interfaces:
position`) and switched on by `ur_control`'s `CompliantController`. It conflicts with
`scaled_joint_trajectory_controller` on the position command interfaces, so only one runs at a
time. Its FT input (`~/ft_sensor_wrench`) is remapped at spawn to **`/wrench/filtered`** (the
gravity-compensated topic — see `ft_filter` below); the client publishes `~/target_frame` +
`~/target_wrench`, which already match the controller's own topics.
**Verified in gz (headless), driven through `CompliantController`:** the controller activates,
tracks target poses and target wrenches, stops on a target-force condition, and switches back to
the JTC afterward.

**Cartesian compliance (2F-85 bringup):** same wiring as Hand-E — `cartesian_compliance_controller`
+ `forward_velocity_controller` in `ur_gz_2f85_controllers.yaml`, inactive spawners in
`ur_2f85_gz_control.launch.py` with `~/ft_sensor_wrench` → `/wrench/filtered`. The 2F-85 URDF
adds a `gripper_tip_link` fixed frame (offset from `tool0`) so the controller/client TCP matches
Hand-E. Not yet verified in gz.

**FT filtering + zeroing (`ft_filter`).** The bringup runs `ur_control_examples/ft_filter -t wrench`,
which Butterworth-filters `/wrench` → `/wrench/filtered` and offers `/wrench/filtered/zero_ftsensor`.
`ur_control`'s `Arm` prefers the filtered topic and uses that service for `zero_ft_sensor()`, which
tares out the ~13 N distal-mass gravity bias (post-zero `/wrench/filtered` reads ≈0). The
`cartesian_compliance_controller` also consumes `/wrench/filtered`, so it runs on the zeroed wrench
(the FZI controller takes its measurement frame from the `ft_sensor_ref_link` param, not the message
`frame_id`, so the filter's unstamped output is fine). Zero at a known no-contact pose before use.

### `ur_gripper_gz_moveit_config` (new — custom MoveIt 2 config)

Self-contained MoveIt 2 config for the gz robots, **parameterized by `ur_type` + `gripper`**
(one package serves both). Vendors its own planning configs (`config/kinematics.yaml`,
`joint_limits.yaml`, `ompl_planning.yaml`, `moveit.rviz` — copied from apt `ur_moveit_config`
so they're free to customize) and a `srdf/ur_gripper_gz.srdf.xacro`. `move_group` reads
`/robot_description` from the running gz bringup and executes on its controllers.

- **Arm group** `ur_manipulator` → `scaled_joint_trajectory_controller` (`FollowJointTrajectory`).
- **Gripper group** `gripper` (single actuated joint; mimics follow via the URDF) with `open`/
  `close` states + an `end_effector` on `tool0`, driven by the per-gripper controller mapping:
  Hand-E `gripper_controller` is a JTC → **`FollowJointTrajectory`** (`follow_joint_trajectory`);
  2F-85 `gripper_controller` is a GripperActionController → **`GripperCommand`** (`gripper_cmd`).
  `moveit_controllers_<gripper>.yaml` selects the right one.
- **`load_gripper:=false`** drops the gripper group (arm-only planning); the gripper links stay
  collision-exempt either way.

Gotchas baked into the config:
- **SRDF `disable_collisions` for gripper links are mandatory.** move_group sees the gripper via
  `/robot_description`; without disables the rigidly-attached links read as start-state
  self-collisions and every plan is rejected (`error_code -10`). The apt `ur_moveit_config` alone
  does **not** work with a gripper-equipped description.
- **Gripper joints need acceleration limits in `joint_limits.yaml`** or MoveIt's time-optimal
  parameterization fails (`No acceleration limit ... finger_joint`). Both grippers' joints are listed.
- **2F-85 knuckle position bound widened** (`min_position: -0.02`): gz rests it at ~-7e-13, a hair
  below the URDF lower bound `0.0`, and Jazzy's `CheckStartStateBounds` hard-fails on any
  out-of-bounds revolute start (no tolerance param).

**Verified in gz (headless):** **2F-85** — arm plan+execute (`error_code=1`) **and** gripper
open↔close via a `MoveGroup` goal on the `gripper` group (knuckle 0 ↔ ~0.78, correct
`gripper_cmd` action). **Hand-E** — arm plan+execute and gripper *close* work; the finger's full
reopen is limited by the gz Hand-E finger model (the `gripper_controller` JTC has no goal tolerance,
so it reports success regardless — same gz quirk the `ur_control` client works around with
trajectory mode + a travel cap). MoveIt config itself is correct for both.

*Test it* — two terminals (both `source install/setup.bash` first):
```bash
# terminal 1 — sim (headless)
ros2 launch ur_gripper_gz ur_2f85_gz_control.launch.py gui:=false            # 2F-85 (ur5e)
# terminal 2 — MoveIt + RViz (launch_rviz defaults true)
ros2 launch ur_gripper_gz_moveit_config ur_moveit.launch.py ur_type:=ur5e gripper:=robotiq_2f85
# Hand-E: ur_gz_control.launch.py  +  ur_type:=ur3e gripper:=hande
```
In RViz **MotionPlanning**: for the arm, pick the `ur_manipulator` group, drag the marker, **Plan
& Execute**; for the gripper, pick the `gripper` group and plan to the `open` / `close` named state.
Headless: send a `MoveGroup` goal (`/move_action`, `plan_only:false`) for group `ur_manipulator`
(joint target) or `gripper` (open/close) and confirm `error_code.val == 1` + `/joint_states` moves.

**Gotcha — FT zeroing must not run against a missing service.** `Arm.zero_ft_sensor()` called the
real-robot `ur_hardware_interface/zero_ftsensor` whenever `use_gazebo_sim=false`; in sim that service
never exists and the no-timeout `client.call()` **hung forever** (looked like "compliance example
freezes, never moves, never ends"). Fixed two ways: `__init_ft_sensor__` now binds the FT service
lambdas only when `wait_for_service` succeeds (else logged no-op), and the compliance example declares
`use_gazebo_sim=true`/`use_real_robot=false` by default (override via `--ros-args -p` for real HW).

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

- **Spike 2 — FT + Cartesian compliance in gz: DONE.** Both halves verified in gz (see above):
  `/wrench` live, and `cartesian_compliance_controller` driving force + motion through
  `CompliantController`, switching cleanly against the JTC.
- **2F-85 Cartesian compliance: DONE.,** wiring ported (controllers.yaml + launch spawners + URDF
  `gripper_tip_link`);
- **FT gravity compensation robustness:** the `ft_filter` zeroing tares the distal-mass bias only at
  the pose where you zero it — the bias is pose-dependent, so it reappears as the arm reconfigures.
  A model-based gravity-compensation node (constant tool mass/COM) would hold across the workspace.
- **Robotiq real-hardware driver:** the `robotiq-cri` `robotiq_control` (pymodbus URCAP/RTU/TCP/
  URScript) is deferred (`COLCON_IGNORE`); re-port when real-robot gripper control is needed.
- **MoveIt 2: DONE** — `ur_gripper_gz_moveit_config` (arm + gripper group, both grippers; see
  above). Remaining MoveIt follow-ups: improve gz Hand-E finger fidelity so MoveIt can fully
  open/close it (the 2F-85 works); optionally add pick-and-place / grasp configs; and drop the
  legacy `ur_gripper_85_moveit_config` / `ur_hande_moveit_config` (MoveIt 1, `COLCON_IGNORE`d),
  now superseded.
- **Tidy-ups:** fix the `PDRotation` quaternion-type mismatch; port or remove `simple_controllers`
  and `mouse_6d` (teleop) when their consumers are migrated.
