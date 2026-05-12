# license included since this code was adapted from cambel UR3 repo

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

import sys  # noqa
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'robosuite_meta/learning'))  # noqa

from import_weights import import_weights
from utils import get_logger
from encoder_utils import encoder_prediction, load_policy
from models import build_encoder_model
import colorsys
from matplotlib import pyplot as plt
import multiprocessing as mp
import rospy
import sys
import cv2
import signal
import argparse
import time
import random
from datetime import datetime

from collections import deque
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch
np.set_printoptions(precision=6, linewidth=2000)
torch.set_printoptions(precision=6, linewidth=2000)
torch.set_float32_matmul_precision('medium')


try:
    from ur_control.exceptions import InverseKinematicsException
    from ur_control.constants import GripperType
    from ur_control import transformations
    from ur_control.mouse_6d import Mouse6D
    from ur_control.arm import Arm
    from ur_control.fzi_cartesian_compliance_controller import CompliantController
except Exception as e:
    pass


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


log = get_logger(__name__)
np.set_printoptions(suppress=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log.info(f'Using device {device}')

axes = "rxyz"

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
POLICY_CKPT_DIR = MODELS_DIR / '85_rigid_fdcc_basic_shapes'
ENCODER_CKPT_DIR = MODELS_DIR / 'encoder_np_rigid_basic_shape_v2/encoder_np_rigid_basic_shape_v2'
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def e2q(e):
    return transformations.quaternion_from_euler(e[0], e[1], e[2], axes=axes)


def log_robot_state(arm):
    log.debug(f'Joint angles: {np.round(np.degrees(arm.joint_angles()), 3)}')
    log.debug(
        f'End Effector: {np.round(arm.end_effector(rot_type="euler"), 3)}')


def xya2df(X, Y, A):
    X, Y, A = [np.array(v) for v in [X, Y, A]]
    df = pd.DataFrame({
        **{f'non_priv-{i}': X[:, i] for i in range(X.shape[1])},
        **{f'priv-{i}': Y[:, i] for i in range(Y.shape[1])},
        **{f'action-{i}': A[:, i] for i in range(A.shape[1])},
    })
    return df


def load_models(cfg):
    # policy
    # policy = load_policy(cfg)
    policy = import_weights(Path(cfg.policy_path).parent.parent)
    log.info(f'Loaded policy: {cfg.policy.model.name} from {cfg.policy_path}')

    # encoder
    encoder = build_encoder_model(
        cfg.encoder, cfg.encoder.non_priv_dim, cfg.encoder.priv_dim, seq_len=cfg.encoder.input_horizon,
    ).to(device)
    encoder.load_state_dict(torch.load(
        cfg.encoder_path, map_location=device, weights_only=True))
    # compile the load to make inference faster
    encoder = torch.compile(encoder)
    encoder.eval()
    log.info(f"Loading encoder weights from {cfg.encoder_path}")

    return policy, encoder


# NOTE: tune this everytimes, should be less than 1cm
COMPENSATE_FOR_HOLE = 0.0  # 0.011 - 0.0068

SPRING_LENGTH = 0.013
GRIP_TO_WRIST = (0.055 + 0.145) + SPRING_LENGTH
PEG_Z_SIZE = 0.075
INSERT_Z_OFFSET = -0.035


def get_obs(cfg, arm, rs_sub, hole_pos_nominal, x=None, q=None):
    x = arm.end_effector() if x is None else x
    q = arm.joint_angles() if q is None else q

    WRIST_POS_OFFSET = GRIP_TO_WRIST - PEG_Z_SIZE - 0.21
    HOLE_POS_OFFSET = - INSERT_Z_OFFSET + COMPENSATE_FOR_HOLE
    wrist_pos_rel = np.array(x[:3])
    wrist_pos_rel = (wrist_pos_rel + np.array([0, 0, WRIST_POS_OFFSET])) - (hole_pos_nominal + np.array([0, 0, HOLE_POS_OFFSET]))
    # print(x[:3], (hole_pos_nominal + np.array([0, 0, HOLE_POS_OFFSET])), WRIST_POS_OFFSET)
    # print(f'{wrist_pos_rel=}')
    scaled_wrist_pos_rel = wrist_pos_rel / cfg.policy.env.obs_pose_scale

    wrist_wrench = arm.get_wrench()
    wrist_force = wrist_wrench[:3]
    scaled_wrist_force = wrist_force / 25 # cfg.policy.env.obs_force_scale
    wrist_torque = wrist_wrench[3:]
    scaled_wrist_torque = wrist_torque / 2.5 # cfg.policy.env.obs_torque_scale

    obs_non_priv = np.hstack((
        scaled_wrist_pos_rel,
        scaled_wrist_force,
        scaled_wrist_torque,
    ))

    if rs_sub:
        # vision data
        rgb = np.array(rs_sub.get_rgb())
        raw_rgb = rgb.copy()
        # depth = np.array(rs_sub.get_depth())
        # raw_depth = depth.copy()

        # rgb = rgb / 255.0
        # depth = (depth - depth.min()) / (depth.max() - depth.min())
    else:
        rgb = None
        raw_rgb = None

    ret = {
        'non_priv': obs_non_priv,
        'rgb': rgb,
        'raw_rgb': raw_rgb,
        'depth': None,
        # 'raw_depth': raw_depth,

    }
    collision = np.linalg.norm(wrist_force) > 50.0 or np.linalg.norm(wrist_torque) > 5.0
    if collision:
        print("Force", np.linalg.norm(wrist_force), 'Torque', np.linalg.norm(wrist_torque))
    return ret, collision


def check_success(x, hole_pos_ref, xy_tol=0.050, z_tol=0.007):
    WRIST_TO_PEG_OFFSET = - GRIP_TO_WRIST - PEG_Z_SIZE
    nominal_peg_pos = x[:3] + np.array([0, 0, WRIST_TO_PEG_OFFSET])
    nominal_hole_pos = hole_pos_ref + np.array([0, 0, COMPENSATE_FOR_HOLE])
    pos_diff = nominal_peg_pos - nominal_hole_pos
    # print(x, nominal_peg_pos, nominal_hole_pos, pos_diff)
    # print("pos diff", pos_diff)
    return np.abs(pos_diff[0]) < xy_tol and np.abs(pos_diff[1]) < xy_tol and np.abs(pos_diff[2]) < z_tol


def start_control(args, arm, mouse6d):
    # POLICY_CKPT_DIR = Path(
    #     'catkin_ws/src/osx_guriguri_bot/models/'
    #     '85_soft_osc_position_basic_shapes'
    # )

    MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

    policy_name = str(POLICY_CKPT_DIR.name).lower()
    is_fdcc = 'fdcc' in policy_name
    controller_type = 'FDCC' if is_fdcc else 'OSC_POSITION'
    
    cfg = OmegaConf.load(str(ENCODER_CKPT_DIR / 'config.yaml'))
    cfg.policy_path = POLICY_CKPT_DIR / 'best_ckpt/best_model.zip'
    cfg.encoder_path = ENCODER_CKPT_DIR / 'ckpt/best.pt'
    cfg.controller_type = controller_type

    if cfg.policy.env.gripper_type == 'Robotiq85GripperSoft':
        # POLICY_ACTION_SCALE = 0.0022 map policy action [-1, 1] to about 0.011 m/s in 20 FPS
        POLICY_ACTION_SCALE = 0.0022
    elif cfg.policy.env.gripper_type == 'Robotiq85Gripper':
        POLICY_ACTION_SCALE = 0.1

    SPEED_ACTION_SCALE = 4.0
    SPACEMOUSE_SENSITIVITY = 0.020

    # HOLE_POS_BASE = np.array([0.0400, 0.6620, 0.130])
    HOLE_POS_BASE = np.array([0.03619, 0.65875, 0.130])

    # HOLE_POS_BASE = np.array([0.0350, 0.6640, 0.130])  # 1mm round (gray)
    CONTROL_FREQ = 20.0
    HORIZON = 200
    SUCCESS_OFFSET = np.array([0., 0., INSERT_Z_OFFSET])
    HOLE_POS_REF = HOLE_POS_BASE + SUCCESS_OFFSET

    Q_HOME = (1.31538856, -1.18991561,  1.24862749, -
              1.62906851, -1.5618518, -0.25763542)
    HOME_POS = HOLE_POS_BASE + np.array([0, 0, 0.05])

    # INITIAL_ORIENTATION = np.array([1, 0, 0, 0])
    INITIAL_ORIENTATION = np.array([0.9974789, -0.0708716, 0.00015349, -0.00360921])
    dt = 1.0 / CONTROL_FREQ
    rate = rospy.Rate(CONTROL_FREQ)

    set_seed(cfg.seed)
    log.info(f'Set seed to {cfg.seed}')

    if args.viz or args.use_seg:
        cfg.policy.env.use_rgb = True
        cfg.policy.env.use_depth = True

    if CONTROL_FREQ != cfg.policy.env.control_freq:
        log.warning(
            f"The control freq is different the config. Expect {cfg.policy.env.control_freq}, got {CONTROL_FREQ}")

    assert Path(cfg.encoder_path).exists(
    ), f"Encoder not found in: {cfg.encoder_path}"
    assert Path(cfg.policy_path).exists(
    ), f"Policy not found in: {cfg.policy_path}"
    # models
    policy, encoder = load_models(cfg)
    # buffer
    non_priv_queue = deque(maxlen=cfg.encoder.input_horizon)

    # Initialize SAM2 process
    args.use_seg = args.use_seg or cfg.policy.env.seg_mode
    if args.use_seg:
        seg_sub = SegSubscriber()

    # # realsense camera
    # rs_sub = RealSenseSubscriber()
    rs_sub = None
    # log.info(f'Subscribe to RealSense data')

    # rerun
    if args.viz:
        rr.init("traj")
        rr.spawn(memory_limit='50%')
        rr.log(
            "raw_camera",
            rr.Pinhole(
                resolution=[640, 480],
                image_from_camera=[[605.71160889,   0., 330.10424805],
                                   [0., 605.06787109, 244.28578186],
                                   [0.,   0.,   1.]],
                camera_xyz=rr.ViewCoordinates.RDF
            )
        )

    X = []
    Y = []
    A = []
    step = 0
    is_begin = True

    try:

        x_des = arm.end_effector().copy()
        arm.zero_ft_sensor()

        prev_time = time.time()
        action = np.zeros(3)

        # Activate cartesian controller (used by CompliantController for both modes)
        arm.activate_cartesian_controller()
            
        log.info("Start Moving Now!")
        while not rospy.is_shutdown():
            if arm.dashboard_services.is_protective_stopped():
                rospy.logerr_throttle(1, "Protective stop triggered")
                rospy.logerr(f"last action {action.tolist()}")
                arm.joint_traj_controller.stop()
                break

            # observation
            x = arm.end_effector()
            q = arm.joint_angles()
            x_des = x.copy()
            obs, collision = get_obs(cfg, arm, rs_sub, hole_pos_nominal=HOLE_POS_REF, x=x, q=q)
            if collision:
                log.info(f'Number of step {step}')
                log.info(f'Collision!!')

                X = []
                Y = []
                A = []
                is_begin = True
                step = 0
                arm.set_cartesian_target_pose(arm.end_effector())
                input("continue?")

            # Process SAM2 asynchronously
            if args.use_seg:
                depth = seg_sub.get_seg_depth()
                # mask = seg_sub.get_mask()

                obs['depth'] = depth
                # obs['mask'] = mask

                # DEBUG: save the depth image
                # nonzero_mask = depth > 0.0
                # hist = np.histogram(depth[nonzero_mask], bins=50)
                # plt.figure(figsize=(10, 5))
                # plt.subplot(1, 2, 1)
                # plt.imshow(depth, cmap='jet', clim=(0, 1))
                # plt.colorbar()
                # plt.subplot(1, 2, 2)
                # plt.bar(hist[1][:-1], hist[0],
                #         width=np.diff(hist[1]), align='edge')
                # plt.xlim(0, 1)
                # plt.tight_layout()
                # plt.savefig('debug.png')
                # break

            # visualize by cv2
            # if args.viz:
            #     display_imgs = [
                # cv2.resize(cv2.cvtColor(
                # obs['raw_rgb'], cv2.COLOR_RGB2BGR), (160, 480)),
                # cv2.cvtColor(
                #     (obs['rgb'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                #     (obs['depth'].repeat(3, axis=-1) * 255).astype(np.uint8),
                # ]
                # display = np.concatenate(display_imgs, axis=0)
                # display = cv2.resize(display, (480, (360 * len(display_imgs))))
                # cv2.imshow('display', display)
                # if cv2.waitKey(1) != -1:
                #     break

            # Convert observations to torch tensors
            for k, v in obs.items():
                if v is not None:
                    obs[k] = torch.from_numpy(v.astype(np.float32))
            non_priv_queue.append(obs['non_priv'])

            if is_begin:
                rospy.loginfo_throttle(1, 'Setup at beginning...')

                if args.pos_var:
                    xy_var = np.random.uniform(-0.02, 0.02, size=2)
                    pos_var = np.array([*xy_var, 0])
                    target_pos = HOME_POS + pos_var
                    rospy.loginfo_throttle(1, f'sample xy var: {xy_var}')
                else:
                    target_pos = HOME_POS.copy()

                # first call takes some time, so preload always at the beginning of the script
                with torch.inference_mode():
                    policy_obs, pred_priv = encoder_prediction(
                        encoder, obs, non_priv_queue, device=device)
                    policy_obs = policy_obs.cpu().numpy()
                    policy.predict(policy_obs, deterministic=True)

                # wait for data to be available
                if (len(non_priv_queue) >= cfg.encoder.input_horizon) \
                        and (not args.use_seg or (args.use_seg and depth.any())):
                    rospy.loginfo_throttle(1, 'Setup done')
                    is_begin = False

            # Rest of the control logic remains the same...
            if mouse6d.joy_buttons[0] == 1 and mouse6d.joy_buttons[1] == 1:
                rospy.loginfo_throttle(1, 'Moving to gripper home')

                EEF_TO_WRIST_OFFSET = np.array(
                    [0, 0, - GRIP_TO_WRIST - PEG_Z_SIZE - COMPENSATE_FOR_HOLE])
                x_pos = x[:3].copy()
                delta = (x_pos + EEF_TO_WRIST_OFFSET) - target_pos

                to_home_error = np.linalg.norm(delta)
                if to_home_error < 1e-4:
                    log.info(f'Moved to home')
                else:
                    log.info(f'To home error: {to_home_error:.6f}')

                    if to_home_error > POLICY_ACTION_SCALE:
                        delta = delta / to_home_error * POLICY_ACTION_SCALE
                    x_des[:3] = x_pos - delta
                    x_des[3:] = INITIAL_ORIENTATION
                    arm.set_cartesian_target_pose(x_des)

            elif mouse6d.joy_buttons[1] == 1:
                arm.zero_ft_sensor()
                rospy.loginfo_throttle(1, 'Zero the FT sensor')
                rospy.loginfo_throttle(1, f'Current q pose {q=}')
                X = []
                Y = []
                A = []
                is_begin = True
                step = 0
                rospy.loginfo_throttle(1, 'Reset states')
            else:
                if mouse6d.joy_buttons[0] == 1:
                    if is_begin:
                        rospy.loginfo_throttle(1, 'Wait for begin')
                    else:
                        is_success = check_success(
                            x, hole_pos_ref=HOLE_POS_REF)
                        if is_success or step >= HORIZON:
                            log.info(f'Number of step {step}')
                            log.info(f'Success: {is_success}')

                            if args.save:
                                df = xya2df(X, Y, A)
                                output_path = Path(
                                    f'trajectories/{datetime.now().strftime("%m%d-%H%M%S")}.csv')
                                output_path.parent.mkdir(exist_ok=True)
                                df.to_csv(output_path, index=False)
                                log.info(f'save trajectory to {output_path}')

                            X = []
                            Y = []
                            A = []
                            is_begin = True
                            step = 0
                            arm.set_cartesian_target_pose(arm.end_effector())
                            input("continue?")

                        # Policy inference
                        with torch.inference_mode():
                            policy_obs, pred_priv = encoder_prediction(encoder, obs, non_priv_queue, device=device)
                            policy_obs = policy_obs.cpu().numpy()
                            # print('policy_obs', policy_obs)
                            action, _ = policy.predict(policy_obs, deterministic=True)
                        print("action", action)
                        action = action[0]

                        # Handle different action dimensions based on controller type
                        if cfg.controller_type == 'FDCC':
                            # For testing, using only position delta (first 3 dimensions)
                            # TODO: implement orientation delta
                            delta_pos = action[:3]
                            x_des[:3] += delta_pos * POLICY_ACTION_SCALE * SPEED_ACTION_SCALE
                        else:
                            # OSC: 3D position action
                            x_des[:3] += action * POLICY_ACTION_SCALE * SPEED_ACTION_SCALE
                        step += 1

                        X.append(obs['non_priv'].cpu().numpy())
                        Y.append(pred_priv.cpu().numpy()[0])
                        A.append(action)

                        if args.viz:
                            rr.log('wrist_pos', rr.Points3D(
                                obs['non_priv'][:3]))
                            rr.log('peg_pos', rr.Points3D(
                                pred_priv.cpu().numpy()[0, :3]))

                            for i, x in enumerate(action):
                                rr.log(f'action/action-{i}', rr.Scalar(x))
                            for i, x in enumerate(obs['non_priv']):
                                rr.log(f'non_priv/non_priv-{i}', rr.Scalar(x))
                            for i, x in enumerate(pred_priv.cpu().numpy()[0]):
                                rr.log(f'priv/priv-{i}', rr.Scalar(x))

                            if 'raw_rgb' in obs:
                                raw_rgb = obs['raw_rgb'].type(torch.uint8)
                                rr.log('raw_camera/raw_rgb',
                                       rr.Image(raw_rgb).compress(jpeg_quality=80))
                            if obs.get('depth') is not None:
                                d = obs['depth']
                                rr.log('obs/depth',
                                       rr.DepthImage(d.cpu().numpy(), meter=1000.0))
                            # if 'raw_depth' in obs:
                            #     d = obs['raw_depth']
                            #     rr.log('obs/raw_depth',
                            #            rr.DepthImage(d.cpu().numpy(), meter=1000.0))
                        # rospy.loginfo(
                        #     f'log time: {time.time() - prof_t0:.4f} s')
                else:
                    # spacemouse control
                    spacenav = np.array(mouse6d.twist)
                    spacenav_normalized = spacenav.copy() / 0.68359375
                    spacenav_normalized[:2] *= -1.0
                    spacenav_normalized[3:5] *= -1.0

                    x_des[:3] += spacenav_normalized[:3] * \
                        SPACEMOUSE_SENSITIVITY
                    x_des[3:] = transformations.rotate_quaternion_by_rpy(
                        *(spacenav_normalized[3:] * np.deg2rad(1.0)), x_des[-4:])

                x_des[3:] = INITIAL_ORIENTATION
                try:
                    arm.set_cartesian_target_pose(x_des)

                except InverseKinematicsException as e:
                    rospy.logwarn_throttle(1, f'Failed to resolve IK. {e}')
                    pass

            rate.sleep()
            now_time = time.time()
            current_fps = np.round(1 / (now_time - prev_time), 4)
            prev_time = now_time
            # rospy.loginfo_throttle(1, f'fps {current_fps}')
            if round(current_fps) < CONTROL_FREQ:
                rospy.logwarn_throttle(
                    1, f'{current_fps=} is lower than expected {CONTROL_FREQ}')

    finally:
        # Return to joint trajectory control for safe shutdown
        try:
            arm.activate_joint_trajectory_controller()
        except AttributeError:
            log.warning("Controller doesn't support joint trajectory activation")
        # Clean up
        cv2.destroyAllWindows()
        if args.use_seg:
            pass
            # sam_process.stop()


def main():
    """3D mouse Control with SAM2 multiprocessing"""
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=arg_fmt, description=main.__doc__)
    parser.add_argument('-seg', '--use-seg', action='store_true')
    parser.add_argument('-pv', '--pos-var', action='store_true')
    parser.add_argument('-viz', '--viz', action='store_true')
    parser.add_argument('-save', '--save', action='store_true')
    parser.add_argument('--policy-ckpt', type=str, help='Path to the policy checkpoint directory')
    parser.add_argument('--encoder-ckpt', type=str, help='Path to the encoder checkpoint directory')
    parser.add_argument('--log_level', '-log', type=str, default='info')

    args = parser.parse_args(rospy.myargv()[1:])

    log.setLevel(args.log_level.upper())

    rospy.init_node("joint_position_keyboard")
    log.info(f'Initialized ros node')

    policy_name = str(POLICY_CKPT_DIR.name).lower()
    is_fdcc = 'fdcc' in policy_name
    controller_type = 'FDCC' if is_fdcc else 'OSC_POSITION'
    log.info(f'Detected controller type: {controller_type} from policy {POLICY_CKPT_DIR.name}')
    
    arm = CompliantController(gripper_type=None)
    
    if controller_type == 'FDCC':
        # FDCC needs lower stiffness for compliant interaction
        # TODO: Tune these values for your specific FDCC setup
        arm.set_control_mode("spring-mass-damper")
        arm.update_stiffness(np.array([500, 500, 500, 100, 100, 100]))
        pd_gains = [0.01, 0.01, 0.01, 0.05, 0.05, 0.05]
        # pd_gains = [1, 1, 1, 1, 1, 1]
        d_gains = [0.0, 0.0, 0.0, 0, 0, 0]
        arm.update_pd_gains(pd_gains, d_gains)
    else:
        # OSC uses higher stiffness for position control
        arm.set_control_mode("parallel")
        arm.update_selection_matrix(np.ones(6))
        arm.update_stiffness(np.array([2000, 2000, 2000, 500, 500, 500]))
        pd_gains = [0.5, 0.5, 0.5, 2.0, 2.0, 3.0]
        d_gains = [0.0, 0.0, 0.0, 0, 0, 0]
        arm.update_pd_gains(pd_gains, d_gains)

    if not arm.dashboard_services.activate_ros_control_on_ur():
        exit(0)
    log.info(f'Initialized arm')

    log.debug(f'Initializing mouse6D')
    mouse6d = Mouse6D()
    log.info(f'Initialized mouse6D')

    start_control(args, arm, mouse6d)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()