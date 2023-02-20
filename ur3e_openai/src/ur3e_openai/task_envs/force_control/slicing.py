
#!/usr/bin/env python

from copy import copy
import rospy
import numpy as np

from ur3e_openai.robot_envs.utils import get_board_color
from ur3e_openai.task_envs.ur3e_force_control import UR3eForceControlEnv
from ur_control import spalg, transformations
from ur_control.constants import FORCE_TORQUE_EXCEEDED, IK_NOT_FOUND
from ur_gazebo.basic_models import get_box_model, get_button_model, get_cucumber_model
from ur_gazebo.model import Model


def get_cl_range(range, curriculum_level):
    return [range[0], range[0] + (range[1] - range[0]) * curriculum_level]


class UR3eSlicingEnv(UR3eForceControlEnv):
    """ Peg in hole with UR3e environment """

    def __init__(self):
        UR3eForceControlEnv.__init__(self)
        self.__load_env_params()

        if not self.real_robot:
            # string_model = get_box_model("cube", self.object_size, color=[0, 1, 0, 0],
            #                              mass=self.object_mass, mu=self.object_friction, kp=self.object_stiffness)
            # string_model = get_cucumber_model(kp=self.object_stiffness, soft_cfm=self.object_soft_cfm, soft_erp=self.object_soft_erp)
            string_model = get_button_model(spring_stiffness=self.object_stiffness, damping=self.object_damping, friction=self.object_friction,
                                            base_mass=1., button_mass=0.1, color=[1, 0, 0, 0], kp=self.object_kp, kd=self.object_kd, max_vel=100.0)
            self.box_model = Model("block", self.object_initial_pose, file_type="string",
                                   string_model=string_model, model_id="target_block", reference_frame="o2ac_ground")

    def __load_env_params(self):
        prefix = "ur3e_gym"

        # Gazebo spawner parameters
        self.randomize_object_properties = rospy.get_param(prefix + "/randomize_object_properties", False)
        self.object_initial_pose = rospy.get_param(prefix + "/object_initial_pose", [])
        self.object_stiffness = rospy.get_param(prefix + "/object_stiffness", 400)
        self.object_damping = rospy.get_param(prefix + "/object_damping", 0.0)
        self.object_kp = rospy.get_param(prefix + "/object_kp", 1e8)
        self.object_kd = rospy.get_param(prefix + "/object_kd", 1)
        self.object_friction = rospy.get_param(prefix + "/object_friction", 1.0)

        self.uncertainty_error = rospy.get_param(prefix + "/uncertainty_error", False)
        self.uncertainty_error_max_range = rospy.get_param(prefix + "/uncertainty_error_max_range", [0, 0, 0, 0, 0, 0])

        self.curriculum_level = rospy.get_param(prefix + "/initial_curriculum_level", 0.1)
        self.max_stiffness_range = rospy.get_param(prefix + "/max_stiffness_range", [5e5, 1e6])
        self.max_damping_range = rospy.get_param(prefix + "/max_damping_range", [0, 2])
        self.max_kp_range = rospy.get_param(prefix + "/max_kp_range", [4e5, 1e6])
        self.max_kd_range = rospy.get_param(prefix + "/max_kd_range", [1, 2])
        self.max_friction_range = rospy.get_param(prefix + "/max_friction_range", [0, 1])

        self.update_target = rospy.get_param(prefix + "/update_target", False)

        # How often to generate a new model, number of episodes
        self.refresh_rate = rospy.get_param(prefix + "/refresh_rate", False)
        self.normal_randomization = rospy.get_param(prefix + "/normal_randomization", True)
        self.basic_randomization = rospy.get_param(prefix + "/basic_randomization", False)
        self.random_type = rospy.get_param(prefix + "/random_type", "uniform")
        self.cl_upgrade_level = rospy.get_param(prefix + "/cl_upgrade_level", 0.8)
        self.cl_downgrade_level = rospy.get_param(prefix + "/cl_downgrade_level", 0.2)
        print(">>>>> ", self.random_type, self.curriculum_learning,
              self.progressive_cl, self.reward_based_on_cl, " <<<<<<")

    def _set_init_pose(self):
        self.controller.stop()
        if not self.real_robot:
            self.set_environment_conditions()

        self.success_counter = 0

        # Move away
        # reset_time = 0.5 if not self.real_robot else 5.0
        # self.ur3e_arm.move_relative([0, 0, 0.03, 0, 0, 0], duration=reset_time, relative_to_tcp=False)

        # Update target pose if needed
        self.update_target_pose()

        # Go to initial pose
        initial_pose = transformations.transform_pose(self.current_target_pose, [-0.05, 0, 0.035, 0, 0, 0], rotated_frame=False)
        self.ur3e_arm.set_target_pose(pose=initial_pose, wait=True, t=self.reset_time)

        self.max_distance = spalg.translation_rotation_error(self.current_target_pose, initial_pose)
        self.max_distance = np.clip(self.max_distance, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [1, 1, 1, np.pi*2, np.pi*2, np.pi*2])

        self.update_scene()
        self.ur3e_arm.zero_ft_sensor()
        self.controller.start()

    def update_scene(self):
        self.stage = 0
        block_pose = self.object_initial_pose  # self.randomize_block_position()
        if self.randomize_object_properties:
            # Randomization type:
            # Uniform within the curriculum level's range
            if not self.curriculum_learning or self.random_type == "uniform":
                randomize_value = self.rng.uniform(low=0.0, high=1.0, size=4)
            # Normal within the curriculum level's range
            elif "normal" in self.random_type:
                mean = self.curriculum_level
                variance = 0.3
                randomize_value = self.rng.normal(loc=mean, scale=variance, size=4)
                randomize_value = np.clip(randomize_value, 0.0, 1.0)
                # Normal within the max range
                if self.random_type == "normal-full":
                    self.kp_range = self.max_kp_range
                    self.kd_range = self.max_kd_range
                    self.stiffness_range = self.max_stiffness_range
                    self.damping_range = self.max_damping_range
                    self.friction_range = self.max_friction_range

            stiffness = np.interp(randomize_value[0], [0., 1.], self.stiffness_range)
            damping = np.interp(randomize_value[1], [0., 1.], self.damping_range)
            friction = np.interp(randomize_value[1], [0., 1.], self.friction_range)
            kp = np.interp(randomize_value[2], [0., 1.], self.kp_range)
            kd = np.interp(randomize_value[3], [0., 1.], self.kd_range)
            color = list(get_board_color(stiffness=stiffness,
                         stiff_lower=self.max_stiffness_range[0], stiff_upper=self.max_stiffness_range[1]))
            color[3] = 0.1
            string_model = get_button_model(spring_stiffness=stiffness, damping=damping, friction=friction,
                                            base_mass=1., button_mass=0.1, color=color, kp=kp, kd=kd, max_vel=100.0)
            self.box_model = Model("block", block_pose, file_type="string",
                                   string_model=string_model, model_id="target_block")
            self.spawner.reset_model(self.box_model)
        else:
            self.box_model.set_pose(block_pose)
            self.spawner.update_model_state(self.box_model)

        self.current_board_pose = transformations.pose_euler_to_quat(block_pose)

    def randomize_block_position(self):
        rand = self.rng.random(size=6)
        rand = np.array([np.interp(rand[i], [0, 1.], self.object_workspace[i]) for i in range(6)])
        rand[3:] = np.deg2rad(rand[3:])
        self.x = rand
        pose = np.copy(self.object_initial_pose)
        pose[:2] += rand[:2]  # only x and y
        return pose

    def set_environment_conditions(self):
        if self.curriculum_learning:
            if self.progressive_cl:
                if np.average(self.episode_hist) >= self.cl_upgrade_level and self.curriculum_level < 1.0:
                    self.curriculum_level += self.curriculum_level_step
                    self.logger.info("CL difficulty UP to %s, ep: %s" %
                                     (round(self.curriculum_level, 2), self.episode_num))
                    self.episode_hist = np.array([0, 1]*10)  # set 50%
                elif np.average(self.episode_hist) <= self.cl_downgrade_level and self.curriculum_level > self.curriculum_level_step:
                    self.curriculum_level -= self.curriculum_level_step
                    self.logger.info("CL difficulty DOWN to %s, ep: %s" %
                                     (round(self.curriculum_level, 2), self.episode_num))
                    self.episode_hist = np.array([0, 1]*10)  # set 50%
                self.curriculum_level = np.clip(self.curriculum_level, 0, 1)
                curriculum_level = self.curriculum_level
                self.logger.info("current CL difficulty: %s, %s, ep: %s" % (round(curriculum_level, 2),
                                 np.round(np.average(self.episode_hist), 2), self.episode_num))
            else:
                max_episodes = 200
                num_eps = self.episode_num + self.cumulative_episode_num  # current training session + previous ones
                curriculum_level = min((num_eps/max_episodes), 1.0)
                self.logger.info(
                    "current CL difficulty: %s, %s, ep: %s" %
                    (round(curriculum_level, 2),
                     np.average(self.episode_hist),
                     self.episode_num))
            self.difficulty_ratio = copy(curriculum_level)
            self.kp_range = get_cl_range(self.max_kp_range, curriculum_level)
            self.kd_range = get_cl_range(self.max_kd_range, curriculum_level)
            self.stiffness_range = get_cl_range(self.max_stiffness_range, curriculum_level)
            self.damping_range = get_cl_range(self.max_damping_range, curriculum_level)
            self.max_friction_range = get_cl_range(self.max_friction_range, curriculum_level)
            self.current_object_workspace = [
                [-max(self.max_object_workspace[i] * curriculum_level, self.min_object_workspace[i]),
                 max(self.max_object_workspace[i] * curriculum_level, self.min_object_workspace[i])] for i in range(6)]
            self.current_object_workspace[2] = [max(self.max_object_workspace[2]
                                                    * curriculum_level, self.min_object_workspace[2]), 0]
            self.position_threshold_cl = 0.005 + (1-curriculum_level) * self.max_position_threshold
        else:
            self.kp_range = self.max_kp_range
            self.kd_range = self.max_kd_range
            self.stiffness_range = self.max_stiffness_range
            self.damping_range = self.max_damping_range
            self.friction_range = self.max_friction_range
            self.current_object_workspace = self.object_workspace
            self.position_threshold_cl = self.position_threshold

    def _is_done(self, observations):
        # Check for excessive contact force
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            self.logger.error("Collision! Aborting")

            self.ur3e_arm.activate_joint_trajectory_controller()
            return True

        pose_error = np.abs(observations[:6]*self.max_distance)

        if self.termination_on_negative_reward:
            if self.cumulated_episode_reward <= self.termination_reward_threshold:
                rospy.loginfo("Fail on reward: %s" % np.round(pose_error[:3], 4))
                self.success_end = False
                self.ur3e_arm.activate_joint_trajectory_controller()
                return True

        # Check how close we are to the target pose (Only y and z, x could be anywhere and its fine)
        position_reached = np.all(pose_error[1:3] < self.position_threshold_cl)

        # If the end effector remains on the target pose for several steps. Then terminate the episode
        if position_reached:
            self.success_counter += 1
        else:
            self.success_counter = 0

        # After 0.1 seconds, terminate the episode
        # if self.success_counter > (0.05 / self.agent_control_dt):
        if self.success_counter > 5:
            self.logger.info("goal reached: %s" % np.round(pose_error[:3], 4))
            self.success_end = True
            if self.real_robot:
                xc = transformations.transform_pose(self.ur3e_arm.end_effector(), [0, 0, 0.013, 0, 0, 0], rotated_frame=True)
                reset_time = 5.0
                self.ur3e_arm.set_target_pose(pose=xc, t=reset_time, wait=True)

            self.ur3e_arm.activate_joint_trajectory_controller()
            return True

        # Check whether we took every available step for this episode
        if self.step_count == self.steps_per_episode-1:
            self.logger.error("Fail!: %s" % np.round(pose_error[:3], 4))

            self.ur3e_arm.activate_joint_trajectory_controller()
            return True

        return False

    def _get_info(self):
        if self.action_result == FORCE_TORQUE_EXCEEDED:
            return "collision"
        return {}

    def _set_action(self, action):
        # print(np.round(self.current_target_pose[:3], 4).tolist())
        self.last_actions = action
        self.action_result = self.controller.act(action, self.current_target_pose, self.action_type)
