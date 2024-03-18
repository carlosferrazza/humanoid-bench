from gymnasium.spaces import Box, Dict
import numpy as np
import mujoco

from humanoid_bench.mjx.flax_to_torch import TorchModel, TorchPolicy
from humanoid_bench.tasks import Task


def get_body_idxs(model):
    # Filter out hand and wrist joints
    body_idxs = []
    body_vel_idxs = []
    curr_idx = 0
    curr_vel_idx = 0
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name.startswith("free_"):
            if joint_name == "free_base":
                body_idxs.extend(list(range(curr_idx, curr_idx + 7)))
                body_vel_idxs.extend(list(range(curr_vel_idx, curr_vel_idx + 6)))
            curr_idx += 7
            curr_vel_idx += 6
            continue
        elif (
            not joint_name.startswith("lh_")
            and not joint_name.startswith("rh_")
            and not "wrist" in joint_name
        ):
            body_idxs.append(curr_idx)
            body_vel_idxs.append(curr_vel_idx)
        curr_idx += 1
        curr_vel_idx += 1

    return body_idxs, body_vel_idxs


class BaseWrapper(Task):
    def __init__(self, task):
        self.task = task
        self.dof = task.dof
        if not hasattr(task._env, "viewer"):
            task._env.viewer = task._env.mujoco_renderer._get_viewer(
                task._env.render_mode
            )

    def step(self, action):
        return self.task.step(action)

    def get_obs(self):
        return self.task.get_obs()

    def get_tactile_obs(self):
        return self.task.get_tactile_obs()

    def get_camera_obs(self):
        return self.task.get_camera_obs()

    def get_reward(self):
        return self.task.get_reward()

    def get_terminated(self):
        return self.task.get_terminated()

    def reset_model(self):
        self.task.reset_model()
        return self.get_obs()

    def normalize_action(self, action):
        return self.task.normalize_action(action)

    def unnormalize_action(self, action):
        return self.task.unnormalize_action(action)

    def render(self):
        return self.task.render()


class SingleReachWrapper(BaseWrapper):
    def __init__(self, task, policy_path, mean_path, var_path, max_delta=0.1, **kwargs):
        super().__init__(task)

        self.policy_path = policy_path
        self.mean_path = mean_path
        self.var_path = var_path

        target_low = task.htarget_low
        target_high = task.htarget_high

        assert target_low.shape == target_high.shape == (3,)
        task._env.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.target_low = target_low
        self.target_high = target_high

        self.last_target = np.zeros(3)

        self.max_delta = max_delta

        reaching_model = TorchModel(55, 19)
        self.reaching_policy = TorchPolicy(reaching_model)
        self.reaching_policy.load(policy_path, mean=mean_path, var=var_path)

        self.body_idxs, self.body_vel_idxs = get_body_idxs(task._env.model)

        print("Body Idxs: ", self.body_idxs)
        print("Body Vel Idxs: ", self.body_vel_idxs)

        if task._env.model.nu > 19:
            self.act_idxs = list(range(15)) + list(range(16, 20))
        else:
            self.act_idxs = list(range(19))

    def get_last_target(self):
        return self.last_target

    def unnormalize_target(self, target):
        return target * self.max_delta

    def unnormalize_body_action(self, action):
        return (action + 1) / 2 * (
            self.task._env.action_high[self.act_idxs]
            - self.task._env.action_low[self.act_idxs]
        ) + self.task._env.action_low[self.act_idxs]

    def get_reach_obs(self):
        position = self.task._env.data.qpos.flat.copy()[self.body_idxs]
        velocity = self.task._env.data.qvel.flat.copy()[self.body_vel_idxs]
        left_hand = self.task.robot.left_hand_position().copy()
        target = self.last_target.copy()

        offset = np.array([position[0], position[1], 0])
        # offset = np.array([position[0], position[1], (self.task.robot.left_foot_height() + self.task.robot.right_foot_height())/2])
        position[:3] -= offset
        left_hand -= offset
        target -= offset

        return np.concatenate((position[2:], velocity, left_hand, target))

    def step(self, action):
        target = self.unnormalize_target(action) + self.last_target
        target = np.clip(target, self.target_low, self.target_high)
        self.last_target = target

        reach_obs = self.get_reach_obs()
        action = self.reaching_policy(reach_obs)
        action = np.clip(action, -1, 1)

        if (
            self.task._env.model.nu > 19
        ):  # only needed because reaching policy is trained without hands
            action = self.unnormalize_body_action(action)

            action_new = self.task._env.data.ctrl.copy()
            action_new[self.act_idxs] = action
            action_new[15] = 1.57
            action_new[20] = 1.57
            action = action_new

            action = self.normalize_action(action)

        return self.task.step(action)

    def render(self):
        found = False
        for i in range(len(self.task._env.viewer._markers)):
            if self.task._env.viewer._markers[i]["objid"] == 790:
                self.task._env.viewer._markers[i]["pos"] = self.last_target
                found = True
                break
        if not found:
            self.task._env.viewer.add_marker(
                pos=self.last_target,
                size=0.05,
                objid=790,
                rgba=(0.28, 0.32, 0.39, 0.8),
                label="",
            )

        return self.task.render()


class DoubleReachBaseWrapper(BaseWrapper):
    # action is [left_hand_target, phi, costheta, u], where (phi, costheta, u) are the spherical coordinates of the right hand target wrt the left hand target

    mode_right = None

    def __init__(
        self,
        task,
        policy_path,
        mean_path,
        var_path,
        max_delta=0.1,
        max_delta_coords=0.1,
        **kwargs
    ):
        super().__init__(task)

        self.policy_path = policy_path
        self.mean_path = mean_path
        self.var_path = var_path

        target_low = task.htarget_low
        target_high = task.htarget_high

        assert target_low.shape == target_high.shape == (3,)  # for left hand
        task._env.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.target_low = target_low
        self.target_high = target_high

        self.last_target_left = np.zeros(3)
        self.last_target_right = np.zeros(3)
        self.last_coords = np.zeros(3)

        self.max_delta = max_delta
        self.max_delta_coords = max_delta_coords

        reaching_model = TorchModel(61, 19)
        self.reaching_policy = TorchPolicy(reaching_model)
        self.reaching_policy.load(policy_path, mean=mean_path, var=var_path)

        self.body_idxs, self.body_vel_idxs = get_body_idxs(task._env.model)

        print("Body Idxs: ", self.body_idxs)
        print("Body Vel Idxs: ", self.body_vel_idxs)

        if task._env.model.nu > 19:
            self.act_idxs = list(range(15)) + list(range(16, 20))
        else:
            self.act_idxs = list(range(19))

    def unnormalize_target(self, target):
        return target * self.max_delta

    def unnormalize_body_action(self, action):
        return (action + 1) / 2 * (
            self.task._env.action_high[self.act_idxs]
            - self.task._env.action_low[self.act_idxs]
        ) + self.task._env.action_low[self.act_idxs]

    def _sample_from_sphere(self, center, radius, coords):
        phi, costheta, u = coords

        theta = np.arccos(costheta)
        r = radius * np.cbrt(u)

        x = r * np.sin(theta) * np.cos(phi) + center[0]
        y = r * np.sin(theta) * np.sin(phi) + center[1]
        z = r * np.cos(theta) + center[2]

        return np.array([x, y, z])

    def get_last_target(self):
        return self.last_target_left, self.last_target_right

    def get_reach_obs(self):
        position = self.task._env.data.qpos.flat.copy()[self.body_idxs]
        velocity = self.task._env.data.qvel.flat.copy()[self.body_vel_idxs]
        left_hand = self.task.robot.left_hand_position().copy()
        right_hand = self.task.robot.right_hand_position().copy()
        target_left = self.last_target_left.copy()
        target_right = self.last_target_right.copy()

        offset = np.array([position[0], position[1], 0])
        # offset = np.array([position[0], position[1], (self.task.robot.left_foot_height() + self.task.robot.right_foot_height())/2])
        position[:3] -= offset
        left_hand -= offset
        right_hand -= offset
        target_left -= offset
        target_right -= offset

        return np.concatenate(
            (position[2:], velocity, left_hand, right_hand, target_left, target_right)
        )

    def step(self, action):
        action_left = action[:3]
        target_left = self.unnormalize_target(action_left) + self.last_target_left
        if self.mode_right == "absolute":
            action_right = action[3:]
            target_right = (
                self.unnormalize_target(action_right) + self.last_target_right
            )
        elif self.mode_right == "relative":
            coords = action[3:]
            coords *= self.max_delta_coords
            coords += self.last_coords
            coords = np.clip(coords, -1, 1)
            coords[0] = (coords[0] + 1) * np.pi
            coords[2] = (coords[2] + 1) / 2
            target_right = self._sample_from_sphere(self.last_target_left, 1, coords)
            self.last_coords = coords
        else:
            raise ValueError("Invalid mode for right hand target")

        target_left = np.clip(target_left, self.target_low, self.target_high)
        target_right = np.clip(target_right, self.target_low, self.target_high)
        self.last_target_left = target_left
        self.last_target_right = target_right

        reach_obs = self.get_reach_obs()

        action = self.reaching_policy(reach_obs)
        action = np.clip(action, -1, 1)

        if (
            self.task._env.model.nu > 19
        ):  # only needed because reaching policy is trained without hands
            action = self.unnormalize_body_action(action)

            action_new = self.task._env.data.ctrl.copy()
            action_new[self.act_idxs] = action
            action_new[15] = 1.57
            action_new[20] = 1.57
            action = action_new

            action = self.normalize_action(action)

        return self.task.step(action)

    def render(self):
        found = False
        for i in range(len(self.task._env.viewer._markers)):
            if self.task._env.viewer._markers[i]["objid"] == 791:
                self.task._env.viewer._markers[i]["pos"] = self.last_target_left
                found = True
                break
        if not found:
            self.task._env.viewer.add_marker(
                pos=self.last_target_left,
                size=0.05,
                objid=791,
                rgba=(0.28, 0.32, 0.39, 0.8),
                label="",
            )

        found = False
        for i in range(len(self.task._env.viewer._markers)):
            if self.task._env.viewer._markers[i]["objid"] == 792:
                self.task._env.viewer._markers[i]["pos"] = self.last_target_right
                found = True
                break
        if not found:
            self.task._env.viewer.add_marker(
                pos=self.last_target_right,
                size=0.05,
                objid=792,
                rgba=(0.39, 0.28, 0.32, 0.8),
                label="",
            )

        return self.task.render()


class DoubleReachAbsoluteWrapper(DoubleReachBaseWrapper):
    mode_right = "absolute"


class DoubleReachRelativeWrapper(DoubleReachBaseWrapper):
    mode_right = "relative"


class BlockedHandsLocoWrapper(BaseWrapper):
    def __init__(self, task, **kwargs):
        super().__init__(task)

        if kwargs["small_obs"] is not None:
            self.small_obs = kwargs.get("small_obs", "False").lower() == "true"
        else:
            self.small_obs = False
        print("Small obs: ", self.small_obs)

        assert (
            task._env.model.nu > 19
        ), "BlockedHandsWrapper only works when hands are present in the action space"

        task._env.action_space = Box(low=-1, high=1, shape=(19,), dtype=np.float32)

        self.body_idxs, self.body_vel_idxs = get_body_idxs(task._env.model)

        print("Body Idxs: ", self.body_idxs)
        print("Body Vel Idxs: ", self.body_vel_idxs)

        self.act_idxs = list(range(15)) + list(range(16, 20))

        if self.small_obs:
            task._env.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.body_idxs) + len(self.body_vel_idxs),),
                dtype=np.float64,
            )

    def unnormalize_body_action(self, action):
        return (action + 1) / 2 * (
            self.task._env.action_high[self.act_idxs]
            - self.task._env.action_low[self.act_idxs]
        ) + self.task._env.action_low[self.act_idxs]

    def get_obs(self):
        """Small obs only works when there are no other observations except for the robot state (e.g., locomotion). Subclass this class to adapt to other tasks."""
        if self.small_obs:
            position = self.task._env.data.qpos.flat.copy()[self.body_idxs]
            velocity = self.task._env.data.qvel.flat.copy()[self.body_vel_idxs]
            return np.concatenate((position, velocity))
        else:
            return super().get_obs()

    def step(self, action):
        action = self.unnormalize_body_action(action)

        action_new = self.task._env.data.ctrl.copy()
        action_new[self.act_idxs] = action
        action_new[15] = 1.57
        action_new[20] = 1.57
        action = action_new

        self.task._env.do_simulation(action, self.task._env.frame_skip)

        obs = self.get_obs()
        reward, reward_info = self.task.get_reward()
        terminated, terminated_info = self.task.get_terminated()

        info = {"per_timestep_reward": reward, **reward_info, **terminated_info}
        return obs, reward, terminated, False, info


class ObservationWrapper(BaseWrapper):
    def __init__(self, task, **kwargs):
        super().__init__(task)

        sensors = kwargs.get("sensors").split(",")
        self._tactile_ob = "tactile" in sensors
        self._camera_ob = "image" in sensors

        if self._tactile_ob:
            assert (
                "H1Touch" == task._env.robot.__class__.__name__
            ), "Tactile observations are only available for H1Touch robot"

    @property
    def observation_space(self):
        proprio_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.task._env.robot.dof * 2 - 1 - 13,),
            dtype=np.float64,
        )

        if self._camera_ob:
            image_example = self.get_camera_obs()
            camera_spaces = [
                (
                    key,
                    Box(
                        low=0, high=255, shape=image_example[key].shape, dtype=np.uint8
                    ),
                )
                for key in image_example
            ]

        if self._tactile_ob:
            tactile_example = self.get_tactile_obs()
            tactile_spaces = [
                (
                    key,
                    Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=tactile_example[key].shape,
                        dtype=np.float64,
                    ),
                )
                for key in tactile_example
            ]

        if self._tactile_ob and self._camera_ob:
            return Dict([("proprio", proprio_space)] + camera_spaces + tactile_spaces)
        elif self._tactile_ob:
            return Dict([("proprio", proprio_space)] + tactile_spaces)
        elif self._camera_ob:
            return Dict([("proprio", proprio_space)] + camera_spaces)
        else:
            return proprio_space

    def get_obs(self):
        position = self.task._env.robot.joint_angles()
        velocity = self.task._env.robot.joint_velocities()
        state = np.concatenate((position, velocity))

        tactile = None
        if self._tactile_ob:
            tactile = self.get_tactile_obs()

        camera = None
        if self._camera_ob:
            camera = self.get_camera_obs()

        if tactile and camera:
            state = dict(
                [("proprio", state)] + list(tactile.items()) + list(camera.items())
            )
        elif tactile:
            state = dict([("proprio", state)] + list(tactile.items()))
        elif camera:
            state = dict([("proprio", state)] + list(camera.items()))

        return state

    def get_tactile_obs(self):
        """
        Touch data is in the form of a dictionary with keys as the sensor names and values as the touch data.
        The touch data is then reshaped to a 3D array with the first dimension as the number of components (x-y-z),
        and the second and third dimensions as the touch data in a 2D grid, e.g., np.reshape(touch_data, (3, 2, 4))[[1, 2, 0]]
        (note that Mujoco returns them in the order z-x-y).
        """
        model = self.task._env.model
        data = self.task._env.data
        sensor_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(model.nsensor)
        ]

        touch = dict(
            [
                ("_".join(["tactile", *name.split("_")[:-1]]), data.sensor(name).data)
                for i, name in enumerate(sensor_names)
                if name.endswith("_touch")
            ]
        )

        for key in touch:
            if key != "tactile_torso":
                touch[key] = touch[key].reshape(3, 2, 4)[[1, 2, 0]]
            else:
                touch[key] = touch[key].reshape(3, 4, 8)[[1, 2, 0]]

        return touch

    def step(self, action):
        _, rew, terminated, truncated, info = self.task.step(action)
        obs = self.get_obs()

        return obs, rew, terminated, truncated, info

    def get_camera_obs(self):
        left_eye = self.task._env.mujoco_renderer.render(
            "rgb_array", camera_name="left_eye_camera"
        )
        right_eye = self.task._env.mujoco_renderer.render(
            "rgb_array", camera_name="left_eye_camera"
        )
        return {"image_left_eye": left_eye, "image_right_eye": right_eye}

    def reset_model(self):
        return self.get_obs()

    def normalize_action(self, action):
        return (
            2
            * (action - self.task._env.action_low)
            / (self.task._env.action_high - self.task._env.action_low)
            - 1
        )
