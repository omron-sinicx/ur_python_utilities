# The MIT License (MIT)
#
# Copyright (c) 2026 Cristian Beltran
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

"""Numerical inverse kinematics via Pinocchio (https://stack-of-tasks.github.io/pinocchio/).

Levenberg-Marquardt damped least squares on the SE(3) log error. Unlike KDL's
``ChainIkSolverPos_NR`` (Newton-Raphson over a pseudo-inverse velocity solver), the
damping is *adaptive*: it grows when a step fails to reduce the error and shrinks when
steps succeed, so the solver degrades gracefully at singularities (straight elbow,
wrist-2 near zero, shoulder singularity) instead of diverging or stalling.

Unlike EAIK (see L{ur_control.eaik_kinematics}), it makes no assumption about the
kinematic family, so it works unchanged on calibrated URDFs whose axes are no longer
exactly parallel/intersecting.
"""

import numpy as np

from ur_control import transformations
from ur_control.constants import JOINT_ORDER

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - exercised only without pinocchio
    pin = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _strip_xml_prolog(xml: str) -> str:
    """Remove the XML declaration; urdfdom rejects strings that carry an encoding decl."""
    import re
    return re.sub(r"<\?xml[^>]*\?>", "", xml, count=1).lstrip()


class PinocchioKinematics(object):
    """Damped least-squares IK for a URDF chain, aligned to the KDL tip frame.

    The public surface mirrors L{ur_control.eaik_kinematics.EAIKKinematics} so the two
    are interchangeable behind C{Arm.inverse_kinematics}.
    """

    def __init__(self,
                 kdl=None,
                 logger=None,
                 robot_description: str = None,
                 file_path: str = None,
                 base_link: str = None,
                 ee_link: str = None,
                 joint_names: list = None,
                 eps_pos: float = 1e-6,
                 eps_rot: float = 1e-6,
                 max_iter: int = 100,
                 damping: float = 1e-6,
                 max_step: float = 0.5,
                 restarts: int = 3):
        """
        Parameters
        ----------
        kdl : ur_pykdl.ur_kinematics, optional
            Existing KDL solver; its base link, tip link and joint names are reused so
            both solvers describe the exact same chain.
        robot_description : str, optional
            URDF XML string (e.g. from the ``/robot_description`` topic).
        file_path : str, optional
            Path to a URDF file on disk. Mutually exclusive with C{robot_description}.
        base_link, ee_link, joint_names : optional
            Override the chain taken from C{kdl}.
        eps_pos, eps_rot : float
            Convergence tolerances on the position [m] and rotation [rad] error.
        max_iter : int
            Maximum LM iterations per attempt.
        damping : float
            Initial LM damping. Adapted during the solve.
        max_step : float
            Maximum norm [rad] of a single joint-space step, to keep the linearisation valid.
        restarts : int
            Extra attempts from perturbed seeds when an attempt fails to converge.
        """
        if pin is None:
            raise ImportError(
                "pinocchio is not installed (apt install ros-$ROS_DISTRO-pinocchio). "
                "Original error: %s" % _IMPORT_ERROR)

        if (robot_description is None) == (file_path is None):
            raise ValueError("Provide exactly one of robot_description or file_path")

        self._logger = logger
        self._eps_pos = eps_pos
        self._eps_rot = eps_rot
        self._max_iter = max_iter
        self._damping0 = damping
        self._max_step = max_step
        self._restarts = restarts
        self._rng = np.random.default_rng(0)

        self._base_link = base_link if base_link is not None else kdl._base_link
        self._ee_link = ee_link if ee_link is not None else kdl._tip_link

        if file_path is not None:
            model = pin.buildModelFromUrdf(file_path)
        else:
            model = pin.buildModelFromXML(_strip_xml_prolog(robot_description))

        for link in (self._base_link, self._ee_link):
            if not model.existFrame(link):
                raise ValueError("URDF has no frame '%s'" % link)

        # Joints on the base_link -> ee_link chain, root to tip. Derived from the model so
        # prefixed robots ('a_bot_shoulder_pan_joint', ...) need no extra configuration.
        chain = self._chain_joint_names(model, self._base_link, self._ee_link)
        if joint_names is None and kdl is not None:
            joint_names = list(getattr(kdl, "_joint_names", JOINT_ORDER))
        if joint_names is not None and all(model.existJointName(n) for n in joint_names):
            self._joint_names = list(joint_names)
        else:
            self._joint_names = chain
        self._num_joints = len(self._joint_names)

        if sorted(self._joint_names) != sorted(chain):
            raise ValueError(
                "joint_names %s do not match the %s -> %s chain %s"
                % (self._joint_names, self._base_link, self._ee_link, chain))

        # Lock every joint that is not part of the arm chain (grippers, extra DOFs) so the
        # configuration vector is exactly the arm's joints, in self._joint_names order.
        extra = [name for name in model.names[1:] if name not in self._joint_names]
        if extra:
            model = pin.buildReducedModel(model,
                                          [model.getJointId(name) for name in extra],
                                          pin.neutral(model))

        self._model = model
        self._data = model.createData()

        # Map JOINT_ORDER -> configuration vector indices. All arm joints are revolute
        # (nq == 1), so the map is a plain permutation.
        self._q_map = np.array([model.joints[model.getJointId(n)].idx_q
                                for n in self._joint_names], dtype=int)
        if model.nq != self._num_joints:
            raise ValueError("Reduced model has nq=%d, expected %d joints"
                             % (model.nq, self._num_joints))

        self._base_fid = model.getFrameId(self._base_link)
        self._ee_fid = model.getFrameId(self._ee_link)

        self._q_lower = model.lowerPositionLimit[self._q_map].copy()
        self._q_upper = model.upperPositionLimit[self._q_map].copy()

        if logger is not None:
            source = file_path if file_path is not None else "robot_description"
            logger.info(
                "Pinocchio IK initialized (joints={}, chain={} -> {}, source={})".format(
                    self._num_joints, self._base_link, self._ee_link, source))

    ### private methods ###

    @staticmethod
    def _chain_joint_names(model, base_link: str, ee_link: str) -> list:
        """Movable joints between two frames, root to tip (fixed joints are frames, not joints)."""
        base_jid = model.frames[model.getFrameId(base_link)].parentJoint
        jid = model.frames[model.getFrameId(ee_link)].parentJoint
        ids = []
        while jid != base_jid and jid > 0:
            ids.append(jid)
            jid = model.parents[jid]
        if jid != base_jid:
            raise ValueError("'%s' is not an ancestor of '%s' in the URDF"
                             % (base_link, ee_link))
        ids.reverse()
        return [model.names[i] for i in ids]

    def _to_model_q(self, q: np.ndarray) -> np.ndarray:
        q_model = np.zeros(self._model.nq)
        q_model[self._q_map] = np.asarray(q, dtype=float)
        return q_model

    def _placement(self, q_model: np.ndarray):
        """End-effector placement in the base_link frame for a model configuration."""
        pin.forwardKinematics(self._model, self._data, q_model)
        pin.updateFramePlacements(self._model, self._data)
        return self._data.oMf[self._base_fid].actInv(self._data.oMf[self._ee_fid])

    def _clamp(self, q_model: np.ndarray) -> np.ndarray:
        """Clamp to the URDF joint limits (no-op for continuous joints: limits are +-inf)."""
        q = q_model[self._q_map]
        np.clip(q, self._q_lower, self._q_upper, out=q)
        q_model[self._q_map] = q
        return q_model

    def _snap_to_seed(self, q: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Shift each joint by multiples of 2*pi toward the seed, staying within limits.

        UR joints span +-2*pi, so an otherwise valid solution can sit a full turn away
        from the seed; that would command a needless 360 deg motion.
        """
        out = np.asarray(q, dtype=float).copy()
        for i in range(self._num_joints):
            best = out[i]
            for k in (-1, 1, -2, 2):
                cand = out[i] + 2.0 * np.pi * k
                if cand < self._q_lower[i] or cand > self._q_upper[i]:
                    continue
                if abs(cand - seed[i]) < abs(best - seed[i]):
                    best = cand
            out[i] = best
        return out

    def _solve_lm(self, target, q_model: np.ndarray):
        """Levenberg-Marquardt descent on the SE(3) log error.

        Returns
        -------
        (q_model, err_pos, err_rot, converged)
        """
        model, data = self._model, self._data
        damping = self._damping0
        eye6 = np.eye(6)

        iMd = self._placement(q_model).actInv(target)
        err = pin.log(iMd).vector
        cost = float(err @ err)

        for _ in range(self._max_iter):
            if (np.linalg.norm(err[:3]) < self._eps_pos
                    and np.linalg.norm(err[3:]) < self._eps_rot):
                return q_model, np.linalg.norm(err[:3]), np.linalg.norm(err[3:]), True

            J = pin.computeFrameJacobian(model, data, q_model, self._ee_fid, pin.LOCAL)
            J = -pin.Jlog6(iMd.inverse()) @ J

            # Try the LM step; on failure grow the damping (-> gradient descent, always
            # a descent direction) and retry. This is what keeps the solver stable when
            # the Jacobian loses rank at a singularity.
            accepted = False
            for _ in range(8):
                v = -J.T @ np.linalg.solve(J @ J.T + damping * eye6, err)
                norm_v = np.linalg.norm(v)
                if norm_v > self._max_step:
                    v = v * (self._max_step / norm_v)
                q_try = self._clamp(pin.integrate(model, q_model, v))
                iMd_try = self._placement(q_try).actInv(target)
                err_try = pin.log(iMd_try).vector
                cost_try = float(err_try @ err_try)
                if cost_try < cost:
                    q_model, iMd, err, cost = q_try, iMd_try, err_try, cost_try
                    damping = max(damping * 0.5, 1e-12)
                    accepted = True
                    break
                damping *= 10.0
            if not accepted:
                break  # no descent direction left: local minimum

        return q_model, np.linalg.norm(err[:3]), np.linalg.norm(err[3:]), False

    ### public methods ###

    @classmethod
    def from_file(cls, file_path: str, kdl=None, logger=None, **kwargs):
        """Construct from a URDF file path."""
        import os
        if not os.path.isfile(file_path):
            raise ValueError("URDF file not found: %s" % file_path)
        return cls(kdl, logger=logger, file_path=file_path, **kwargs)

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: pose of ee_link in base_link as [x, y, z, qx, qy, qz, qw]."""
        bMf = self._placement(self._to_model_q(q))
        quat = transformations.quaternion_from_matrix(bMf.rotation)
        return np.concatenate((bMf.translation, quat))

    def inverse_kinematics(self, pose: np.ndarray, seed: np.ndarray = None) -> np.ndarray:
        """Return a joint configuration reaching C{pose}, or None if none was found.

        Parameters
        ----------
        pose : ndarray
            Target pose of ee_link in base_link as [x, y, z, qx, qy, qz, qw] (or a
            6-vector with sxyz Euler angles).
        seed : ndarray, optional
            Initial guess; the solution stays as close to it as the branch allows.
        """
        target_T = transformations.pose_to_transform(np.asarray(pose, dtype=float))
        target = pin.SE3(target_T[:3, :3], target_T[:3, 3])

        seed = (np.zeros(self._num_joints) if seed is None
                else np.asarray(seed, dtype=float))

        q_model = self._clamp(self._to_model_q(seed))
        for attempt in range(self._restarts + 1):
            q_model, err_pos, err_rot, converged = self._solve_lm(target, q_model)
            if converged:
                return self._snap_to_seed(q_model[self._q_map], seed)
            # Stuck in a local minimum: restart from a perturbed seed. The perturbation
            # grows with the attempt so the first retry stays close to the seed branch.
            scale = 0.1 * (attempt + 1) ** 2
            q_model = self._clamp(
                self._to_model_q(seed + self._rng.normal(scale=scale,
                                                         size=self._num_joints)))

        if self._logger is not None:
            self._logger.debug(
                "Pinocchio IK did not converge (pos err {:.3e} m, rot err {:.3e} rad)"
                .format(err_pos, err_rot))
        return None
