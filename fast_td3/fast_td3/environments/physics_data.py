import torch

class PhysicsData:
    """Wrapper for physics data to be used in the environment."""

    def __init__(self, xpos, qpos, qvel):
        self._xpos = xpos
        self._qpos = qpos
        self._qvel = qvel

    @property
    def xpos(self):
        return self._xpos

    @property
    def qpos(self):
        return self._qpos

    @property
    def qvel(self):
        return self._qvel
