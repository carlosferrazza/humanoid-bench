import embodied
import numpy as np


class HRLGrid(embodied.Env):
    def __init__(self, grid=16, length=1000):
        assert length > 0
        self.grid = grid
        self.goal = None
        self.particle = None
        # self.moves = {
        #     (1, 8, 1, 2, 4, 1, 6, 4, 9, 4)[:difficulty]: (0, +1),
        #     (4, 7, 9, 5, 2, 3, 6, 3, 7, 8)[:difficulty]: (0, -1),
        #     (8, 8, 1, 8, 3, 8, 8, 6, 9, 5)[:difficulty]: (+1, 0),
        #     (2, 8, 6, 9, 8, 4, 6, 3, 5, 8)[:difficulty]: (-1, 0),
        # }
        # self.prev = collections.deque(maxlen=difficulty)
        self.moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        self.length = length
        self.random = np.random.RandomState()
        self.player = None
        self.count = None
        self.done = True

    @property
    def act_space(self):
        return {
            "action": embodied.Space(np.int64, (), 0, len(self.moves)),
            "reset": embodied.Space(bool),
        }

    @property
    def obs_space(self):
        return {
            "image": embodied.Space(np.uint8, (64, 64, 3)),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    def step(self, action):
        w = h = self.grid
        if self.done or action["reset"]:
            randpos = lambda: (
                self.random.randint(1, w - 1),
                self.random.randint(1, h - 1),
            )
            self.player = randpos()
            self.particle = randpos()
            self.goal = randpos()
            self.count = 0
            self.done = False
            # self.prev.clear()
            return self._obs(reward=0.0, is_first=True)
        # self.prev.append(action['action'])
        # move = self.moves.get(tuple(self.prev), (0, 0))
        particle = (
            np.clip(self.particle[0] + self.random.randint(-1, 2), 1, w - 2),
            np.clip(self.particle[1] + self.random.randint(-1, 2), 1, h - 2),
        )
        if particle != self.player:
            self.particle = particle
        move = self.moves[action["action"]]
        player = (
            np.clip(self.player[0] + move[0], 1, w - 2),
            np.clip(self.player[1] + move[1], 1, h - 2),
        )
        if player != self.particle:
            self.player = player
        reward = float(self.player == self.goal)
        self.count += 1
        self.done = self.done or (self.count >= self.length)
        return self._obs(reward=reward, is_last=self.done)

    def render(self):
        w = h = self.grid
        image = np.zeros((w, h, 3), np.uint8) + 255
        image[self.goal] = (0, 192, 0)
        image[self.player] = (0, 0, 192)
        image[self.particle] = (192, 0, 0)
        image = np.repeat(np.repeat(image, 64 // w, 0), 64 // h, 1)
        image[: +64 // w, :] = [192, 192, 192]
        image[-64 // w :, :] = [192, 192, 192]
        image[:, : +64 // h] = [192, 192, 192]
        image[:, -64 // h :] = [192, 192, 192]
        assert image.shape == (64, 64, 3), image.shape
        return image

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        return dict(
            image=self.render(),
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
