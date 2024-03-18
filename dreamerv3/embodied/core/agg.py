from collections import defaultdict

import numpy as np


class Agg:
    def __init__(self, maxlen=int(1e6)):
        self.maxlen = maxlen
        self.avgs = defaultdict(lambda: [0.0, 0.0])
        self.sums = defaultdict(float)
        self.mins = defaultdict(float)
        self.maxs = defaultdict(float)
        self.lasts = defaultdict(lambda: None)
        self.stacks = defaultdict(list)

    def add(self, key_or_dict, value=None, agg="default", prefix=None, nan="keep"):
        aggs = (agg,) if isinstance(agg, str) else agg
        assert nan in ("keep", "ignore")
        if value is None:
            for key, value in dict(key_or_dict).items():
                key = f"{prefix}/{key}" if prefix else key
                self._add_single(key, value, aggs, nan)
        else:
            assert not prefix, prefix
            self._add_single(key_or_dict, value, aggs, nan)

    def result(self, reset=True, prefix=None):
        metrics = {}
        metrics.update({k: v[0] / v[1] for k, v in self.avgs.items()})
        metrics.update(self.sums)
        metrics.update(self.mins.items())
        metrics.update(self.maxs.items())
        metrics.update(self.lasts.items())
        metrics.update({k: np.stack(v) for k, v in self.stacks.items()})
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        reset and self.reset()
        return metrics

    def reset(self):
        self.avgs.clear()
        self.sums.clear()
        self.mins.clear()
        self.maxs.clear()
        self.lasts.clear()
        self.stacks.clear()

    def _add_single(self, key, value, aggs, nan):
        value = np.asarray(value)
        if nan == "ignore" and np.isnan(value):
            return
        for agg in aggs:
            name = key if len(aggs) == 1 else f"{key}/{agg}"
            if agg == "default":
                agg = "avg" if len(value.shape) <= 1 else "last"
            if agg == "avg":
                avg = self.avgs[name]
                avg[0] += value
                avg[1] += 1
                # assert not np.shares_memory(self.avgs[name][0], value)
            elif agg == "sum":
                self.sums[name] += value
            elif agg == "min":
                self.mins[name] = min(self.mins[name], value)
                # assert not np.shares_memory(self.mins[name], value)
            elif agg == "max":
                self.maxs[name] = max(self.maxs[name], value)
                # assert not np.shares_memory(self.mins[name], value)
            elif agg == "last":
                self.lasts[name] = value
            elif agg == "stack":
                stack = self.stacks[name]
                if len(stack) < self.maxlen:
                    stack.append(value)
            else:
                raise KeyError(agg)
