# The source code for meters

import pickle
from typing import Dict, Optional


class Meter(object):
    """Meter base class"""

    def __init__(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def save(self):
        pass

    def reset(self):
        """reset the meter"""
        raise NotImplementedError

    def update(self, val):
        pass

    @property
    def smoothed_value(self) -> float:
        """Smoothed value for monitoring"""
        raise NotImplementedError


class AverageMeter(Meter):
    """Computes and stores the average and current value"""
    def __init__(self, round: Optional[int]):
        self.round = round
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def state_dict(self):
        return {
            'val': self.val,
            'sum': self.count,
            'count': self.count,
            'round': self.round
        }

    def load_state_dict(self, state_dict):
        assert isinstance(state_dict, Dict)
        self.val = state_dict['val']
        self.sum = state_dict['sum']
        self.count = state_dict['count']
        self.round = state_dict.get('round', None)

    def save(self, out_file: str):
        with open(out_file, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.val

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            if hasattr(val, "__round__"):
                val = round(val, self.round)
        return val