"""Centre-line helpers shared by the expert (collect.py) and evaluator (evaluate.py).

The figure-eight CROSSES ITSELF, and that breaks the obvious "nearest point on
the centre line" lookup: at the crossing the nearest point can belong to the
other branch, so the expert would suddenly steer onto the wrong loop and the
cross-track error would read near-zero while the car drove off. `Progress`
fixes it by only ever searching a window around where the car already was, so
the index advances monotonically instead of teleporting.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_centreline(path=None):
    """Load the centre line, generating the track first if it is not there.

    The world and the centre line are written together by make_track.py and are
    only meaningful together, so neither is committed and both are (re)built on
    demand. Shipping one and generating the other is how a controller ends up
    following a road that is no longer where it thinks it is.
    """
    path = Path(path) if path else (HERE / "centreline.csv")
    if not path.exists():
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "behavioral_cloning.make_track"],
                       cwd=str(HERE.parent), check=True)
    return np.loadtxt(path, delimiter=",")


def lap_length(centre):
    return float(np.sum(np.hypot(*(np.roll(centre, -1, 0) - centre).T)))


class Progress:
    """Track position along a possibly self-intersecting centre line."""

    def __init__(self, centre, back=6, ahead=60):
        self.c = centre
        self.i = None
        self.back, self.ahead = back, ahead

    def update(self, x, y):
        """Return (index, cross_track_distance) for the car at (x, y)."""
        n = len(self.c)
        if self.i is None:                      # first call: global search
            d = np.hypot(self.c[:, 0] - x, self.c[:, 1] - y)
            self.i = int(np.argmin(d))
            return self.i, float(d[self.i])
        idx = (np.arange(self.i - self.back, self.i + self.ahead) % n)
        d = np.hypot(self.c[idx, 0] - x, self.c[idx, 1] - y)
        k = int(np.argmin(d))
        self.i = int(idx[k])
        return self.i, float(d[k])

    def ahead_point(self, lookahead, step):
        j = (self.i + max(1, int(lookahead / max(step, 1e-6)))) % len(self.c)
        return self.c[j]
