"""Core track generation, projection, and centre-line interchange."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _points_array(points: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        raise ValueError("centre line must contain at least three (x, y) points")
    if not np.all(np.isfinite(array)):
        raise ValueError("centre-line coordinates must be finite")
    if np.any(np.linalg.norm(np.roll(array, -1, axis=0) - array, axis=1) <= 1e-9):
        raise ValueError("adjacent centre-line points must be distinct")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def signed_curvature(points: np.ndarray) -> np.ndarray:
    """Estimate signed planar curvature for a closed sampled curve."""

    p = _points_array(points)
    previous = np.roll(p, 1, axis=0)
    following = np.roll(p, -1, axis=0)
    first = (following - previous) * 0.5
    second = following - (2.0 * p) + previous
    denominator = np.maximum(np.sum(first * first, axis=1) ** 1.5, 1e-12)
    return (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator


@dataclass(frozen=True, slots=True)
class TrackProjection:
    progress: float
    cross_track_error: float
    heading: float
    point: tuple[float, float]
    segment_index: int
    distance: float


@dataclass(frozen=True, slots=True)
class ParametricTrack:
    """Closed centre line with arc-length-aware nearest-segment projection."""

    points: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _points_array(self.points))

    @classmethod
    def generate(
        cls,
        samples: int = 800,
        mean_radius: float = 6.0,
        wave_amplitude: float = 2.2,
        lobes: int = 3,
    ) -> "ParametricTrack":
        """Generate a simple, closed circuit containing both turn directions.

        The polar curve ``r(t)=R+A*cos(k*t)`` is non-self-intersecting because
        ``R>A``.  The chosen amplitude is large enough to introduce negative
        curvature between its outward lobes, defeating constant-steer policies.
        """

        if samples < 32:
            raise ValueError("samples must be at least 32")
        if mean_radius <= wave_amplitude or wave_amplitude <= 0.0:
            raise ValueError("require mean_radius > wave_amplitude > 0")
        if lobes < 2:
            raise ValueError("lobes must be at least 2")
        theta = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
        radius = mean_radius + wave_amplitude * np.cos(lobes * theta)
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        return cls(points)

    @property
    def segment_vectors(self) -> np.ndarray:
        return np.roll(self.points, -1, axis=0) - self.points

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.segment_vectors, axis=1)

    @property
    def cumulative_lengths(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.segment_lengths)))

    @property
    def length(self) -> float:
        return float(np.sum(self.segment_lengths))

    def project(self, position: tuple[float, float] | np.ndarray) -> TrackProjection:
        query = np.asarray(position, dtype=np.float64)
        if query.shape != (2,) or not np.all(np.isfinite(query)):
            raise ValueError("position must be a finite (x, y) pair")

        starts = self.points
        vectors = self.segment_vectors
        squared_lengths = np.sum(vectors * vectors, axis=1)
        fractions = np.clip(np.sum((query - starts) * vectors, axis=1) / squared_lengths, 0.0, 1.0)
        candidates = starts + fractions[:, None] * vectors
        offsets = query - candidates
        squared_distances = np.sum(offsets * offsets, axis=1)
        index = int(np.argmin(squared_distances))
        tangent = vectors[index] / np.sqrt(squared_lengths[index])
        # Positive cross-track error is left of the centre-line direction.
        signed = tangent[0] * offsets[index, 1] - tangent[1] * offsets[index, 0]
        progress = self.cumulative_lengths[index] + fractions[index] * self.segment_lengths[index]
        return TrackProjection(
            progress=float(progress % self.length),
            cross_track_error=float(signed),
            heading=float(np.arctan2(tangent[1], tangent[0])),
            point=(float(candidates[index, 0]), float(candidates[index, 1])),
            segment_index=index,
            distance=float(np.sqrt(squared_distances[index])),
        )

    def point_at(self, progress: float) -> tuple[float, float]:
        wrapped = float(progress) % self.length
        cumulative = self.cumulative_lengths
        index = min(int(np.searchsorted(cumulative, wrapped, side="right") - 1), len(self.points) - 1)
        fraction = (wrapped - cumulative[index]) / self.segment_lengths[index]
        point = self.points[index] + fraction * self.segment_vectors[index]
        return float(point[0]), float(point[1])

    def save_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y"))
            writer.writerows(self.points.tolist())

    @classmethod
    def load_csv(cls, path: str | Path) -> "ParametricTrack":
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if {"x", "y"}.issubset(fields):
                x_name, y_name = "x", "y"
            elif {"x_m", "y_m"}.issubset(fields):
                # The generated world includes units in its richer CSV schema.
                x_name, y_name = "x_m", "y_m"
            else:
                raise ValueError("track CSV must have x/y or x_m/y_m columns")
            return cls([(float(row[x_name]), float(row[y_name])) for row in reader])

    from_csv = load_csv

    def to_csv(self, path: str | Path) -> None:
        self.save_csv(path)


def wrapped_progress_delta(current: float, previous: float, track_length: float) -> float:
    """Shortest signed arc-length change, including the finish-line wrap."""

    if track_length <= 0.0:
        raise ValueError("track_length must be positive")
    return float((current - previous + 0.5 * track_length) % track_length - 0.5 * track_length)
