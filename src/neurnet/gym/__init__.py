from dataclasses import dataclass
from enum import Enum
from typing import Self

import numpy as np

from neurnet.utils.image import show_image

GRASS_PIXEL = np.array([0.16, 0.34, 0.16], dtype=np.float32)
"""Pixel representing grass."""
ROAD_PIXEL = np.array([0.44, 0.44, 0.46], dtype=np.float32)
"""Pixel representing road."""
KERB_PIXEL = np.array([0.75, 0.72, 0.62], dtype=np.float32)
"""Pixel representing kerb."""
CAR_PIXEL = np.array([0.92, 0.20, 0.15], dtype=np.float32)
"""Pixel representing car."""

PHI_MAX = 0.7
"""Maximum angle relative to track's direction of travel, in radians."""
PHI_TURN = 0.15
"""Descrete angle the car turns each time."""


class MiniRacerAction(Enum):
    SteerLeft = -1
    Straight = 0
    SteerRight = 1

    @classmethod
    def random(cls, seed: int = 123) -> Self:
        rng = np.random.default_rng(seed)
        value = rng.integers(-1, 2)
        return cls(value)

    @staticmethod
    def count() -> int:
        return 3


@dataclass(frozen=True)
class MiniRacerReward:
    value: float


@dataclass
class MiniRacerState:
    """Represent's the state of the experiment."""

    theta: float
    """Angular position around the track's center in radians."""
    d: float
    """Distance of car from the track's centerline."""
    phi: float
    """Angle relative to track's direction of travel, in radians."""
    direction: float
    """
    Direction on which the car circulates the track: clock-wise
    or counter clock-wise.
    """
    time: int
    """Time step."""

    def to_numpy(self) -> np.ndarray:
        return np.array([self.theta, self.d, self.phi, self.direction])


class MiniRacer:
    """Represents the mini racer environment."""

    def __init__(
        self,
        size_x: int,
        size_y: int,
        radius: float,
        street_width: float,
        kerb_width: float,
        speed: float,
        time_max: int,
        seed: int = 123,
    ):
        self._rng = np.random.default_rng(seed)

        center_x, center_y = (size_x - 1) / 2, (size_y - 1) / 2
        self.size_x = size_x
        self.size_y = size_y
        self.radius = radius
        self.speed = speed
        self.center = (center_x, center_y)
        self.half_width = street_width / 2
        self.kerb_width = kerb_width
        self.time_max = time_max

        # Create track:
        # 1. Fill the whole canvas with grass.
        # 2. Fill a ring of street.
        # 3. Fill the two kerb boundaries.
        x_idxs, y_idxs = np.mgrid[0:size_x, 0:size_y]
        self.track = np.tile(GRASS_PIXEL, (size_x, size_y, 1))
        radii = np.sqrt((x_idxs - center_x) ** 2 + (y_idxs - center_y) ** 2)
        self.track[np.abs(radii - radius) <= self.half_width] = ROAD_PIXEL
        self.track[np.abs(np.abs(radii - radius) - self.half_width) <= kerb_width] = (
            KERB_PIXEL
        )

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets the episode.

        At the beginning, a coin is flipped to decide whether the car
        circulates clockwise or counter-clockwise during the whole
        episode. This direction is part of the state but it is not
        observable by a snapshot.
        """
        self._state = MiniRacerState(
            theta=self._rng.uniform(0, 2 * np.pi),
            d=self._rng.uniform(-1.0, 1.0),
            phi=self._rng.uniform(-0.2, 0.2),
            direction=int(self._rng.choice([-1, 1])),
            time=0,
        )

        return self.observe()

    def observe(self) -> np.ndarray:
        """Places the car on the track and returns the image."""
        img = self.track.copy()
        distance = self.radius + self._state.d
        position_x = round(self.center[0] + distance * np.cos(self._state.theta))
        position_y = round(self.center[1] + distance * np.sin(self._state.theta))

        for dx, dy in zip((-1, 0), (-1, 0)):
            pixel_x, pixel_y = position_x + dx, position_y + dy
            if 0 <= pixel_x < self.size_x and 0 <= pixel_y < self.size_y:
                img[pixel_x, pixel_y] = CAR_PIXEL

        return img

    def step(self, action: MiniRacerAction) -> tuple[np.ndarray, MiniRacerReward, bool]:
        self._state.phi = float(
            np.clip(self._state.phi + PHI_TURN * action.value, -PHI_MAX, PHI_MAX)
        )

        self._state.d = float(self._state.d + 1.1 * np.sin(self._state.phi))
        if abs(self._state.d) > self.half_width:
            self._state.d = float(
                np.clip(self._state.d, -self.half_width, self.half_width)
            )
            self._state.phi *= 0.4

        self._state.theta = self._state.theta + self._state.direction * self.speed / (
            self.radius + self._state.d
        ) % (2 * np.pi)

        reward = MiniRacerReward(1.0 - abs(self._state.d) / self.half_width)
        self._state.time += 1

        return (self.observe(), reward, self._state.time >= self.time_max)

    def show_track(self) -> None:
        """Renders the track as an image."""
        show_image(self.observe())
