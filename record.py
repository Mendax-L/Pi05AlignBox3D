"""Save the camera2 frames seen by the policy during each validation rollout."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


class Recorder:
    def __init__(self, directory: Path, fps: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.container = None
        self.stream = None
        self.path = None
        self.frames = 0

    def start(self, seed: int) -> None:
        self.close()
        path = self.directory / f"seed-{seed}.mp4"
        if path.exists():
            raise FileExistsError(f"Video already exists: {path}")
        self.path = path
        self.stream = None
        self.frames = 0

    def add(self, image: np.ndarray) -> None:
        if self.path is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Recorder needs an active episode and an HWC uint8 RGB frame")
        height, width, _ = image.shape
        if self.container is None:
            self.container = av.open(str(self.path), mode="w")
        if self.stream is None:
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = "yuv420p"
        if (width, height) != (self.stream.width, self.stream.height):
            raise ValueError("Camera dimensions changed during an episode")
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        if self.container is None:
            self.path = None
            return
        if self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()
        self.container = None
        self.stream = None
        self.path = None
