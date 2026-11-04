"""Connect a LeRobot policy to ReposeEnv's existing policy_interface.py client."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import socket
import struct
import sys
import threading

import numpy as np


MAX_BYTES = 1_048_576


def _receive(connection: socket.socket) -> dict:
    def exact(size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            block = connection.recv(size - len(chunks))
            if not block:
                raise ConnectionError("ReposeEnv closed the connection")
            chunks.extend(block)
        return bytes(chunks)

    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_BYTES:
        raise ValueError("Observation exceeds the 1 MiB protocol limit")
    message = json.loads(exact(size))
    if not isinstance(message, dict):
        raise ValueError("Expected a JSON object")
    return message


def _send(connection: socket.socket, message: dict) -> None:
    raw = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("Response exceeds the 1 MiB protocol limit")
    connection.sendall(struct.pack("!I", len(raw)) + raw)


def _observation(payload: dict, contract: dict, image_shape: tuple[int, int, int]):
    fields = contract["interface"]["observation"]["robot_state"]
    robot, env = payload["robot_state"], payload["env"]
    if set(robot) != set(fields):
        raise ValueError("Robot observation fields differ from the dataset contract")
    parts = []
    for key in fields:
        values = np.asarray(robot[key], dtype=np.float32)
        size = contract["interface_features"][key]["shape"][0]
        if values.shape != (1, size) or not np.isfinite(values).all():
            raise ValueError(f"Invalid robot observation: {key}")
        parts.append(values[0])
    state = np.concatenate(parts)
    video = env["images.camera2"]
    if video["shape"] != [1, *image_shape]:
        raise ValueError("camera2 image shape differs from the dataset")
    raw = base64.b64decode(video["data"], validate=True)
    if len(raw) != int(np.prod(image_shape)):
        raise ValueError("camera2 byte count differs from its shape")
    image = np.frombuffer(raw, dtype=np.uint8).reshape(image_shape).copy()
    instruction = env["instruction"]
    if not isinstance(instruction, list) or len(instruction) != 1 or not isinstance(instruction[0], str):
        raise ValueError("Environment must send one language instruction")
    return state, image, instruction[0]


class Bridge:
    """One-client policy server; ReposeEnv stays in its separate Conda process."""

    def __init__(self, model, recorder=None, timeout: float = 300.0):
        self.model, self.recorder = model, recorder
        self.socket = socket.create_server(("127.0.0.1", 0))
        self.socket.settimeout(0.5)
        self.port = self.socket.getsockname()[1]
        self.timeout = timeout
        self.stopped = threading.Event()
        self.error = None

    def close(self) -> None:
        self.stopped.set()
        self.socket.close()

    def serve(self) -> None:
        try:
            while not self.stopped.is_set():
                try:
                    connection, _ = self.socket.accept()
                    break
                except socket.timeout:
                    continue
            else:
                return
            with connection:
                connection.settimeout(self.timeout)
                while not self.stopped.is_set():
                    try:
                        request = _receive(connection)
                    except ConnectionError:
                        break
                    try:
                        response = self._handle(request)
                    except Exception as error:
                        response = {"ok": False, "error": str(error)}
                    _send(connection, response)
                    if request.get("op") == "close":
                        break
        except Exception as error:
            self.error = error
        finally:
            self.close()

    def _handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "health":
            if request.get("protocol") != 1:
                raise ValueError("Protocol version must be 1")
            return {"ok": True, "protocol": 1, "runtime": "model",
                    "env": Path(sys.prefix).name, "environment": "alignbox3d",
                    "policy_name": "pi05", "input_mode": "multimodal",
                    "contract": self.model.contract, "obs_shape": [1, 20],
                    "action_shape": [self.model.action_steps, 14], "action_horizon": self.model.chunk_size,
                    "image_shape": list(self.model.image_shape),
                    "image_key": "observation.images.camera2", "requires_instruction": True}
        if op == "reset":
            seed = request["seed"]
            self.model.reset(seed)
            if self.recorder is not None:
                self.recorder.start(seed)
            return {"ok": True}
        if op == "predict":
            state, image, instruction = _observation(
                request["obs"], self.model.contract, self.model.image_shape)
            if self.recorder is not None:
                self.recorder.add(image)
            actions = self.model.predict(state, image, instruction)
            return {"ok": True, "action": np.asarray(actions, dtype=np.float32).tolist()}
        if op == "close":
            return {"ok": True}
        raise ValueError(f"Unsupported request: {op!r}")
