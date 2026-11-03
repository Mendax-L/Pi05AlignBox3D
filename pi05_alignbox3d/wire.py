"""Small TCP adapter for ReposeEnv policy_interface.py protocol version 1."""
from __future__ import annotations

import base64
import binascii
import json
import math
import socket
import struct
import threading

import numpy as np


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576


def _receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        piece = connection.recv(size - len(chunks))
        if not piece:
            raise ConnectionError("Peer closed the IPC connection.")
        chunks.extend(piece)
    return bytes(chunks)


def receive_message(connection):
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("IPC message exceeds 1 MiB.")
    value = json.loads(_receive_exact(connection, size),
                       parse_constant=lambda constant: (_ for _ in ()).throw(
                           ValueError(f"Non-finite JSON value: {constant}")))
    if not isinstance(value, dict):
        raise ValueError("IPC message must be a JSON object.")
    return value


def send_message(connection, value):
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("IPC response exceeds 1 MiB.")
    connection.sendall(struct.pack("!I", len(data)) + data)


def decode_observation(payload, contract, image_shape):
    """Use the named robot fields and one RGB frame from the public client."""
    robot = payload.get("robot_state")
    environment = payload.get("env")
    if not isinstance(robot, dict) or not isinstance(environment, dict):
        raise ValueError("Prediction requires named robot_state and env fields.")
    fields = contract["interface"]["observation"]["robot_state"]
    if set(robot) != set(fields):
        raise ValueError("Robot fields differ from the dataset contract.")
    parts = []
    for name in fields:
        values = np.asarray(robot[name], dtype=np.float32)
        expected = contract["interface_features"][name]["shape"][0]
        if values.shape != (1, expected) or not np.isfinite(values).all():
            raise ValueError(f"Robot field {name} must be finite [1,{expected}].")
        parts.append(values)
    state = np.concatenate(parts, axis=1)
    image = environment.get("images.camera2")
    if not isinstance(image, dict) or image.get("shape") != [1, *image_shape]:
        raise ValueError("camera2 RGB history must have shape [1,H,W,3].")
    encoded = image.get("data")
    if not isinstance(encoded, str):
        raise ValueError("camera2 payload must contain base64 data.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid camera2 base64 payload.") from error
    if len(raw) != math.prod([1, *image_shape]):
        raise ValueError("camera2 byte length differs from shape.")
    image_array = np.frombuffer(raw, dtype=np.uint8).reshape(1, *image_shape)
    instruction = environment.get("instruction")
    if (not isinstance(instruction, list) or len(instruction) != 1
            or not isinstance(instruction[0], str) or not instruction[0].strip()):
        raise ValueError("Client must send one nonempty task instruction.")
    return state, image_array


def validate_contract(contract):
    if not isinstance(contract, dict) or contract.get("obs_dim") != 20 or contract.get("action_dim") != 14:
        raise ValueError("Expected the AlignBox3D 20-D state and 14-D action contract.")
    if not isinstance(contract.get("interface"), dict) or not isinstance(contract.get("interface_features"), dict):
        raise ValueError("Dataset lacks its public named interface contract.")
    if any(not isinstance(contract.get(name), list) or len(contract[name]) != 14
           for name in ("action_low", "action_high")):
        raise ValueError("Dataset action bounds are invalid.")
    if any(not math.isfinite(a) or not math.isfinite(b) or a >= b
           for a, b in zip(contract["action_low"], contract["action_high"])):
        raise ValueError("Dataset action bounds must be finite and ordered.")
    return contract


class PolicyServer:
    """Serve one ReposeEnv client while the simulator runs as a subprocess."""

    def __init__(self, handler, host="127.0.0.1", port=0, timeout=300.0):
        self.handler = handler
        self.socket = socket.create_server((host, port), family=socket.AF_INET)
        self.socket.settimeout(0.2)
        self.port = self.socket.getsockname()[1]
        self.timeout = timeout
        self._stop = threading.Event()
        self._connection = None

    def serve(self):
        while not self._stop.is_set():
            try:
                connection, _ = self.socket.accept()
                break
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
        else:
            return
        self._connection = connection
        with connection:
            connection.settimeout(self.timeout)
            while not self._stop.is_set():
                request = None
                try:
                    request = receive_message(connection)
                    response = self.handler(request)
                except (ConnectionError, socket.timeout):
                    break
                except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as error:
                    response = {"ok": False, "error": str(error)}
                try:
                    send_message(connection, response)
                except OSError:
                    break
                if request is None or request.get("op") == "close":
                    break

    def close(self):
        self._stop.set()
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
