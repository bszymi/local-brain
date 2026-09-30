"""Hard guarantee that nothing leaves this machine.

Once `enforce()` is called, any attempt to open a network connection to a
non-loopback address raises. Only 127.0.0.1 / ::1 / unix sockets are allowed,
which is all we need to talk to the local Ollama server.
"""

import ipaddress
import os
import socket

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_getaddrinfo = socket.getaddrinfo


class NetworkBlocked(RuntimeError):
    pass


def _is_local(address) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    host = address[0]
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guarded_connect(self, address):
    if not _is_local(address):
        raise NetworkBlocked(f"local-brain: blocked outbound connection to {address!r}")
    return _original_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_local(address):
        raise NetworkBlocked(f"local-brain: blocked outbound connection to {address!r}")
    return _original_connect_ex(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if host not in (None, "localhost", b"localhost"):
        try:
            if not ipaddress.ip_address(host if isinstance(host, str) else host.decode()).is_loopback:
                raise NetworkBlocked(f"local-brain: blocked DNS lookup of {host!r}")
        except ValueError:
            raise NetworkBlocked(f"local-brain: blocked DNS lookup of {host!r}")
    return _original_getaddrinfo(host, *args, **kwargs)


def enforce() -> None:
    # Tell Hugging Face / faster-whisper to never try to reach the Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo
