"""Regression tests for the maintained API command-line entry point."""

import pytest

from aimedres.cli import commands, secure_serve, serve


class _FakeServer:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)


def test_loopback_serve_uses_maintained_server(monkeypatch):
    fake_server = _FakeServer()
    monkeypatch.delenv("AIMEDRES_ENV", raising=False)
    monkeypatch.delenv("STRICT_SECURITY", raising=False)
    monkeypatch.setattr(secure_serve, "SecureAPIServer", lambda: fake_server)

    assert secure_serve.main(["--host", "127.0.0.1", "--port", "8090"]) == 0
    assert fake_server.calls == [
        {"host": "127.0.0.1", "port": 8090, "debug": False, "ssl_context": None}
    ]


def test_production_mode_rejects_debug(monkeypatch):
    monkeypatch.setenv("AIMEDRES_ENV", "production")

    with pytest.raises(SystemExit):
        secure_serve.main(["--debug"])


def test_public_bind_requires_production_security_gate(monkeypatch):
    monkeypatch.delenv("AIMEDRES_ENV", raising=False)
    monkeypatch.delenv("STRICT_SECURITY", raising=False)

    with pytest.raises(SystemExit):
        secure_serve.main(["--host", "0.0.0.0"])


def test_unified_and_legacy_serve_commands_delegate_to_maintained_entrypoint(monkeypatch):
    received = []

    def fake_main(argv):
        received.append(argv)
        return 0

    monkeypatch.setattr(secure_serve, "main", fake_main)

    assert commands.main(["serve", "--host", "127.0.0.1", "--port", "8091"]) == 0
    assert serve.main(["--host", "127.0.0.1", "--port", "8092"]) == 0
    assert received == [
        ["--host", "127.0.0.1", "--port", "8091"],
        ["--host", "127.0.0.1", "--port", "8092"],
    ]
