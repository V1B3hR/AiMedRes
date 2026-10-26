"""Regression coverage for legacy training verification command execution."""

from types import SimpleNamespace

from aimedres.core import verify_training


def test_verify_training_uses_argument_list_without_shell(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="expected output", stderr="")

    monkeypatch.setattr(verify_training.subprocess, "run", fake_run)

    assert verify_training.run_test("safe command", ["python", "-V"], ["expected"])
    assert captured["command"] == ["python", "-V"]
    assert captured["shell"] is False
