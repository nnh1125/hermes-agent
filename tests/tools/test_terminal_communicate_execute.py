"""Local foreground terminal execution through subprocess.communicate()."""

import subprocess
from types import SimpleNamespace

from tools.environments.local import LocalEnvironment


def _environment(tmp_path):
    return LocalEnvironment(cwd=str(tmp_path), timeout=5)


def test_local_bounded_foreground_uses_communicate(monkeypatch, tmp_path):
    env = _environment(tmp_path)
    calls = []
    process = SimpleNamespace(returncode=0, pid=1234)

    def communicate(*, timeout=None):
        calls.append(timeout)
        return "output", None

    process.communicate = communicate
    monkeypatch.setattr(env, "_run_bash", lambda *args, **kwargs: process)
    monkeypatch.setattr(env, "_update_cwd", lambda _result: None)
    monkeypatch.setattr(
        "agent.deadline.run_bounded_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bounded sync path used")),
    )

    result = env.execute("printf output", timeout=3, bounded_capture=True)

    assert calls == [3]
    assert result["output"] == "output"
    assert result["returncode"] == 0


def test_local_bounded_foreground_kills_tree_after_communicate_timeout(monkeypatch, tmp_path):
    env = _environment(tmp_path)
    calls = []
    killed = []
    process = SimpleNamespace(returncode=124, pid=5678)

    def communicate(*, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("printf output", timeout, output="partial")
        return "partial", None

    process.communicate = communicate
    monkeypatch.setattr(env, "_run_bash", lambda *args, **kwargs: process)
    monkeypatch.setattr(env, "_kill_process", lambda proc: killed.append(("process", proc)))
    monkeypatch.setattr(
        "agent.deadline.kill_process_tree",
        lambda pid, **_kwargs: killed.append(("tree", pid)),
    )
    monkeypatch.setattr(env, "_update_cwd", lambda _result: None)

    result = env.execute("printf output", timeout=3, bounded_capture=True)

    assert calls == [3, None]
    assert killed == [("process", process), ("tree", 5678)]
    assert result["returncode"] == 124
    assert "timed out" in result["output"]
