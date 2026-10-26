"""Local foreground terminal execution through subprocess.communicate()."""

import subprocess
from types import SimpleNamespace

import pytest

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


def test_communicate_host_exiting_returns_130(monkeypatch, tmp_path):
    env = _environment(tmp_path)
    monkeypatch.setattr("tools.environments.base._enter_foreground_spawn", lambda: False)

    result = env.execute("echo test", bounded_capture=True)

    assert result["returncode"] == 130
    assert "host is exiting" in result["output"]


def test_communicate_keyboard_interrupt_kills_tree_and_cleans_up(monkeypatch, tmp_path):
    env = _environment(tmp_path)
    killed = []
    process = SimpleNamespace(returncode=-2, pid=4321)

    def communicate(*, timeout=None):
        raise KeyboardInterrupt()

    process.communicate = communicate
    monkeypatch.setattr(env, "_run_bash", lambda *args, **kwargs: process)
    monkeypatch.setattr(env, "_kill_spawned_tree", lambda proc: killed.append(proc))

    with pytest.raises(KeyboardInterrupt):
        env.execute("echo test", bounded_capture=True)

    assert killed == [process]


def test_communicate_propagates_recreated_environment_notice(monkeypatch, tmp_path):
    env = _environment(tmp_path)
    env._mark_recreated()
    process = SimpleNamespace(returncode=0, pid=1111)
    process.communicate = lambda *, timeout=None: ("done", None)

    monkeypatch.setattr(env, "_run_bash", lambda *args, **kwargs: process)
    monkeypatch.setattr(env, "_update_cwd", lambda _result: None)

    result = env.execute("echo test", bounded_capture=True)

    assert result["environment_recreated"] is True
    assert getattr(env, "_recreated_notice_pending", False) is False


@pytest.mark.parametrize(
    "is_local, bounded_capture, with_yield_handler",
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_communicate_bypassed_when_conditions_not_met(
    monkeypatch, tmp_path, is_local, bounded_capture, with_yield_handler,
):
    env = _environment(tmp_path)
    env.is_local = is_local
    sync_called = []

    monkeypatch.setattr(
        env,
        "_execute_local_foreground_communicate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("communicate fast path used")),
    )
    monkeypatch.setattr(
        "agent.deadline.run_bounded_sync",
        lambda _fn, timeout, **_kwargs: sync_called.append(timeout) or SimpleNamespace(
            timed_out=False, value={"output": "sync_path", "returncode": 0},
        ),
    )
    monkeypatch.setattr(env, "_update_cwd", lambda _result: None)

    yield_handler = (lambda _proc, _out: {}) if with_yield_handler else None
    result = env.execute("echo test", timeout=4, bounded_capture=bounded_capture, yield_handler=yield_handler)

    assert len(sync_called) == 1
    assert result["output"] == "sync_path"


@pytest.mark.platforms("posix")
def test_local_bounded_foreground_live_execution(tmp_path):
    env = _environment(tmp_path)
    result = env.execute("printf 'hermes_inline_test'", bounded_capture=True)

    assert result["returncode"] == 0
    assert "hermes_inline_test" in result["output"]
