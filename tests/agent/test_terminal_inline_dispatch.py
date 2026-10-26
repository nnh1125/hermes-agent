"""Inline dispatch eligibility for local foreground terminal calls."""

from types import SimpleNamespace

import pytest
from agent import tool_executor


@pytest.fixture
def middleware_result():
    return SimpleNamespace(result="ok")


def _call_kwargs():
    return {
        "agent": SimpleNamespace(),
        "function_name": "terminal",
        "function_args": {},
        "effective_task_id": "task-1",
        "tool_call_id": "call-1",
        "execute": lambda _args: "ok",
    }


def test_local_foreground_terminal_bypasses_worker_pool(monkeypatch, middleware_result):
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(is_local=True),
    )
    monkeypatch.setattr(
        tool_executor,
        "_run_agent_tool_execution_middleware",
        lambda *_args, **_kwargs: middleware_result,
    )

    class UnexpectedPoolCreation:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("local foreground terminal call created a worker pool")

    monkeypatch.setattr(
        "tools.daemon_pool.DaemonThreadPoolExecutor",
        UnexpectedPoolCreation,
    )

    result = tool_executor._run_sequential_tool_execution_middleware(**_call_kwargs())

    assert result is middleware_result


def test_prepared_task_does_not_bypass_worker_pool(monkeypatch, middleware_result):
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(is_local=True),
    )
    monkeypatch.setattr(
        tool_executor,
        "_run_agent_tool_execution_middleware",
        lambda *_args, **_kwargs: middleware_result,
    )
    pool_created = []

    class TrackedPool:
        def __init__(self, *_args, **_kwargs):
            pool_created.append(True)

        def submit(self, _fn, *args, **kwargs):
            return SimpleNamespace(result=lambda: middleware_result)

        def shutdown(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "tools.daemon_pool.DaemonThreadPoolExecutor",
        TrackedPool,
    )

    dummy_future = SimpleNamespace(
        result=lambda **_kwargs: middleware_result,
        cancel=lambda: None,
        done=lambda: True,
    )
    prepared = SimpleNamespace(
        batch=SimpleNamespace(
            authorization_gate=SimpleNamespace(excluded_seconds=lambda: 0.0),
            executor=SimpleNamespace(),
        ),
        tids=[],
        future=dummy_future,
    )
    monkeypatch.setattr("agent.terminal_approval_batch.take_prepared_call", lambda _id: prepared)
    result = tool_executor._run_sequential_tool_execution_middleware(**_call_kwargs())

    assert result is middleware_result


@pytest.mark.parametrize(
    "function_args, is_local, expected",
    [
        ({"background": True}, True, False),
        ({"pty": True}, True, False),
        ({}, False, False),
        ({}, True, True),
    ],
)
def test_terminal_inline_eligibility_requires_local_foreground(
    monkeypatch, function_args, is_local, expected,
):
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(is_local=is_local),
    )

    assert tool_executor._can_run_terminal_inline(
        "terminal", function_args, "task-1"
    ) is expected


def test_non_terminal_call_is_not_inline(monkeypatch):
    monkeypatch.setattr(
        tool_executor,
        "get_active_env",
        lambda _task_id: SimpleNamespace(is_local=True),
    )

    assert not tool_executor._can_run_terminal_inline("read_file", {}, "task-1")


@pytest.mark.parametrize(
    "env_config, expected",
    [
        ({"env_type": "local"}, True),
        ({"env_type": "docker"}, False),
    ],
)
def test_terminal_inline_fallback_to_env_config(monkeypatch, env_config, expected):
    monkeypatch.setattr(tool_executor, "get_active_env", lambda _task_id: None)
    monkeypatch.setattr("tools.terminal_tool._get_env_config", lambda: env_config)

    assert tool_executor._can_run_terminal_inline("terminal", {}, "task-1") is expected


def test_terminal_inline_fallback_handles_env_config_exception(monkeypatch):
    monkeypatch.setattr(tool_executor, "get_active_env", lambda _task_id: None)

    def _broken_config():
        raise RuntimeError("broken config")

    monkeypatch.setattr("tools.terminal_tool._get_env_config", _broken_config)

    assert tool_executor._can_run_terminal_inline("terminal", {}, "task-1") is False

