"""Inline dispatch eligibility for local foreground terminal calls."""

from types import SimpleNamespace

import pytest

import agent.tool_executor as tool_executor


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
