"""Tests for streaming UI hooks module."""

from unittest.mock import MagicMock

import pytest
from amplifier_core import HookResult
from amplifier_module_hooks_streaming_ui import StreamingUIHooks
from amplifier_module_hooks_streaming_ui import mount


@pytest.mark.asyncio
async def test_mount_registers_hooks():
    """Test that mount registers all required hooks."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.register = MagicMock()

    config = {"ui": {"show_thinking_stream": True, "show_tool_lines": 5, "show_token_usage": True}}

    await mount(coordinator, config)

    # Verify all hooks are registered
    expected_events = ["content_block:start", "content_block:end", "tool:pre", "tool:post", "llm:response"]

    for event in expected_events:
        # Find if this event was registered
        registered = any(call[0][0] == event for call in coordinator.hooks.register.call_args_list)
        assert registered, f"Event {event} was not registered"


@pytest.mark.asyncio
async def test_mount_with_defaults():
    """Test mount works with default config."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.register = MagicMock()

    # Empty config should use defaults
    config = {}

    await mount(coordinator, config)

    # Should register 6 hooks: content_block:start, content_block:end, tool:pre, tool:post, llm:response, prompt:complete
    assert coordinator.hooks.register.call_count == 6


class TestStreamingUIHooks:
    """Test the StreamingUIHooks class."""

    @pytest.mark.asyncio
    async def test_thinking_block_start(self, capsys):
        """Test thinking block start detection."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=True)

        data = {"data": {"block_type": "thinking", "block_index": 0}}

        result = await hooks.handle_content_block_start("content_block:start", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 0 in hooks.thinking_blocks

        captured = capsys.readouterr()
        assert "🧠 Thinking..." in captured.out

    @pytest.mark.asyncio
    async def test_thinking_block_disabled(self, capsys):
        """Test thinking blocks are not shown when disabled."""
        hooks = StreamingUIHooks(show_thinking=False, show_tool_lines=5, show_token_usage=True)

        data = {"data": {"block_type": "thinking", "block_index": 0}}

        result = await hooks.handle_content_block_start("content_block:start", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 0 not in hooks.thinking_blocks

        captured = capsys.readouterr()
        assert "Thinking" not in captured.out

    @pytest.mark.asyncio
    async def test_thinking_block_end(self, capsys):
        """Test thinking block display on end."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=True)

        # Track the block first
        hooks.thinking_blocks[0] = {"started": True}

        data = {
            "data": {"block_index": 0, "block": {"type": "thinking", "thinking": "This is a test thought process."}}
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 0 not in hooks.thinking_blocks  # Should be cleaned up

        captured = capsys.readouterr()
        assert "=" * 60 in captured.out
        assert "Thinking:" in captured.out
        assert "This is a test thought process." in captured.out

    @pytest.mark.asyncio
    async def test_tool_pre(self, capsys):
        """Test tool invocation display."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=3, show_token_usage=True)

        data = {
            "tool_name": "filesystem_read",
            "tool_input": {"path": "/some/long/path/to/file.txt", "encoding": "utf-8"},
        }

        result = await hooks.handle_tool_pre("tool:pre", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "🔧 Using tool: filesystem_read" in captured.out
        assert "Arguments:" in captured.out
        assert "path" in captured.out

    @pytest.mark.asyncio
    async def test_tool_post_success(self, capsys):
        """Test successful tool result display."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=3, show_token_usage=True)

        data = {"tool_name": "filesystem_read", "tool_response": {"success": True, "output": "File contents here"}}

        result = await hooks.handle_tool_post("tool:post", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "✅ Tool result: filesystem_read" in captured.out
        assert "File contents here" in captured.out

    @pytest.mark.asyncio
    async def test_tool_post_failure(self, capsys):
        """Test failed tool result display."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=3, show_token_usage=True)

        data = {"tool_name": "filesystem_read", "tool_response": {"success": False, "output": "Error: File not found"}}

        result = await hooks.handle_tool_post("tool:post", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "❌ Tool result: filesystem_read" in captured.out
        assert "Error: File not found" in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_display(self, capsys):
        """Test token usage buffering at llm:response and display at prompt:complete."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=True)

        # Step 1: llm:response should BUFFER usage, not display
        llm_data = {
            "status": "ok",
            "data": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "usage": {"input": 1234, "output": 567},
            },
        }

        result = await hooks.handle_llm_response("llm:response", llm_data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        # Should NOT display yet (buffered)
        captured = capsys.readouterr()
        assert "📊 Token Usage" not in captured.out, "Should buffer, not display at llm:response"

        # Step 2: prompt:complete should DISPLAY buffered usage
        prompt_data = {
            "prompt": "test",
            "response": "test response",
        }

        result = await hooks.handle_prompt_complete("prompt:complete", prompt_data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        # NOW should display
        captured = capsys.readouterr()
        assert "📊 Token Usage" in captured.out
        assert "Input: 1,234" in captured.out
        assert "Output: 567" in captured.out
        assert "Total: 1,801" in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_disabled(self, capsys):
        """Test token usage is not shown when disabled."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=False)

        data = {
            "status": "ok",
            "data": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "usage": {"input": 1234, "output": 567},
            },
        }

        result = await hooks.handle_llm_response("llm:response", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "Token Usage" not in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_on_error(self, capsys):
        """Test token usage is not shown on error responses."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=True)

        data = {
            "status": "error",
            "error": "API timeout",
        }

        result = await hooks.handle_llm_response("llm:response", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "Token Usage" not in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_missing_data(self, capsys):
        """Test token usage handles missing usage data gracefully."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=True)

        data = {
            "status": "ok",
            "data": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                # No usage field
            },
        }

        result = await hooks.handle_llm_response("llm:response", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "Token Usage" not in captured.out

    def test_truncate_lines(self):
        """Test line truncation logic."""
        hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=3, show_token_usage=True)

        # Test short text (no truncation)
        text = "line1\nline2\nline3"
        result = hooks._truncate_lines(text, 3)
        assert result == text

        # Test long text (truncation)
        text = "line1\nline2\nline3\nline4\nline5"
        result = hooks._truncate_lines(text, 3)
        assert result == "line1\nline2\nline3\n... (2 more lines)"

        # Test empty text
        result = hooks._truncate_lines("", 3)
        assert result == "(empty)"

        # Test single line
        text = "single line"
        result = hooks._truncate_lines(text, 3)
        assert result == text


@pytest.mark.asyncio
async def test_non_thinking_blocks_ignored():
    """Test that non-thinking blocks are ignored."""
    hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=True)

    # Test text block (should be ignored)
    data = {"data": {"block_type": "text", "block_index": 0}}

    result = await hooks.handle_content_block_start("content_block:start", data)
    assert isinstance(result, HookResult)
    assert result.action == "continue"
    assert 0 not in hooks.thinking_blocks


@pytest.mark.asyncio
async def test_tool_with_string_result(capsys):
    """Test tool result when result is a plain string."""
    hooks = StreamingUIHooks(show_thinking=True, show_tool_lines=5, show_token_usage=True)

    data = {"tool_name": "some_tool", "tool_response": "Simple string result"}

    result = await hooks.handle_tool_post("tool:post", data)

    assert isinstance(result, HookResult)
    assert result.action == "continue"

    captured = capsys.readouterr()
    assert "✅ Tool result: some_tool" in captured.out
    assert "Simple string result" in captured.out
