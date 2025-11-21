"""Streaming UI Hooks Module

Display streaming LLM output (thinking blocks, tool calls, and token usage) to console.
"""

import logging
import sys
from typing import Any

from amplifier_core.models import HookResult
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

logger = logging.getLogger(__name__)

# Create console with muted theme for thinking blocks
_muted_theme = Theme(
    {
        "markdown.h1": "dim italic underline",
        "markdown.h2": "dim bold",
        "markdown.h3": "dim bold",
        "markdown.h4": "dim",
        "markdown.h5": "dim",
        "markdown.h6": "dim",
        "markdown.text": "dim",
        "markdown.code": "dim",
        "markdown.code_block": "dim",
        "markdown.link": "dim underline",
        "markdown.bold": "dim bold",
        "markdown.italic": "dim italic",
        "markdown.item.bullet": "dim",
    }
)
_muted_console = Console(theme=_muted_theme, file=sys.stdout, highlight=False)


async def mount(coordinator: Any, config: dict[str, Any]) -> None:
    """Mount streaming UI hooks module.

    Args:
        coordinator: The amplifier coordinator instance
        config: Configuration from profile
    """
    # Extract config from ui section
    ui_config = config.get("ui", {})
    show_thinking = ui_config.get("show_thinking_stream", True)
    show_tool_lines = ui_config.get("show_tool_lines", 5)
    show_token_usage = ui_config.get("show_token_usage", True)

    # Create hook handlers
    hooks = StreamingUIHooks(show_thinking, show_tool_lines, show_token_usage)

    # Register hooks on the coordinator
    coordinator.hooks.register("content_block:start", hooks.handle_content_block_start)
    coordinator.hooks.register("content_block:end", hooks.handle_content_block_end)
    coordinator.hooks.register("tool:pre", hooks.handle_tool_pre)
    coordinator.hooks.register("tool:post", hooks.handle_tool_post)
    coordinator.hooks.register("llm:response", hooks.handle_llm_response)
    coordinator.hooks.register("prompt:complete", hooks.handle_prompt_complete)

    # Log successful mount
    logger.info("Mounted hooks-streaming-ui")

    return


class StreamingUIHooks:
    """Hooks for displaying streaming UI output."""

    def __init__(self, show_thinking: bool, show_tool_lines: int, show_token_usage: bool):
        """Initialize streaming UI hooks.

        Args:
            show_thinking: Whether to display thinking blocks
            show_tool_lines: Number of lines to show for tool I/O
            show_token_usage: Whether to display token usage
        """
        self.show_thinking = show_thinking
        self.show_tool_lines = show_tool_lines
        self.show_token_usage = show_token_usage
        self.thinking_blocks: dict[int, dict[str, Any]] = {}
        self.current_agent_context: str | None = None  # Track current sub-agent
        self.buffered_token_usage: dict[str, Any] | None = None  # Buffer for display after content

    def _parse_agent_from_session_id(self, session_id: str | None) -> str | None:
        """Extract agent name from hierarchical session ID.

        Session ID format follows W3C Trace Context principles:
        {parent-span}-{child-span}_{agent-name}

        Examples:
        - Sub-session: 0000000000000000-7cc787dd22d54f6c_developer-expertise-zen-architect
        - Parent session: 12345678-1234-1234-1234-123456789012 (no underscore, no agent)

        Args:
            session_id: Session ID with optional agent name after underscore

        Returns:
            Agent name if child session (contains underscore), None if parent session
        """
        if not session_id:
            return None

        # W3C Trace Context format: {parent-span}-{child-span}_{agent-name}
        # Underscore separator marks the boundary before agent name
        if "_" in session_id:
            parts = session_id.split("_", 1)  # Split on first underscore only
            if len(parts) == 2:
                # Everything after underscore is agent name
                # Handles namespaced agents like "developer-expertise-zen-architect"
                return parts[1]

        # No underscore = parent session (no agent name)
        return None

    async def handle_content_block_start(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Detect thinking blocks and prepare for display.

        Args:
            _event: Event name (content_block:start) - unused
            data: Event data containing block information

        Returns:
            HookResult with action="continue"
        """
        block_type = data.get("block_type")
        block_index = data.get("block_index")

        # Detect sub-agent context for visual distinction
        session_id = data.get("session_id")
        agent_name = self._parse_agent_from_session_id(session_id)

        # Only track thinking blocks if configured to show them
        if block_type in {"thinking", "reasoning"} and self.show_thinking and block_index is not None:
            self.thinking_blocks[block_index] = {"started": True, "agent": agent_name}
            if agent_name:
                # Sub-agent thinking: status line cyan, 4-space indent
                sys.stderr.write(f"\n    \033[36m🤔 [{agent_name}] Thinking...\033[0m\n")
                sys.stderr.flush()
            else:
                # Parent thinking: status line cyan
                sys.stderr.write("\n\033[36m🧠 Thinking...\033[0m\n")
                sys.stderr.flush()

        return HookResult(action="continue")

    async def handle_content_block_end(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display complete thinking block.

        Args:
            _event: Event name (content_block:end) - unused
            data: Event data containing complete block

        Returns:
            HookResult with action="continue"
        """
        block_index = data.get("block_index")
        block = data.get("block", {})
        block_type = block.get("type")

        # Display thinking block if we were tracking it
        if block_type in {"thinking", "reasoning"} and block_index is not None and block_index in self.thinking_blocks:
            # Extract thinking text from block
            thinking_text = block.get("thinking", "") or block.get("text", "") or _flatten_reasoning_block(block)
            agent_name = self.thinking_blocks[block_index].get("agent")

            if thinking_text:
                # Display formatted thinking block with agent context
                if agent_name:
                    # Sub-agent thinking: dark gray, 4-space indent, markdown rendered
                    print(f"\n    \033[90m{'=' * 56}\033[0m")
                    print(f"    \033[90m[{agent_name}] Thinking:\033[0m")
                    print(f"    \033[90m{'-' * 56}\033[0m")
                    # Render markdown with muted theme, indent each line
                    from io import StringIO

                    buffer = StringIO()
                    temp_console = Console(theme=_muted_theme, file=buffer, highlight=False, width=52)
                    temp_console.print(Markdown(thinking_text))
                    rendered = buffer.getvalue()
                    for line in rendered.rstrip().split("\n"):
                        print(f"    {line}")
                    print(f"    \033[90m{'=' * 56}\033[0m\n")
                else:
                    # Parent thinking: markdown rendered with muted theme (same as sub-agent)
                    from io import StringIO

                    buffer = StringIO()
                    temp_console = Console(theme=_muted_theme, file=buffer, highlight=False, width=60)
                    temp_console.print(Markdown(thinking_text))
                    rendered = buffer.getvalue()

                    print(f"\n\033[90m{'=' * 60}\033[0m")
                    print("\033[90mThinking:\033[0m")
                    print(f"\033[90m{'-' * 60}\033[0m")
                    print(rendered.rstrip())  # Print the muted markdown
                    print(f"\033[90m{'=' * 60}\033[0m\n")

            # Clean up tracking
            del self.thinking_blocks[block_index]

        return HookResult(action="continue")

    async def handle_tool_pre(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display tool invocation with truncated input.

        Shows sub-agent tool calls with indentation and agent name for clarity.

        Args:
            _event: Event name (tool:pre) - unused
            data: Event data containing tool and arguments (includes session_id from defaults)

        Returns:
            HookResult with action="continue"
        """
        tool_name = data.get("tool_name", "unknown")
        tool_input = data.get("tool_input", {})
        session_id = data.get("session_id")

        # Detect if this is a sub-agent's tool call
        agent_name = self._parse_agent_from_session_id(session_id)

        # Format tool input for display - ensure it's a string
        input_str = str(tool_input) if tool_input is not None else ""
        truncated = self._truncate_lines(input_str, self.show_tool_lines)

        if agent_name:
            # Sub-agent tool call: status line cyan, 4-space indent, box drawing
            print(f"\n    \033[36m┌─ 🔧 [{agent_name}] Using tool: {tool_name}\033[0m")
            print(f"    \033[36m│\033[0m  \033[2mArguments: {truncated}\033[0m")
        else:
            # Parent tool call: status line cyan
            print(f"\n\033[36m🔧 Using tool: {tool_name}\033[0m")
            print(f"   \033[2mArguments: {truncated}\033[0m")  # Dim text

        return HookResult(action="continue")

    async def handle_tool_post(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display tool result with truncated output.

        Shows sub-agent tool results with indentation and agent name for clarity.

        Args:
            _event: Event name (tool:post) - unused
            data: Event data containing tool result (includes session_id from defaults)

        Returns:
            HookResult with action="continue"
        """
        tool_name = data.get("tool_name", "unknown")
        result = data.get("tool_response", data.get("result", {}))
        session_id = data.get("session_id")

        # Detect if this is a sub-agent's tool result
        agent_name = self._parse_agent_from_session_id(session_id)

        # Extract output from result (handle different result formats)
        if isinstance(result, dict):
            output = result.get("output")
            # Always ensure output is a string regardless of what we got
            if output is None:
                # No output field, use string representation of entire result
                output = str(result)
            else:
                # Output exists - convert to string if it's not already
                # This handles dicts, lists, numbers, booleans, etc.
                output = str(output)
            success = result.get("success", True)
        else:
            # Result is not a dict, convert to string
            output = str(result) if result is not None else ""
            success = True

        # Truncate output for display
        truncated = self._truncate_lines(output, self.show_tool_lines)

        # Choose icon based on success
        icon = "✅" if success else "❌"

        if agent_name:
            # Sub-agent tool result: status line cyan, 4-space indent, box drawing
            print(f"    \033[36m└─ {icon} [{agent_name}] Tool result: {tool_name}\033[0m")
            print(f"       \033[2m{truncated}\033[0m\n")
        else:
            # Parent tool result: status line cyan
            print(f"\033[36m{icon} Tool result: {tool_name}\033[0m")
            print(f"   \033[2m{truncated}\033[0m\n")  # Dim text

        return HookResult(action="continue")

    async def handle_llm_response(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Buffer token usage from LLM response for later display.

        Token usage is buffered here and displayed by handle_prompt_complete()
        after the response content is shown to the user.

        Args:
            _event: Event name (llm:response) - unused
            data: Event data containing usage information

        Returns:
            HookResult with action="continue"
        """
        # Only buffer if configured and response was successful
        if not self.show_token_usage:
            return HookResult(action="continue")

        status = data.get("status", "ok")
        if status != "ok":
            return HookResult(action="continue")

        # Extract usage data
        usage = data.get("usage", {})

        if not usage:
            return HookResult(action="continue")

        # Buffer token usage for display after content
        self.buffered_token_usage = {
            "input": usage.get("input", 0),
            "output": usage.get("output", 0),
        }

        return HookResult(action="continue")

    async def handle_prompt_complete(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display buffered token usage after prompt completes.

        Listens for canonical prompt:complete kernel event, which apps
        emit after displaying response content to user.

        Args:
            _event: Event name (prompt:complete) - canonical kernel event
            data: Event data containing prompt and response

        Returns:
            HookResult with action="continue"
        """

        # Display buffered token usage if available
        if self.buffered_token_usage:
            input_tokens = self.buffered_token_usage["input"]
            output_tokens = self.buffered_token_usage["output"]
            total_tokens = input_tokens + output_tokens

            # Format numbers with commas for readability
            input_str = f"{input_tokens:,}"
            output_str = f"{output_tokens:,}"
            total_str = f"{total_tokens:,}"

            # Display token usage with blank line before for visual separation
            print()  # Blank line before token usage
            print("\033[2m│  📊 Token Usage\033[0m")
            print(f"\033[2m└─ Input: {input_str} | Output: {output_str} | Total: {total_str}\033[0m")

            # Clear buffer
            self.buffered_token_usage = None

        return HookResult(action="continue")

    def _truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to max_lines with ellipsis.

        Handles both multi-line text and single-line output (like dicts).
        For single lines over 200 chars, truncates with character limit.

        Args:
            text: Text to truncate (may be any type despite type hint)
            max_lines: Maximum number of lines to show

        Returns:
            Truncated text with ellipsis if needed
        """
        # Defensive: ensure text is actually a string before any operations
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if not text:
            return "(empty)"

        lines = text.split("\n")

        # If it's a single line over 200 chars, truncate by character
        if len(lines) == 1 and len(text) > 200:
            return text[:200] + f"... ({len(text) - 200} more chars)"

        # Multi-line: truncate by line count
        if len(lines) <= max_lines:
            return text

        # Truncate and add indicator
        truncated = lines[:max_lines]
        remaining = len(lines) - max_lines
        truncated.append(f"... ({remaining} more lines)")
        return "\n".join(truncated)


def _flatten_reasoning_block(block: dict[str, Any]) -> str:
    """Flatten OpenAI reasoning block structures into plain text."""
    fragments: list[str] = []

    def _collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value:
                fragments.append(value)
            return
        if isinstance(value, dict):
            _collect(value.get("text"))
            _collect(value.get("thinking"))
            _collect(value.get("summary"))
            _collect(value.get("content"))
            return
        if isinstance(value, list):
            for item in value:
                _collect(item)
            return
        text_attr = getattr(value, "text", None)
        if isinstance(text_attr, str) and text_attr:
            fragments.append(text_attr)

    _collect(block.get("thinking"))
    _collect(block.get("text"))
    _collect(block.get("summary"))
    _collect(block.get("content"))

    return "\n".join(fragment for fragment in fragments if fragment)


__all__ = ["mount", "StreamingUIHooks"]
