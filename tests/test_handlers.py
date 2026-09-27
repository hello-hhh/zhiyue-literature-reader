"""test_handlers.py — Unit tests for ui/handlers.py and cli/runner.py.

Tests cover:
  - Pure helper functions in ui/handlers.py (no Streamlit server needed)
  - Import contract: all expected public names are exported
  - cli/runner.py: run_benchmark(), run_agent(), run_chat()

Mock strategy:
  - Pure helper functions are imported and called directly — no Streamlit needed
  - Functions that call Streamlit (render_*, handle_*) are tested by mocking st.*
  - ollama is mocked via conftest.py — no Ollama installation required
  - cli/ functions mock Agent, Benchmarker, and VectorStore via patch
"""

from unittest.mock import MagicMock, patch

# ═══════════════════════════════════════════════════════════════════════════════
# ui/handlers.py — import contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandlersImportContract:
    """Verify that ui/handlers.py exports every function app.py needs."""

    def test_all_public_names_are_importable(self):
        """Every name in __all__ can be imported from ui.handlers."""
        from ui import handlers
        from ui.handlers import __all__ as exported_names
        for name in exported_names:
            assert hasattr(handlers, name), f"ui.handlers is missing exported name: {name}"

    def test_expected_functions_present(self):
        """The handler functions app.py calls are defined in the right modules."""
        from ui.handlers import handle_file_upload, handle_url_ingestion, handle_user_input, render_sidebar
        from ui.renderers import render_chat_history, render_clear_button, render_header, render_mode_selector
        assert callable(render_header)
        assert callable(render_sidebar)
        assert callable(handle_user_input)


# ═══════════════════════════════════════════════════════════════════════════════
# _format_agent_steps_html — pure function, no Streamlit calls
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatAgentStepsHtml:
    """Tests for the private _format_agent_steps_html helper in ui/handlers.py."""

    def _call(self, steps: list) -> str:
        """Import and call the helper directly — no mocking needed."""
        from ui.handlers import _format_agent_steps_html
        return _format_agent_steps_html(steps)

    def test_empty_steps_returns_empty_string(self):
        """No steps → empty string (nothing to render)."""
        assert self._call([]) == ""

    def test_single_step_contains_tool_name(self):
        """A single step includes the tool name and step number in the output."""
        steps = [{'step': 1, 'tool': 'calculator', 'arg': '2 + 2', 'result': '4'}]
        result = self._call(steps)
        assert 'calculator' in result
        assert 'Step 1' in result

    def test_long_arg_is_truncated(self):
        """Arguments longer than 50 characters are shortened with '...'."""
        long_arg = 'x' * 100
        steps = [{'step': 1, 'tool': 'rag_search', 'arg': long_arg, 'result': 'found'}]
        result = self._call(steps)
        assert '...' in result
        # The full 100-character arg should NOT appear in the output
        assert 'x' * 100 not in result

    def test_long_result_is_truncated(self):
        """Results longer than 80 characters are shortened with '...'."""
        long_result = 'y' * 200
        steps = [{'step': 1, 'tool': 'finish', 'arg': 'done', 'result': long_result}]
        result = self._call(steps)
        assert '...' in result

    def test_multiple_steps_all_present(self):
        """Each step in the list appears in the output."""
        steps = [
            {'step': 1, 'tool': 'rag_search', 'arg': 'cats', 'result': 'found cats'},
            {'step': 2, 'tool': 'finish',     'arg': 'done', 'result': 'done'},
        ]
        result = self._call(steps)
        assert 'Step 1' in result
        assert 'Step 2' in result
        assert 'rag_search' in result
        assert 'finish' in result

    def test_output_contains_html_step_div(self):
        """Output uses the CSS class 'step' so the Ocean Blue styles apply."""
        steps = [{'step': 1, 'tool': 'calculator', 'arg': '1+1', 'result': '2'}]
        result = self._call(steps)
        assert 'class="step"' in result


# ═══════════════════════════════════════════════════════════════════════════════
# _pick_avatar — pure function
# ═══════════════════════════════════════════════════════════════════════════════

class TestPickAvatar:
    """Tests for the private _pick_avatar helper in ui/handlers.py."""

    def _call(self, role: str) -> str:
        """Import and call _pick_avatar directly."""
        from ui.renderers import _pick_avatar
        return _pick_avatar(role)

    def test_user_avatar(self):
        """'user' maps to the person emoji."""
        assert self._call('user') == '🧑'

    def test_agent_avatar(self):
        """'agent' maps to the robot emoji."""
        assert self._call('agent') == '🤖'

    def test_assistant_avatar(self):
        """'assistant' maps to the speech bubble emoji."""
        assert self._call('assistant') == '💬'

    def test_unknown_role_returns_default(self):
        """Any unrecognised role falls back to the speech bubble emoji."""
        assert self._call('unknown_role') == '💬'


# ═══════════════════════════════════════════════════════════════════════════════
# cli/runner.py — import contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunnerImportContract:
    """Verify that cli/runner.py exports every function main.py needs."""

    def test_all_runner_functions_importable(self):
        """initialize, run_benchmark, run_agent, run_chat are all importable."""
        from cli.runner import initialize, run_agent, run_benchmark, run_chat
        assert callable(initialize)
        assert callable(run_benchmark)
        assert callable(run_agent)
        assert callable(run_chat)


# ═══════════════════════════════════════════════════════════════════════════════
# cli/runner.py — run_benchmark
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunBenchmark:
    """run_benchmark() should load benchmark docs, create a Benchmarker, and call .run()."""

    def test_calls_benchmarker_run(self):
        """run_benchmark passes loader and store; calls Benchmarker(store).run()."""
        mock_loader            = MagicMock()
        mock_store             = MagicMock()
        mock_benchmarker       = MagicMock()

        # chunk_directory returns empty list so the add_chunks branch is skipped
        mock_loader.chunk_directory.return_value = []

        with patch('src.cli.runner.Benchmarker', return_value=mock_benchmarker) as mock_cls:
            with patch('src.cli.runner.run_tool_benchmarks') as mock_tool_bench:
                from src.cli.runner import run_benchmark
                run_benchmark(mock_loader, mock_store)

        mock_cls.assert_called_once_with(mock_store)
        mock_benchmarker.run.assert_called_once()
        mock_tool_bench.assert_called_once_with(mock_store)


# ═══════════════════════════════════════════════════════════════════════════════
# cli/runner.py — run_agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAgent:
    """run_agent() reads terminal input and calls agent.run() for each task."""

    def test_exits_on_exit_command(self, capsys):
        """Typing 'exit' ends the loop cleanly and prints Goodbye."""
        mock_store = MagicMock()
        with patch('builtins.input', return_value='exit'):
            from cli.runner import run_agent
            run_agent(mock_store)
        assert 'Goodbye' in capsys.readouterr().out

    def test_runs_agent_on_valid_input(self):
        """A real task string is passed to agent.run() before the loop exits."""
        mock_store  = MagicMock()
        mock_agent  = MagicMock()
        mock_agent.run.return_value = {'answer': '42'}
        inputs = iter(['what is the answer?', 'exit'])

        with patch('cli.runner.Agent', return_value=mock_agent), \
             patch('builtins.input', side_effect=inputs):
            from cli.runner import run_agent
            run_agent(mock_store)

        mock_agent.run.assert_called_once_with('what is the answer?')

    def test_skips_empty_input(self):
        """Empty Enter presses are skipped — the loop asks again."""
        mock_store  = MagicMock()
        mock_agent  = MagicMock()
        mock_agent.run.return_value = {'answer': 'ok'}
        # First input is empty, second is real, third exits
        inputs = iter(['', 'do something', 'exit'])

        with patch('cli.runner.Agent', return_value=mock_agent), \
             patch('builtins.input', side_effect=inputs):
            from cli.runner import run_agent
            run_agent(mock_store)

        # Only the real task (not the empty input) should have been run
        mock_agent.run.assert_called_once_with('do something')


# ═══════════════════════════════════════════════════════════════════════════════
# cli/runner.py — run_chat
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunChat:
    """run_chat() calls store.run_pipeline() for regular questions."""

    def test_exits_on_quit_command(self, capsys):
        """Typing 'quit' ends the loop cleanly and prints Goodbye."""
        mock_store = MagicMock()
        with patch('builtins.input', return_value='quit'):
            from cli.runner import run_chat
            run_chat(mock_store)
        assert 'Goodbye' in capsys.readouterr().out

    def test_calls_run_pipeline_for_normal_query(self):
        """A regular question is passed to store.run_pipeline()."""
        mock_store = MagicMock()
        mock_store.run_pipeline.return_value = {
            'response':     'Cats sleep 16 hours.',
            'is_confident': True,
            'best_score':   0.85,
            'query_type':   'factual',
            'queries':      ['how long do cats sleep?'],
            'retrieved':    [],
            'reranked':     [],
        }
        inputs = iter(['how long do cats sleep?', 'exit'])

        with patch('builtins.input', side_effect=inputs):
            from cli.runner import run_chat
            run_chat(mock_store)

        mock_store.run_pipeline.assert_called_once_with('how long do cats sleep?')

    def test_agent_prefix_routes_to_agent(self):
        """'agent: <question>' calls Agent.run() instead of run_pipeline()."""
        mock_store  = MagicMock()
        mock_agent  = MagicMock()
        mock_agent.run.return_value = {'answer': 'agent answer'}
        inputs = iter(['agent: what do cats eat?', 'exit'])

        with patch('cli.runner.Agent', return_value=mock_agent), \
             patch('builtins.input', side_effect=inputs):
            from cli.runner import run_chat
            run_chat(mock_store)

        mock_agent.run.assert_called_once_with('what do cats eat?')
