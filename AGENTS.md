# AGENTS.md

## Lake Merritt AI Evaluation Workbench

You are OpenAI Codex, working in code‑mode.

Project goal: implement Lake Merritt “Eval Packs” architecture (see docs/dev_plan.md).

Always work on ONE numbered task at a time as directed by the user; write tests, make changes, run `pytest`, then commit and raise a PR that references the task id.

Whenever you are in doubt about the task or how it fits into the broader dev plan for the major upgrade your task is part of, always refer to the full dev plan in docs/dev_plan.md for the full context of this project.

### Environment Setup
This project requires Python 3.9+ (lately we run on python-3.13 actually) and defines dependencies in `pyproject.toml`.
We use `uv` for fast, reliable dependency installation.

### Testing Guidelines

**IMPORTANT**: Many tests require API keys that are not available in the CI environment. 
Always run tests with the marker filter to skip API-dependent tests:

```bash
# Run all tests EXCEPT those requiring API keys
pytest -v -m "not requires_api"

# Run only unit tests (recommended for CI)
pytest tests/unit -v -m "not requires_api"

# If you need to run a specific test file
pytest tests/unit/test_exact_match.py -v
```

### IMPORTANT: Testing Protocol
1. NEVER run tests that require API keys
2. ALWAYS use: pytest -v -m "not requires_api"
3. If a test needs internet for pip, that's OK
4. NEVER commit .env files or expose API keys

### Code Style
- Use Black for formatting
- Type hints are required for all new functions
- Docstrings follow Google style

### Common Tasks
- **Install dependencies**: `uv pip install -e ".[test,dev]"`
- **Run safe tests**: `pytest -v -m "not requires_api"`
- **Run a specific scorer test**: `pytest tests/unit/test_exact_match.py -v`
- **Check types**: `mypy core --ignore-missing-imports`
- **Format code**: `black core tests`

### Test Categories
- **Unit tests** (`tests/unit/`): Test individual components in isolation
- **Integration tests** (`tests/integration/`): Test component interactions
- **API tests**: Marked with `@pytest.mark.requires_api` - these need real API credentials

### Important Notes
- Do NOT commit API keys or .env files
- The Streamlit app requires manual testing (not suitable for automated CI)
- Focus test efforts on the `core/` module business logic
- If uv is not available, fallback to regular pip
- Tests marked with `requires_api` will be skipped in CI environments

### Quick Test Commands
```bash
# Before committing - run the safe test suite
pytest -v -m "not requires_api"

# Test a specific module
pytest tests/unit/test_exact_match.py -v

# Run with coverage (excluding API tests)
pytest -v -m "not requires_api" --cov=core --cov-report=term-missing
```
