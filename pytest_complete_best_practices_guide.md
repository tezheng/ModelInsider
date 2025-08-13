# Complete Pytest Best Practices Guide (2025)

A comprehensive guide covering all aspects of pytest, from basic usage to advanced patterns and project organization.

## Table of Contents

1. [Project Structure & Organization](#project-structure--organization)
2. [Test Discovery & Naming Conventions](#test-discovery--naming-conventions)
3. [Fixtures: The Heart of Pytest](#fixtures-the-heart-of-pytest)
4. [Markers & Test Categorization](#markers--test-categorization)
5. [Parametrization: Data-Driven Testing](#parametrization-data-driven-testing)
6. [Assertions & Error Handling](#assertions--error-handling)
7. [Configuration & Settings](#configuration--settings)
8. [Conftest.py: Shared Test Logic](#conftest-py-shared-test-logic)
9. [Mocking & Monkeypatching](#mocking--monkeypatching)
10. [Performance & Optimization](#performance--optimization)
11. [CI/CD Integration](#cicd-integration)
12. [Plugin Ecosystem](#plugin-ecosystem)
13. [Common Patterns & Anti-Patterns](#common-patterns--anti-patterns)
14. [Debugging & Troubleshooting](#debugging--troubleshooting)
15. [Best Practices Checklist](#best-practices-checklist)

---

## Project Structure & Organization

### Recommended Layout

```
project/
├── src/                        # Source code
│   └── myproject/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── engine.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── helpers.py
│       └── api/
│           ├── __init__.py
│           └── endpoints.py
├── tests/                      # Test directory
│   ├── __init__.py            # Makes tests a package (optional - see note below)
│   ├── conftest.py            # Shared fixtures and configuration
│   ├── unit/                  # Unit tests
│   │   ├── __init__.py
│   │   ├── test_engine.py
│   │   └── test_helpers.py
│   ├── integration/           # Integration tests
│   │   ├── __init__.py
│   │   └── test_api.py
│   ├── e2e/                   # End-to-end tests
│   │   ├── __init__.py
│   │   └── test_workflows.py
│   └── fixtures/              # Shared test data/utilities
│       ├── __init__.py
│       └── test_data.py
├── pyproject.toml            # Modern Python project config (preferred)
├── pytest.ini                 # Legacy pytest configuration (avoid)
├── .coveragerc               # Coverage configuration
└── tox.ini                   # Multiple environment testing
```

### Key Principles

1. **Mirror Source Structure**: Test directory structure should mirror your source code
2. **Separate Test Types**: Keep unit, integration, and e2e tests in separate directories
3. **`__init__.py` in Tests**: Optional - use only when you need to import between test modules (see detailed explanation below)
4. **Centralize Fixtures**: Use `conftest.py` for shared fixtures

### Should You Use `__init__.py` in Test Directories?

The use of `__init__.py` in test directories is **optional** and depends on your specific needs:

#### When to USE `__init__.py` in tests ✅

1. **Cross-test imports**: When you need to import helper functions or classes between test modules
   ```python
   # tests/unit/test_user.py
   from tests.helpers.factories import UserFactory  # Requires __init__.py
   ```

2. **Test utilities as a package**: When you have reusable test utilities that need to be imported
   ```
   tests/
   ├── __init__.py
   ├── helpers/
   │   ├── __init__.py
   │   ├── factories.py
   │   └── assertions.py
   ```

3. **Namespace packages**: When you need to avoid naming conflicts with application modules
   ```python
   # Disambiguates tests.models from myapp.models
   from tests.models import TestUser
   from myapp.models import User
   ```

#### When NOT to use `__init__.py` in tests ❌

1. **Simple test structures**: Most projects don't need it - pytest discovers tests without it
2. **Import mode conflicts**: Can cause issues with pytest's import mechanisms
3. **Accidental test collection**: May cause pytest to collect non-test files

#### Best Practice Recommendation

**Default approach**: Start WITHOUT `__init__.py` in test directories. Only add it when you have a specific need for cross-test imports or test utilities.

```
# Recommended minimal structure
tests/
├── conftest.py          # Shared fixtures (no __init__.py needed)
├── unit/
│   └── test_models.py   # Tests work without __init__.py
└── integration/
    └── test_api.py
```

#### pytest.ini Configuration for Import Issues

If you encounter import issues, configure pytest's import mode instead of adding `__init__.py`:

```ini
# pytest.ini
[pytest]
# Use importlib mode for better import handling
import_mode = importlib

# Or use prepend mode (default)
import_mode = prepend
```

### Alternative Layouts

#### Tests Outside Application Code (Recommended)
```
project/
├── src/myproject/
└── tests/
```

#### Tests as Part of Application (Less Common)
```
project/
└── myproject/
    ├── core/
    │   ├── engine.py
    │   └── tests/
    │       └── test_engine.py
    └── utils/
        ├── helpers.py
        └── tests/
            └── test_helpers.py
```

---

## Test Discovery & Naming Conventions

### Default Discovery Rules

Pytest automatically discovers tests following these patterns:

- **Test files**: `test_*.py` or `*_test.py`
- **Test classes**: `Test*` (must not have an `__init__` method)
- **Test functions**: `test_*`
- **Test methods**: `test_*` inside `Test*` classes

### Naming Best Practices

```python
# ❌ Bad: Unclear test names
def test_1():
    pass

def test_user():
    pass

def test_function():
    pass

# ✅ Good: Descriptive test names
def test_user_creation_with_valid_email():
    """Test that a user can be created with a valid email address."""
    pass

def test_user_creation_fails_with_duplicate_email():
    """Test that creating a user with an existing email raises an error."""
    pass

def test_password_reset_sends_email_to_registered_user():
    """Test that password reset email is sent to registered users."""
    pass
```

### Test Class Organization

```python
class TestUserAuthentication:
    """Test cases for user authentication functionality."""
    
    def test_login_with_valid_credentials_returns_token(self):
        """Test successful login returns authentication token."""
        pass
    
    def test_login_with_invalid_password_returns_401(self):
        """Test login with wrong password returns 401 status."""
        pass
    
    def test_login_with_nonexistent_user_returns_404(self):
        """Test login with non-existent user returns 404 status."""
        pass
```

### Custom Discovery Configuration

```ini
# pytest.ini
[pytest]
# Custom patterns for test discovery
python_files = test_*.py check_*.py
python_classes = Test* Check*
python_functions = test_* check_*

# Ignore specific directories
norecursedirs = .git .tox build dist *.egg
```

---

## Fixtures: The Heart of Pytest

### Basic Fixture Concepts

```python
import pytest

# Simple fixture
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"name": "John", "age": 30}

# Fixture with teardown
@pytest.fixture
def database_connection():
    """Create database connection and clean up after test."""
    conn = create_connection()
    yield conn  # This is where the test runs
    conn.close()  # Teardown happens after test

# Using fixtures in tests
def test_user_data(sample_data):
    assert sample_data["name"] == "John"
```

### Fixture Scopes

```python
# Function scope (default) - run once per test function
@pytest.fixture(scope="function")
def function_resource():
    return expensive_setup()

# Class scope - run once per test class
@pytest.fixture(scope="class")
def class_resource():
    return expensive_setup()

# Module scope - run once per module
@pytest.fixture(scope="module")
def module_resource():
    return expensive_setup()

# Session scope - run once per test session
@pytest.fixture(scope="session")
def session_resource():
    return expensive_setup()

# Package scope - run once per package
@pytest.fixture(scope="package")
def package_resource():
    return expensive_setup()
```

### Advanced Fixture Patterns

#### Factory Fixtures
```python
@pytest.fixture
def make_user():
    """Factory fixture for creating users."""
    created_users = []
    
    def _make_user(name, email=None):
        user = User(name=name, email=email or f"{name}@example.com")
        created_users.append(user)
        return user
    
    yield _make_user
    
    # Cleanup all created users
    for user in created_users:
        user.delete()

def test_user_interactions(make_user):
    alice = make_user("alice")
    bob = make_user("bob", "bob@company.com")
    assert alice.can_message(bob)
```

#### Parametrized Fixtures
```python
@pytest.fixture(params=["sqlite", "postgresql", "mysql"])
def database(request):
    """Test with multiple database backends."""
    return setup_database(request.param)

def test_query_performance(database):
    # This test runs three times, once for each database
    result = database.execute("SELECT * FROM users")
    assert result.execution_time < 100  # ms
```

#### Dynamic Fixture Scope
```python
def determine_scope(fixture_name, config):
    """Dynamically determine fixture scope based on config."""
    if config.getoption("--quick", None):
        return "session"  # Reuse fixtures for speed
    return "function"    # Fresh fixtures for isolation

@pytest.fixture(scope=determine_scope)
def api_client():
    return APIClient()
```

#### Fixture Dependencies
```python
@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def database(config):
    return Database(config["db_url"])

@pytest.fixture
def api_client(config, database):
    # Fixtures can depend on other fixtures
    return APIClient(config["api_url"], database)
```

### Auto-use Fixtures

```python
@pytest.fixture(autouse=True)
def reset_global_state():
    """Automatically run before each test without explicit request."""
    clear_caches()
    reset_singletons()
    yield
    # Cleanup happens after test

@pytest.fixture(autouse=True, scope="session")
def configure_test_environment():
    """Set up test environment once for entire session."""
    os.environ["TESTING"] = "true"
    configure_logging("debug")
```

### Fixture Finalization

```python
@pytest.fixture
def resource_with_finalizer(request):
    """Using request.addfinalizer for cleanup."""
    resource = acquire_resource()
    
    def cleanup():
        release_resource(resource)
    
    request.addfinalizer(cleanup)
    return resource

# Equivalent using yield
@pytest.fixture
def resource_with_yield():
    """Using yield for cleanup (preferred)."""
    resource = acquire_resource()
    yield resource
    release_resource(resource)
```

---

## Markers & Test Categorization

### Built-in Markers

```python
import pytest
import sys

# Skip marker
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass

# Conditional skip
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_pattern_matching():
    match value:
        case 1: return "one"
        case _: return "other"

# Expected failure
@pytest.mark.xfail(reason="Known bug #123")
def test_known_issue():
    assert buggy_function() == expected_value

# Strict xfail - fails if test passes
@pytest.mark.xfail(strict=True, reason="Should be fixed in v2.0")
def test_upcoming_fix():
    assert new_feature() == expected

# Platform-specific tests
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only test")
def test_linux_specific():
    pass

# Import skip
def test_optional_dependency():
    numpy = pytest.importorskip("numpy", minversion="1.20.0")
    # Test only runs if numpy >= 1.20.0 is available
```

### Custom Markers

```ini
# pytest.ini - Register custom markers
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    smoke: core functionality that must always work
    integration: requires external services
    unit: fast isolated unit tests
    flaky: tests that occasionally fail
    requires_db: tests that need database access
    requires_network: tests that need network access
```

```python
# Using custom markers
@pytest.mark.slow
@pytest.mark.integration
def test_full_workflow():
    """Test complete user workflow with external services."""
    pass

@pytest.mark.smoke
def test_critical_functionality():
    """Test that must always pass."""
    pass

# Multiple markers
@pytest.mark.unit
@pytest.mark.smoke
def test_core_logic():
    """Fast unit test for critical functionality."""
    pass
```

### Marker Expressions

```bash
# Run only smoke tests
pytest -m smoke

# Run all tests except slow ones
pytest -m "not slow"

# Complex expressions
pytest -m "smoke and not slow"
pytest -m "(unit or integration) and not flaky"

# List all markers
pytest --markers
```

### Applying Markers Dynamically

```python
# In conftest.py
def pytest_collection_modifyitems(items):
    """Dynamically add markers during collection."""
    for item in items:
        # Add marker based on test location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add marker based on test name
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
```

---

## Parametrization: Data-Driven Testing

### Basic Parametrization

```python
import pytest

# Single parameter
@pytest.mark.parametrize("number", [1, 2, 3, 4, 5])
def test_square(number):
    assert number ** 2 == number * number

# Multiple parameters
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (-2, 4),
])
def test_square_with_expected(input, expected):
    assert input ** 2 == expected

# Using test IDs for better output
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (-2, 4),
], ids=["positive_2", "positive_3", "negative_2"])
def test_square_with_ids(input, expected):
    assert input ** 2 == expected

# ID function
def idfn(val):
    return f"num_{val}"

@pytest.mark.parametrize("number", [1, 2, 3], ids=idfn)
def test_with_id_function(number):
    assert number > 0
```

### Advanced Parametrization

```python
# Nested parametrization
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiplication(x, y):
    # Runs 4 times: (1,10), (1,20), (2,10), (2,20)
    assert x * y == y * x

# Parametrize with marks
@pytest.mark.parametrize("test_input,expected", [
    ("3+5", 8),
    ("2+4", 6),
    pytest.param("6*9", 42, marks=pytest.mark.xfail(reason="Hitchhiker's joke")),
    pytest.param("1/0", 0, marks=pytest.mark.skip(reason="Division by zero")),
])
def test_eval(test_input, expected):
    assert eval(test_input) == expected

# Indirect parametrization (parametrize fixtures)
@pytest.mark.parametrize("db_name", ["sqlite", "postgres"], indirect=True)
def test_database_operations(db_name):
    # db_name fixture receives the parameter value
    assert db_name.connect()
```

### Parametrization Patterns

```python
# Test class parametrization
@pytest.mark.parametrize("browser", ["chrome", "firefox", "safari"])
class TestWebApplication:
    def test_login(self, browser):
        # Each test method runs with each browser
        pass
    
    def test_search(self, browser):
        pass

# Dynamic parametrization
def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests."""
    if "dynamic_value" in metafunc.fixturenames:
        values = load_test_values_from_file()
        metafunc.parametrize("dynamic_value", values)

# Parametrization from fixtures
@pytest.fixture(params=["admin", "user", "guest"])
def user_role(request):
    return create_user_with_role(request.param)

def test_permissions(user_role):
    # Test runs for each user role
    assert user_role.can_access("/dashboard") == user_role.is_admin
```

---

## Assertions & Error Handling

### Enhanced Assertions

```python
# Pytest rewrites assert statements for better output
def test_assertion_introspection():
    data = {"name": "Alice", "items": [1, 2, 3]}
    # Pytest shows detailed diff on failure
    assert data == {"name": "Bob", "items": [1, 2, 3]}

# Custom assertion messages
def test_with_message():
    result = complex_calculation()
    assert result > 0, f"Expected positive result, got {result}"
```

### Exception Testing

```python
import pytest

# Basic exception testing
def test_raises_exception():
    with pytest.raises(ValueError):
        raise ValueError("Invalid value")

# Check exception message
def test_exception_message():
    with pytest.raises(ValueError, match="Invalid.*value"):
        raise ValueError("Invalid value provided")

# Access exception info
def test_exception_info():
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("test error")
    
    assert str(exc_info.value) == "test error"
    assert exc_info.type == ValueError

# Test multiple exceptions (ExceptionGroup)
def test_exception_group():
    with pytest.raises(ExceptionGroup) as exc_info:
        raise ExceptionGroup("errors", [
            ValueError("error 1"),
            TypeError("error 2")
        ])
    
    assert len(exc_info.value.exceptions) == 2
```

### Warning Testing

```python
import warnings
import pytest

def test_warns():
    with pytest.warns(UserWarning):
        warnings.warn("This is a warning", UserWarning)

def test_warns_with_match():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        warnings.warn("This function is deprecated", DeprecationWarning)

def test_no_warnings():
    # Ensure no warnings are raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clean_function()  # Should not raise any warnings
```

### Approximate Comparisons

```python
import pytest

def test_float_comparison():
    assert 0.1 + 0.2 == pytest.approx(0.3)

def test_list_approximate():
    assert [0.1 + 0.2, 0.2 + 0.4] == pytest.approx([0.3, 0.6])

def test_dict_approximate():
    assert {"a": 0.1 + 0.2} == pytest.approx({"a": 0.3})

# Custom tolerance
def test_custom_tolerance():
    assert 1.0001 == pytest.approx(1.0, rel=1e-3)
    assert 1.0001 == pytest.approx(1.0, abs=1e-3)
```

---

## Configuration & Settings

### pyproject.toml Configuration (Recommended)

Using `pyproject.toml` is the modern, preferred approach for Python project configuration. It consolidates all project metadata and tool configurations in one place.

```toml
# pyproject.toml
[tool.pytest.ini_options]
# Minimum pytest version
minversion = "7.0"

# Default command line options
addopts = [
    "--strict-markers",      # Fail on unknown markers
    "--strict-config",       # Fail on config errors
    "--verbose",             # Verbose output
    "-ra",                   # Show all test outcomes
    "--cov=myproject",       # Coverage for your project
    "--cov-report=html",     # HTML coverage report
    "--cov-report=term-missing",  # Terminal report with missing lines
]

# Test discovery
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*", "*Tests"]
python_functions = ["test_*"]

# Python path configuration
pythonpath = ["src"]

# Import mode (importlib is recommended for most projects)
import_mode = "importlib"

# Custom markers registration
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: requires external services",
    "unit: fast isolated unit tests",
    "smoke: core functionality that must always work",
    "flaky: tests that occasionally fail",
    "requires_network: tests that need network access",
]

# Output configuration
console_output_style = "progress"

# Directories to ignore
norecursedirs = [".git", ".tox", "dist", "build", "*.egg", "__pycache__"]

# Logging configuration
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"

# Warning filters
filterwarnings = [
    "error",                          # Turn warnings into errors
    "ignore::UserWarning",            # Ignore user warnings
    "ignore::DeprecationWarning",     # Ignore deprecation warnings
    "default:.*deprecated.*:DeprecationWarning",  # Show deprecation warnings with "deprecated" in message
]

# Test timeout (requires pytest-timeout)
timeout = 300
timeout_method = "thread"

# Strict xfail
xfail_strict = true

# Asyncio configuration (requires pytest-asyncio)
asyncio_mode = "auto"

# Coverage configuration (can also be in [tool.coverage])
[tool.coverage.run]
source = ["myproject"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/.venv/*",
    "*/migrations/*",
    "*/__pycache__/*",
    "*/.pytest_cache/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if typing.TYPE_CHECKING:",
]

[tool.coverage.html]
directory = "htmlcov"

[tool.coverage.xml]
output = "coverage.xml"
```

### Complete pyproject.toml Example

Here's a complete `pyproject.toml` that includes project metadata along with pytest configuration:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "1.0.0"
description = "My awesome project"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"},
]
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-asyncio>=0.21.0",
    "pytest-timeout>=2.1.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/username/myproject"
Documentation = "https://myproject.readthedocs.io"
Repository = "https://github.com/username/myproject.git"
Issues = "https://github.com/username/myproject/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
# ... (configuration from above)

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]
include = '\.pyi?$'

[tool.ruff]
line-length = 88
target-version = "py38"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Migration from pytest.ini to pyproject.toml

If you have an existing `pytest.ini`, here's how to migrate:

```ini
# OLD: pytest.ini
[pytest]
markers =
    slow: slow tests
testpaths = tests
```

Becomes:

```toml
# NEW: pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: slow tests",
]
testpaths = ["tests"]
```

### Legacy pytest.ini (Not Recommended)

While `pytest.ini` still works, it's considered legacy. Use `pyproject.toml` instead for these benefits:
- Single configuration file for all Python tools
- Better IDE support
- TOML format is more readable
- Standardized by PEP 518 and PEP 621

### Command Line Configuration

```bash
# Common command line options
pytest -v                    # Verbose output
pytest -q                    # Quiet output
pytest -s                    # No capture, show print statements
pytest -x                    # Stop on first failure
pytest --maxfail=3          # Stop after 3 failures
pytest -k "user"            # Run tests matching "user"
pytest -m "not slow"        # Run tests not marked as slow
pytest --lf                 # Run last failed tests
pytest --ff                 # Run failed tests first
pytest --tb=short           # Short traceback format
pytest --tb=no              # No traceback
pytest --setup-show         # Show fixture setup/teardown
pytest --fixtures           # Show available fixtures
pytest --markers            # Show available markers
pytest --collect-only       # Only collect tests, don't run
pytest --cache-clear        # Clear cache before run
pytest --doctest-modules    # Run doctests
pytest --cov=myproject      # Coverage report
pytest --cov-report=html    # HTML coverage report
pytest --durations=10       # Show 10 slowest tests
pytest --pdb                # Drop to debugger on failure
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb  # Use IPython debugger
```

---

## Conftest.py: Shared Test Logic

### Fixture Sharing

```python
# tests/conftest.py - Available to all tests
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Shared test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def temp_dir():
    """Create temporary directory for test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

# tests/unit/conftest.py - Available to unit tests only
@pytest.fixture
def mock_database():
    """Mock database for unit tests."""
    return MockDatabase()

# tests/integration/conftest.py - Available to integration tests only
@pytest.fixture(scope="module")
def real_database():
    """Real database connection for integration tests."""
    db = Database()
    yield db
    db.cleanup()
```

### Hooks in conftest.py

```python
# Modify test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test file location
    for item in items:
        # Add markers based on location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Skip tests based on environment
        if "requires_gpu" in item.keywords and not has_gpu():
            item.add_marker(pytest.mark.skip(reason="GPU not available"))

# Custom command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )

# Configure based on options
def pytest_configure(config):
    """Configure pytest based on command line options."""
    if config.getoption("--run-slow"):
        config.option.markexpr = "slow"

# Custom markers registration
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
```

### Plugin Registration

```python
# Register external plugins
pytest_plugins = [
    "myproject.testing.fixtures",
    "myproject.testing.helpers",
]

# Conditional plugin loading
import sys
if sys.platform.startswith("win"):
    pytest_plugins.append("myproject.testing.windows")
```

---

## Mocking & Monkeypatching

### Using pytest-mock

```python
# Install: pip install pytest-mock

def test_with_mock(mocker):
    """Using pytest-mock plugin."""
    # Mock a module function
    mock_func = mocker.patch("mymodule.function")
    mock_func.return_value = 42
    
    # Mock an object method
    mock_method = mocker.patch.object(MyClass, "method")
    mock_method.return_value = "mocked"
    
    # Spy on a function
    spy = mocker.spy(mymodule, "function")
    mymodule.function()
    spy.assert_called_once()

# Using side effects
def test_side_effects(mocker):
    mock = mocker.patch("mymodule.function")
    mock.side_effect = [1, 2, 3]  # Returns different values each call
    
    assert mymodule.function() == 1
    assert mymodule.function() == 2
    assert mymodule.function() == 3

# Mock with exceptions
def test_mock_exception(mocker):
    mock = mocker.patch("mymodule.function")
    mock.side_effect = ValueError("Error!")
    
    with pytest.raises(ValueError):
        mymodule.function()
```

### Monkeypatch

```python
def test_monkeypatch_env(monkeypatch):
    """Monkeypatch environment variables."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.delenv("OLD_VAR", raising=False)
    
    assert os.environ["API_KEY"] == "test-key"
    assert "OLD_VAR" not in os.environ

def test_monkeypatch_attribute(monkeypatch):
    """Monkeypatch object attributes."""
    class MyClass:
        value = 10
    
    obj = MyClass()
    monkeypatch.setattr(obj, "value", 20)
    assert obj.value == 20

def test_monkeypatch_module(monkeypatch):
    """Monkeypatch module functions."""
    import time
    
    def mock_time():
        return 123456.0
    
    monkeypatch.setattr(time, "time", mock_time)
    assert time.time() == 123456.0

def test_monkeypatch_dict(monkeypatch):
    """Monkeypatch dictionary items."""
    config = {"url": "production.com"}
    monkeypatch.setitem(config, "url", "test.com")
    assert config["url"] == "test.com"
```

### Advanced Mocking Patterns

```python
# Context manager mocking
def test_context_manager(mocker):
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__.return_value = "resource"
    mock_cm.__exit__.return_value = None
    
    mocker.patch("mymodule.get_resource", return_value=mock_cm)
    
    with mymodule.get_resource() as resource:
        assert resource == "resource"
    
    mock_cm.__enter__.assert_called_once()
    mock_cm.__exit__.assert_called_once()

# Property mocking
def test_property_mock(mocker):
    mock_property = mocker.PropertyMock(return_value=42)
    mocker.patch("mymodule.MyClass.my_property", new_callable=mock_property)
    
    obj = mymodule.MyClass()
    assert obj.my_property == 42
    mock_property.assert_called_once()

# Async mocking
async def test_async_mock(mocker):
    mock_async = mocker.AsyncMock(return_value="async result")
    mocker.patch("mymodule.async_function", mock_async)
    
    result = await mymodule.async_function()
    assert result == "async result"
    mock_async.assert_awaited_once()
```

---

## Performance & Optimization

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto          # Use all available CPUs
pytest -n 4            # Use 4 workers
pytest -n 2 --dist loadscope  # Group by module
pytest -n 2 --dist loadfile   # Group by file
```

### Test Duration Analysis

```python
# Show test durations
pytest --durations=10   # Show 10 slowest tests
pytest --durations=0    # Show all test durations

# In conftest.py - Custom timing
import time

@pytest.fixture(autouse=True)
def measure_test_time(request):
    start = time.time()
    yield
    duration = time.time() - start
    print(f"\n{request.node.name} took {duration:.2f}s")
```

### Caching

```python
# Using pytest cache
def test_expensive_computation(cache):
    # Check cache
    result = cache.get("computation_result", None)
    if result is None:
        # Compute and cache
        result = expensive_computation()
        cache.set("computation_result", result)
    
    assert result == expected_value

# Cache command line
pytest --cache-show     # Show cache contents
pytest --cache-clear    # Clear cache
```

### Fixture Optimization

```python
# Reuse expensive fixtures with broader scope
@pytest.fixture(scope="session")
def expensive_resource():
    """Create once, use many times."""
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()

# Lazy fixture creation
@pytest.fixture
def maybe_expensive():
    """Only created if actually used by test."""
    return ExpensiveObject()

# Fixture factories for controlled creation
@pytest.fixture
def resource_factory():
    resources = []
    
    def _make_resource(**kwargs):
        resource = Resource(**kwargs)
        resources.append(resource)
        return resource
    
    yield _make_resource
    
    # Cleanup all at once
    for resource in resources:
        resource.cleanup()
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run tests
      run: |
        pytest -v --cov=myproject --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Stages

```yaml
# Multi-stage testing
stages:
  - quick-tests
  - full-tests
  - integration-tests

quick-tests:
  script:
    - pytest -m "unit and not slow" --fail-fast

full-tests:
  script:
    - pytest -m "not integration"

integration-tests:
  script:
    - pytest -m integration
  only:
    - main
    - merge_requests
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = myproject
omit = 
    */tests/*
    */venv/*
    */migrations/*
    */__init__.py

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov

[xml]
output = coverage.xml
```

---

## Plugin Ecosystem

### Essential Plugins

```bash
# Coverage
pip install pytest-cov

# Parallel execution
pip install pytest-xdist

# Mocking
pip install pytest-mock

# Timeout
pip install pytest-timeout

# HTML reports
pip install pytest-html

# BDD
pip install pytest-bdd

# Benchmarking
pip install pytest-benchmark

# Django
pip install pytest-django

# Asyncio
pip install pytest-asyncio

# Flake8 integration
pip install pytest-flake8

# Order randomization
pip install pytest-randomly
```

### Plugin Usage Examples

```python
# pytest-timeout
@pytest.mark.timeout(10)  # 10 second timeout
def test_slow_operation():
    perform_slow_operation()

# pytest-benchmark
def test_performance(benchmark):
    result = benchmark(my_function, arg1, arg2)
    assert result == expected

# pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected

# pytest-randomly (randomize test order)
# Just install and it works automatically
# Use --randomly-seed=1234 to reproduce order
```

---

## Common Patterns & Anti-Patterns

### Patterns ✅

```python
# Good: Descriptive test names
def test_user_registration_sends_welcome_email():
    pass

# Good: Focused tests
def test_calculate_tax_for_standard_rate():
    income = 50000
    assert calculate_tax(income) == 10000

# Good: Using fixtures for setup
@pytest.fixture
def authenticated_client(client, user):
    client.login(username=user.username, password="password")
    return client

# Good: Parametrize instead of loops
@pytest.mark.parametrize("value,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_square(value, expected):
    assert value ** 2 == expected

# Good: Clear test structure (Arrange-Act-Assert)
def test_user_creation():
    # Arrange
    data = {"username": "john", "email": "john@example.com"}
    
    # Act
    user = User.create(**data)
    
    # Assert
    assert user.username == "john"
    assert user.email == "john@example.com"
```

### Anti-Patterns ❌

```python
# Bad: Test doing too much
def test_everything():
    user = create_user()
    post = create_post(user)
    comment = create_comment(post)
    assert user.is_active
    assert post.author == user
    assert comment.post == post
    # Too many things tested at once

# Bad: Modifying global state
def test_with_global_state():
    global CONFIG
    CONFIG["debug"] = True  # Don't modify globals
    assert my_function() == expected

# Bad: Tests depending on order
def test_first():
    global shared_data
    shared_data = setup_data()

def test_second():
    # Depends on test_first running first
    assert shared_data.value == expected

# Bad: Catching all exceptions
def test_broad_exception():
    try:
        risky_operation()
    except Exception:  # Too broad
        pass  # Test passes even if unexpected error

# Bad: No assertion
def test_without_assertion():
    result = my_function()
    # No assert - test always passes
```

---

## Debugging & Troubleshooting

### Debugging Techniques

```python
# Drop into debugger on failure
pytest --pdb

# Drop into IPython debugger
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb

# Set breakpoint in code
def test_debug():
    value = calculate()
    import pdb; pdb.set_trace()  # or breakpoint() in Python 3.7+
    assert value == expected

# Print debugging (use -s flag)
def test_with_print():
    print("Debug info:", value)  # Visible with pytest -s
    assert value == expected

# Capture logs
def test_with_logging(caplog):
    with caplog.at_level(logging.INFO):
        my_function()
    assert "Expected message" in caplog.text

# Detailed failure info
pytest -vv  # Very verbose
pytest --tb=short  # Short traceback
pytest --tb=line   # One line per failure
pytest --tb=no     # No traceback
```

### Common Issues & Solutions

```python
# Issue: Import errors
# Solution: Check PYTHONPATH and use --import-mode
pytest --import-mode=importlib

# Issue: Fixture not found
# Solution: Check scope and conftest.py location
pytest --fixtures  # List available fixtures

# Issue: Tests not discovered
# Solution: Check naming conventions
pytest --collect-only  # See what's collected

# Issue: Flaky tests
# Solution: Use pytest-rerunfailures
pip install pytest-rerunfailures
pytest --reruns 3 --reruns-delay 1

# Issue: Test isolation
# Solution: Use fixtures and avoid global state
@pytest.fixture(autouse=True)
def reset_state():
    cleanup_before_test()
    yield
    cleanup_after_test()
```

---

## Best Practices Checklist

### ✅ DO's

1. **Write descriptive test names** that explain what is being tested
2. **Use fixtures** for setup and teardown
3. **Keep tests focused** - one concept per test
4. **Use parametrize** for data-driven tests
5. **Organize tests** to mirror source code structure
6. **Register custom markers** in pytest.ini
7. **Use appropriate scopes** for fixtures
8. **Mock external dependencies** in unit tests
9. **Run fastest tests first** in CI/CD
10. **Use pytest.raises** for exception testing
11. **Document complex test scenarios**
12. **Use tmp_path fixture** for file operations
13. **Configure pytest** in pyproject.toml or pytest.ini
14. **Use pytest plugins** to extend functionality
15. **Profile slow tests** and optimize
16. **Start without `__init__.py`** in test directories - add only when needed

### ❌ DON'Ts

1. **Don't write tests that depend on execution order**
2. **Don't use global state** that affects other tests
3. **Don't catch broad exceptions** without re-raising
4. **Don't hardcode paths** - use fixtures and tmp_path
5. **Don't skip writing tests** for "simple" functions
6. **Don't mix test types** in the same file
7. **Don't use production credentials** in tests
8. **Don't ignore flaky tests** - fix or mark them
9. **Don't write tests without assertions**
10. **Don't duplicate test logic** - use fixtures
11. **Don't test implementation details** - test behavior
12. **Don't use time.sleep** - use proper synchronization
13. **Don't modify source code** for testing - use mocks
14. **Don't run all tests locally** for every change
15. **Don't ignore test warnings** - fix or suppress explicitly
16. **Don't add `__init__.py` to tests by default** - pytest works without it

### Final Recommendations

1. **Start Simple**: Begin with basic tests and add complexity as needed
2. **Test First**: Consider TDD for complex logic
3. **Continuous Integration**: Run tests automatically on every commit
4. **Code Coverage**: Aim for high coverage but focus on critical paths
5. **Performance**: Monitor and optimize test suite performance
6. **Documentation**: Document complex test scenarios and fixtures
7. **Maintenance**: Regularly update and refactor tests
8. **Team Standards**: Establish and follow team testing conventions

Remember: Good tests are as important as good code. They provide confidence, documentation, and safety for refactoring.