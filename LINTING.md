# Code Quality and Linting Setup

This project uses multiple linting and formatting tools to ensure code quality and consistency.

## Tools Included

### 🚀 Ruff v0.12.2 - Ultra-fast Python linter and formatter

- **Purpose**: Primary linter and formatter (replaces many tools)
- **Config**: `pyproject.toml`
- **Features**:
  - Linting with 800+ rules
  - Auto-fixing capabilities
  - Import sorting
  - Code formatting
  - Extremely fast execution

### 🔍 Pylint v3.3.7 - Comprehensive Python code analysis

- **Purpose**: Detailed code quality analysis
- **Config**: `pylintrc`
- **Features**:
  - Code quality metrics
  - Design pattern analysis
  - Bug detection
  - Refactoring suggestions

### 🐍 Flake8 - Style guide enforcement

- **Purpose**: PEP 8 compliance checking
- **Config**: Command-line args in tasks
- **Features**:
  - Style guide enforcement
  - Error detection
  - Legacy support

## VS Code Integration

### Extensions Installed

- `charliermarsh.ruff` - Ruff official extension
- `ms-python.pylint` - Pylint official extension
- `ms-python.flake8` - Flake8 integration
- `ms-python.autopep8` - AutoPEP8 integration

### Settings

- Ruff is set as the primary formatter
- Auto-format on save enabled
- Import organization on save
- Real-time linting feedback

## Available Tasks

Access via Command Palette (`Ctrl+Shift+P`) → "Tasks: Run Task":

1. **Lint Code (Ruff)** - Run Ruff linter
2. **Lint Code (Pylint)** - Run Pylint analysis
3. **Lint Code (Flake8)** - Run Flake8 checks
4. **Format Code (Ruff)** - Format code with Ruff
5. **Fix Code (Ruff)** - Auto-fix issues with Ruff
6. **Check All (Ruff + Pylint)** - Run comprehensive checks
7. **Check Tool Versions** - Verify installed tool versions
8. **Install Pre-commit Hooks** - Install git pre-commit hooks
9. **Run Pre-commit on All Files** - Test pre-commit hooks
10. **Check Pre-commit Status** - Verify pre-commit installation

## Command Line Usage

### Quick Commands

```bash
# Format code with Ruff
ruff format .

# Check and fix issues
ruff check . --fix

# Run Pylint on main file
pylint app.py
```

### Comprehensive Check

```bash
# Run all linting tools
./lint_check.sh

# Check installed versions
./version_check.sh

# Check pre-commit status
./precommit_check.sh

# Test pre-commit hooks
pre-commit run --all-files
```

## Pre-commit Hooks ✅ INSTALLED

Pre-commit hooks are automatically installed when you rebuild your devcontainer! They will run before every commit to ensure code quality.

### What Happens on Every Commit:

1. **Trailing whitespace** - Automatically removed
2. **End-of-file fixing** - Ensures files end with newlines
3. **YAML syntax** - Validates configuration files
4. **Large file detection** - Prevents accidental large file commits
5. **Merge conflicts** - Detects unresolved merge conflicts
6. **Debug statements** - Finds leftover debug code
7. **Ruff linting** - Auto-fixes code issues
8. **Ruff formatting** - Ensures consistent code style
9. **Pylint analysis** - Comprehensive code quality checks

### Manual Installation (if needed):

```bash
# Install pre-commit (already in requirements.txt)
pip install pre-commit

# Install git hooks
pre-commit install

# Test on all files
pre-commit run --all-files
```

### Hooks Included

- Trailing whitespace removal
- End-of-file fixing
- YAML syntax checking
- Large file detection
- Merge conflict detection
- Debug statement detection
- Ruff linting and formatting
- Pylint analysis

## Configuration Files

- `pyproject.toml` - Ruff configuration
- `pylintrc` - Pylint configuration
- `.pre-commit-config.yaml` - Pre-commit hooks
- `.vscode/tasks.json` - VS Code tasks
- `.devcontainer/devcontainer.json` - Development container settings

## Integration with Development Workflow

1. **Real-time**: VS Code provides instant feedback while typing
2. **On Save**: Auto-formatting and import organization
3. **Pre-commit**: Automatic checks before git commits
4. **Manual**: Run comprehensive checks with tasks or scripts
5. **CI/CD**: Can be integrated into GitHub Actions or similar

## Troubleshooting

### Common Issues

- **Extension conflicts**: Disable conflicting Python extensions
- **Path issues**: Tools are installed in dev container paths

### Performance Tips

- Ruff is fastest for most operations
- Use Pylint for detailed analysis (slower but comprehensive)
- Run `./lint_check.sh` for complete validation

## Best Practices

1. **Fix Ruff issues first** - They're usually quick fixes
2. **Address Pylint warnings** - Focus on code quality improvements
3. **Use auto-fix features** - Let tools handle formatting
4. **Regular checks** - Run comprehensive checks before major commits
5. **Customize rules** - Adjust configurations for your project needs
