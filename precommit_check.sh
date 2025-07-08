#!/bin/bash

# Pre-commit status check script
echo "🔍 Checking pre-commit installation status..."
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if pre-commit is installed
if ! command -v pre-commit >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} Pre-commit is not installed"
    echo "Run: pip install pre-commit"
    exit 1
fi

echo -e "${GREEN}✓${NC} Pre-commit is installed"
precommit_version=$(pre-commit --version 2>/dev/null | cut -d' ' -f2)
echo "  Version: $precommit_version"
echo ""

# Check if this is a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} Not a git repository"
    echo "Initialize git with: git init"
    exit 1
fi

echo -e "${GREEN}✓${NC} Git repository detected"
echo ""

# Check if pre-commit hooks are installed
if [ -f ".git/hooks/pre-commit" ]; then
    echo -e "${GREEN}✓${NC} Pre-commit hooks are installed"

    # Check if it's actually a pre-commit hook
    if grep -q "pre-commit" .git/hooks/pre-commit 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Pre-commit hook is active"
    else
        echo -e "${YELLOW}⚠${NC} Pre-commit hook exists but may not be from pre-commit"
    fi
else
    echo -e "${RED}✗${NC} Pre-commit hooks are NOT installed"
    echo "Install with: pre-commit install"
    echo ""
    echo -e "${BLUE}What happens without hooks:${NC}"
    echo "  - Commits will proceed without automatic linting"
    echo "  - You'll need to run linting manually"
    echo "  - Code quality issues might slip through"
    echo ""
    echo -e "${BLUE}To install hooks:${NC}"
    echo "  1. Run: pre-commit install"
    echo "  2. Or use VS Code task: 'Install Pre-commit Hooks'"
    echo ""
    exit 1
fi

echo ""
echo -e "${BLUE}Pre-commit configuration:${NC}"
if [ -f ".pre-commit-config.yaml" ]; then
    echo -e "${GREEN}✓${NC} Configuration file found: .pre-commit-config.yaml"

    # Show hooks that will run
    echo ""
    echo -e "${BLUE}Hooks that will run on commit:${NC}"
    echo "  - Trailing whitespace removal"
    echo "  - End-of-file fixing"
    echo "  - YAML syntax checking"
    echo "  - Large file detection"
    echo "  - Merge conflict detection"
    echo "  - Debug statement detection"
    echo "  - Ruff linting and formatting"
    echo "  - Pylint analysis (app.py only)"
else
    echo -e "${RED}✗${NC} Configuration file not found"
fi

echo ""
echo -e "${BLUE}Test the setup:${NC}"
echo "  Run: pre-commit run --all-files"
echo "  Or use VS Code task: 'Run Pre-commit on All Files'"
echo ""

# Show recent commits to demonstrate hooks are working
echo -e "${BLUE}Recent commits:${NC}"
if git log --oneline -5 2>/dev/null; then
    echo ""
    echo -e "${GREEN}✅ Pre-commit is ready to protect your commits!${NC}"
else
    echo "No commits yet - make your first commit to test pre-commit hooks"
fi
