#!/bin/bash

# Version check script for TreePlanter project
# Verifies that the latest versions are installed

echo "🔍 Checking installed versions..."
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Expected versions
EXPECTED_PYTHON="3.13"
EXPECTED_RUFF="0.12.2"
EXPECTED_PYLINT="3.3.7"

echo -e "${BLUE}Expected versions:${NC}"
echo "  Python: $EXPECTED_PYTHON"
echo "  Ruff: $EXPECTED_RUFF"
echo "  Pylint: $EXPECTED_PYLINT"
echo ""

echo -e "${BLUE}Installed versions:${NC}"

# Check Python version
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    if [[ $python_version == $EXPECTED_PYTHON* ]]; then
        echo -e "${GREEN}✓${NC} Python: $python_version"
    else
        echo -e "${YELLOW}⚠${NC} Python: $python_version (expected $EXPECTED_PYTHON)"
    fi
else
    echo -e "${RED}✗${NC} Python not found"
fi

# Check Ruff version
if command -v ruff >/dev/null 2>&1; then
    ruff_version=$(ruff --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
    if [[ $ruff_version == $EXPECTED_RUFF ]]; then
        echo -e "${GREEN}✓${NC} Ruff: $ruff_version"
    else
        echo -e "${YELLOW}⚠${NC} Ruff: $ruff_version (expected $EXPECTED_RUFF)"
    fi
else
    echo -e "${RED}✗${NC} Ruff not found"
fi

# Check Pylint version
if command -v pylint >/dev/null 2>&1; then
    pylint_version=$(pylint --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
    if [[ $pylint_version == $EXPECTED_PYLINT ]]; then
        echo -e "${GREEN}✓${NC} Pylint: $pylint_version"
    else
        echo -e "${YELLOW}⚠${NC} Pylint: $pylint_version (expected $EXPECTED_PYLINT)"
    fi
else
    echo -e "${RED}✗${NC} Pylint not found"
fi

# Check additional tools
echo ""
echo -e "${BLUE}Additional tools:${NC}"

if command -v pre-commit >/dev/null 2>&1; then
    precommit_version=$(pre-commit --version 2>/dev/null | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Pre-commit: $precommit_version"
else
    echo -e "${RED}✗${NC} Pre-commit not found"
fi

echo ""
echo -e "${BLUE}To install/upgrade to latest versions:${NC}"
echo "  pip install --upgrade -r requirements.txt"
echo ""
