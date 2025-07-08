#!/bin/bash

# Comprehensive linting and formatting script for TreePlanter project
# Using latest versions: Python 3.13, Ruff 0.12.2, Pylint 3.3.7
echo "🔍 Running comprehensive code quality checks..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Exit codes
EXIT_CODE=0

# 1. Check if required tools are installed
print_section "Checking Required Tools"
TOOLS=("ruff" "pylint")
for tool in "${TOOLS[@]}"; do
    if command_exists "$tool"; then
        version_info=""
        case $tool in
            "ruff")
                version_info=$(ruff --version 2>/dev/null | head -1 || echo "version unknown")
                ;;
            "pylint")
                version_info=$(pylint --version 2>/dev/null | head -1 || echo "version unknown")
                ;;
        esac
        echo -e "${GREEN}✓${NC} $tool is available ($version_info)"
    else
        echo -e "${RED}✗${NC} $tool is not installed"
        EXIT_CODE=1
    fi
done

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Please install missing tools: pip install -r requirements.txt${NC}"
    exit $EXIT_CODE
fi

echo ""

# 2. Run Ruff check
print_section "Running Ruff Linter"
if ruff check . --output-format=text; then
    echo -e "${GREEN}✓ Ruff linting passed${NC}"
else
    echo -e "${YELLOW}⚠ Ruff found issues (see above)${NC}"
    EXIT_CODE=1
fi

echo ""

# 3. Run Ruff format check
print_section "Running Ruff Format Check"
if ruff format --check .; then
    echo -e "${GREEN}✓ Ruff formatting check passed${NC}"
else
    echo -e "${YELLOW}⚠ Ruff formatting issues found (run 'ruff format .' to fix)${NC}"
    EXIT_CODE=1
fi

echo ""

# 4. Run Pylint on main files
print_section "Running Pylint"
PYTHON_FILES=("main.py" "tree_planner/")
for file in "${PYTHON_FILES[@]}"; do
    if [ -f "$file" ] || [ -d "$file" ]; then
        echo "Checking $file..."
        if pylint "$file" --rcfile=pylintrc; then
            echo -e "${GREEN}✓ Pylint check passed for $file${NC}"
        else
            echo -e "${YELLOW}⚠ Pylint found issues in $file${NC}"
            EXIT_CODE=1
        fi
    else
        echo -e "${YELLOW}⚠ File $file not found${NC}"
    fi
done

echo ""

# Summary
print_section "Summary"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 All code quality checks passed!${NC}"
    echo -e "${GREEN}Your code is ready for commit.${NC}"
else
    echo -e "${YELLOW}⚠ Some issues were found. Please review and fix them.${NC}"
    echo ""
    echo -e "${BLUE}Quick fixes:${NC}"
    echo "  - Run 'ruff format .' to fix formatting"
    echo "  - Run 'ruff check . --fix' to auto-fix some issues"
    echo "  - Review Pylint suggestions for code quality improvements"
fi

echo ""
exit $EXIT_CODE
