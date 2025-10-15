#!/bin/bash

# BrowserGym Quick Setup Script
# This script will get you up and running with BrowserGym quickly

set -e  # Exit on any error

echo "🚀 BrowserGym Quick Setup"
echo "========================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "demo_agent" ]; then
    echo "❌ Error: Please run this script from the BrowserGym root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected to find: README.md and demo_agent/ directory"
    exit 1
fi

echo "✅ Found BrowserGym directory"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10+ required, found: $python_version"
    echo "   Please install Python 3.10 or higher"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "🐍 Conda found, using conda environment setup"
    USE_CONDA=true
else
    echo "🐍 Conda not found, using pip setup"
    USE_CONDA=false
fi

# Setup environment
if [ "$USE_CONDA" = true ]; then
    echo "📦 Creating conda environment..."
    conda env create -f demo_agent/environment.yml --force
    echo "✅ Conda environment created"
    
    echo "🔧 Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate demo-agent
    echo "✅ Environment activated"
else
    echo "📦 Installing packages with pip..."
    pip install browsergym openai
    echo "✅ Packages installed"
fi

# Install Playwright browser
echo "🌐 Installing Playwright browser..."
python -m playwright install chromium
echo "✅ Playwright browser installed"

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   You'll need to set it before running the demo agent:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "   Or create a .env file with your key"
else
    echo "✅ OpenAI API key found"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "   1. Set your OpenAI API key (if not already set):"
echo "      export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "   2. Test the basic environment:"
echo "      python simple_agent_test.py"
echo ""
echo "   3. Run the demo agent:"
echo "      cd demo_agent"
echo "      python run_demo.py --task_name openended --start_url https://www.google.com"
echo ""
echo "   4. Try different benchmarks:"
echo "      python run_demo.py --task_name miniwob.click-dialog"
echo "      python run_demo.py --task_name webarena.310"
echo ""
echo "📚 Read browsergym_getting_started.md for detailed instructions"
echo ""
echo "🎯 Happy testing! This will help you understand how to build your green agent."
