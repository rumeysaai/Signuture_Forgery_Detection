#!/bin/bash
# macOS launcher script for Signature Forgery Detection System

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set environment variables for macOS
export PYTHONUNBUFFERED=1
export DISPLAY=:0

# Run the application
python main.py


