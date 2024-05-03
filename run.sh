#!/bin/bash

echo "Deactivating any existing virtual environment..."
deactivate 2>/dev/null || true

echo "Creating a virtual environment..."
python3 -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing the required Python packages..."
pip install numpy scipy matplotlib pyod

echo "Running the Python script..."
python main.py 1>/dev/stdout 2>/dev/stderr

echo "Deactivating the virtual environment..."
deactivate
