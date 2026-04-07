#!/bin/bash

# Function to check if uv is installed, if not install it otherwise print "found uv"
check_and_install_uv() {
    if command -v uv &> /dev/null; then
        echo "found uv"
    else
        echo "uv not found. Installing..."
        # On macOS and Linux.
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
}

# Function to check .venv in the folder, if present activate it, otherwise use uv sync and locked file to create it
check_and_setup_venv() {
    if [ -d ".venv" ]; then
        echo "Virtual environment found. Activating..."
        source .venv/bin/activate
    else
        echo "Virtual environment not found. Creating using uv sync and locked file..."
        # Assuming you have a uv command to create the virtual environment from a lock file
        # Replace the following line with the appropriate command for your system
        uv sync   # Example command
    fi
}

# Function to check if invoke is present, install it via pip if not
check_and_install_invoke() {
    if . .venv/bin/activate && python -c "import invoke" /dev/null; then
        echo "found invoke"
    else
        echo "invoke not found. Installing via pip..."
        . .venv/bin/activate && uv pip install invoke
    fi
}

# Function to check if spacy is installed, download spacy model if not present
check_and_install_spacy() {
    if . .venv/bin/activate &&python -c "import spacy" &> /dev/null; then
        echo "found spacy"
    else
        echo "spacy not found. Installing via pip..."
        . .venv/bin/activate && uv pip install spacy
    fi

    # Check for a specific spacy model, e.g., 'en_core_web_sm'
    if . .venv/bin/activate && python -c "import spacy; nlp = spacy.load('en_core_web_sm')" &> /dev/null; then
        echo "found en_core_web_sm model"
    else
        echo "en_core_web_sm model not found. Downloading..."
        . .venv/bin/activate && uv pip install pip && python -m spacy download en_core_web_sm
    fi
}
# Example usage of the functions
check_and_install_uv
check_and_setup_venv
check_and_install_invoke
check_and_install_spacy
