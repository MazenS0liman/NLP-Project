#!/bin/bash
# This script sets up the environment for the project.

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run server.py --server.port 8501 --server.address 0.0.0.0