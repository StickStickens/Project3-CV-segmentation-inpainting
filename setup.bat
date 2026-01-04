#!/bin/bash

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Pull DVC data
dvc pull
