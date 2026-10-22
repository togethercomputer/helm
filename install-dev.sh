#!/bin/bash

# This script installs pinned dependencies, then install CRFM-HELM in edit mode

set -e

# Install all pinned dependencies (torch/torchvision are pinned in requirements.txt)
pip install --no-cache-dir -r requirements.txt
# upgrade pip to install in edit mode without setup.py
pip install --upgrade pip
# Install HELM in edit mode
pip install -e .[all]
# Check dependencies
pip check
