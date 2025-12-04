#!/bin/bash
# Raspberry Pi 5 Setup Script for Dementia Assistant
# Run with: chmod +x setup_pi.sh && ./setup_pi.sh

set -e

echo "=========================================="
echo "Dementia Assistant - Pi 5 Setup"
echo "=========================================="

# Update system
echo "[1/7] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for dlib/face_recognition
echo "[2/7] Installing build tools and dependencies..."
sudo apt install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-pip \
    python3-venv \
    python3-tk \
    portaudio19-dev \
    libjpeg-dev \
    libpng-dev \
    wget \
    unzip

# Add user to video and audio groups (for camera and mic access)
echo "[3/7] Adding user to video and audio groups..."
sudo usermod -aG video $USER
sudo usermod -aG audio $USER

# Create virtual environment
echo "[4/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "[5/7] Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install dlib first (takes ~15 minutes to compile on Pi 5)
echo "[6/7] Installing dlib (this may take 10-15 minutes)..."
pip install dlib

# Install remaining requirements
echo "[7/7] Installing Python dependencies..."
pip install -r requirements.txt

# Download Vosk model
echo "[OPTIONAL] Downloading Vosk speech model..."
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODELS_DIR="models"

if [ ! -d "$MODELS_DIR/vosk-model-small-en-us-0.15" ]; then
    echo "Downloading Vosk model (~40MB)..."
    wget -q $VOSK_MODEL_URL -O vosk-model.zip
    unzip -q vosk-model.zip -d $MODELS_DIR
    rm vosk-model.zip
    echo "Vosk model downloaded successfully!"
else
    echo "Vosk model already exists, skipping download."
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Log out and log back in for group changes to take effect."
echo ""
echo "To run the application:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Next steps:"
echo "  1. Add face images to data/known_faces/<person_name>/"
echo "  2. Edit data/relationships.json with names and relationships"
echo "  3. Run: python main.py"
