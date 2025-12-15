#!/bin/bash

# Quick start script for Sentinel Backend

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found."
    echo "   Creating .env file with placeholder..."
    echo "API_KEY=your_api_key_here" > .env
    echo "   Please edit .env and add your Google Gemini API key"
    echo "   Get one at: https://makersuite.google.com/app/apikey"
    echo ""
fi

# Start the server
echo "🚀 Starting Sentinel Backend server..."
echo "   Server will be available at: http://localhost:5001"
echo "   Press Ctrl+C to stop"
echo ""
python server.py


