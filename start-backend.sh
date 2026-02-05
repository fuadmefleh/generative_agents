#!/bin/bash

# Start the backend server
cd backend

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the FastAPI server with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 9010 --reload
