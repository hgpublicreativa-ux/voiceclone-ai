#!/bin/bash
echo "Iniciando backend Voice Cloner..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
