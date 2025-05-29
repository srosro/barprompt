#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory if not already there
if [ "$PWD" != "$SCRIPT_DIR" ]; then
    echo "Changing to script directory: $SCRIPT_DIR"
    cd "$SCRIPT_DIR"
fi

# Log file path
LOG_FILE="prompt_sync.log"

# Function to trim log file if it gets too large
trim_log() {
    if [ -f "$LOG_FILE" ] && [ $(wc -l < "$LOG_FILE") -gt 5000 ]; then
        echo "Log file too large, trimming first 1000 lines..."
        tail -n +1001 "$LOG_FILE" > "${LOG_FILE}.tmp"
        mv "${LOG_FILE}.tmp" "$LOG_FILE"
    fi
}

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log "Starting prompt synchronization process"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log "Error: Virtual environment not found. Please create it first."
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source venv/bin/activate

# Run copy_prompts.py
log "Running copy_prompts.py..."
python copy_prompts.py >> "$LOG_FILE" 2>&1
COPY_EXIT_CODE=$?

if [ $COPY_EXIT_CODE -ne 0 ]; then
    log "Error: copy_prompts.py failed with exit code $COPY_EXIT_CODE"
    # Don't exit here, continue with verification
fi

# Run verify_prompts.py
log "Running verify_prompts.py..."
python verify_prompts.py >> "$LOG_FILE" 2>&1
VERIFY_EXIT_CODE=$?

if [ $VERIFY_EXIT_CODE -ne 0 ]; then
    log "Error: verify_prompts.py failed with exit code $VERIFY_EXIT_CODE"
fi

# Deactivate virtual environment
deactivate

# Final status
if [ $COPY_EXIT_CODE -eq 0 ] && [ $VERIFY_EXIT_CODE -eq 0 ]; then
    log "Prompt synchronization completed successfully"
    exit 0
else
    log "Prompt synchronization completed with errors"
    exit 1
fi 