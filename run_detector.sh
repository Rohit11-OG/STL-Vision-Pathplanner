#!/bin/bash
# Industrial STL Object Detection Launcher
# Auto-restart on failure with graceful shutdown

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PYTHON_CMD="python"
DETECTOR_SCRIPT="main.py"
RESTART_DELAY=5
MAX_RESTARTS=10
LOG_FILE="logs/launcher.log"

# Create logs directory
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cleanup() {
    log "Received shutdown signal, stopping detector..."
    kill $DETECTOR_PID 2>/dev/null
    wait $DETECTOR_PID 2>/dev/null
    log "Detector stopped gracefully"
    exit 0
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM

log "============================================"
log "STL Object Detection System Starting"
log "============================================"

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    log "Starting detector (attempt $((restart_count + 1))/$MAX_RESTARTS)"
    
    # Run detector in background
    $PYTHON_CMD $DETECTOR_SCRIPT detect --camera 0 &
    DETECTOR_PID=$!
    
    # Wait for detector to finish
    wait $DETECTOR_PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "Detector exited normally"
        break
    else
        log "Detector crashed with exit code $EXIT_CODE"
        restart_count=$((restart_count + 1))
        
        if [ $restart_count -lt $MAX_RESTARTS ]; then
            log "Restarting in $RESTART_DELAY seconds..."
            sleep $RESTART_DELAY
        fi
    fi
done

if [ $restart_count -ge $MAX_RESTARTS ]; then
    log "ERROR: Max restarts reached, giving up"
    exit 1
fi

log "Detector shutdown complete"
