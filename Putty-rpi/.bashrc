if [[ $- == *i* && -n "$SSH_CONNECTION" && -z "${PHASEON_AUTOSTART:-}" ]]; then
    export PHASEON_AUTOSTART=1
    cd "$HOME/barcode_project" || return
    python3 phaseon_camera_test.py
fi