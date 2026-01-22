# Hold-to-Transcribe for Bazzite (Distrobox + Faster-Whisper)

A lightweight "hold a hotkey, speak (or play audio), release hotkey, get text in clipboard" workflow for Bazzite (GNOME/Wayland). Uses **Faster-Whisper** (CTranslate2) with **NVIDIA GPU acceleration** and writes results to the Wayland clipboard (paste anywhere with `Ctrl+V`).

## What this does

*   **Hold F9** → records microphone while held → transcribes on GPU → copies text to clipboard
*   **Hold F8** → records desktop/system audio (monitor source) while held → transcribes on GPU → copies text to clipboard
*   Runs well as a **systemd user service** so it’s available after login.

## What this does not do

*   No cloud calls, no external transcription service.
*   No UI; it’s designed to be "always on" in the background.
*   It does not magically bypass permissions: reading keyboard devices + system audio capture still depends on your system/container setup.

## Requirements (Bazzite host)

*   Bazzite GNOME (Wayland) with PipeWire/WirePlumber (default on Bazzite)
*   NVIDIA drivers working on host (`nvidia-smi` works)
*   **Distrobox** installed (Bazzite includes it in most images; otherwise install via system tooling)

### The container needs these binaries

Inside the container:
*   `wl-copy` (Wayland clipboard) → package: `wl-clipboard`
*   `wpctl`, `pw-record` (PipeWire utils) → package: `pipewire-utils`
*   For best desktop-audio capture: `pactl`, `parec` → package: `pulseaudio-utils`

*Note: This does not require running PulseAudio as the server; PipeWire provides Pulse compatibility.*

## Repo Layout

Example:
```text
whisper-project/
  transcribe.py
  README.md

If you use a venv:
whisper-env/   (created locally, not committed)
```

## Installation (Distrobox on Bazzite)

### 1) Create a Distrobox container (Fedora-based is easiest on Bazzite)

On the host:
```bash
distrobox create -n whisper-box -i fedora:40
distrobox enter -n whisper-box
```
*If you already have a box, just `distrobox enter -n whisper-box`.*

### 2) Install container packages

Inside the box:
```bash
sudo dnf install -y \
  python3 python3-pip python3-virtualenv \
  wl-clipboard pipewire-utils wireplumber \
  pulseaudio-utils \
  gcc gcc-c++ make \
  portaudio portaudio-devel \
  libsndfile libsndfile-devel
```

**Notes:**
*   `sounddevice` uses PortAudio; you generally need `portaudio` available.
*   `pulseaudio-utils` gives you `pactl` and `parec` which makes "desktop audio" capture far more reliable.

### 3) Create and activate a Python venv

Inside the box:
```bash
cd ~
python3 -m venv ~/whisper-env
source ~/whisper-env/bin/activate
python -m pip install -U pip
```

### 4) Install Python deps

Inside the venv:
```bash
pip install -U \
  numpy sounddevice evdev huggingface_hub \
  faster-whisper
```

### 5) Install NVIDIA CUDA user-space libs via pip (cuBLAS + cuDNN)

Inside the venv:
```bash
pip install -U nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

**Why pip libs?**
*   In container land, matching Fedora/NVIDIA repos can be a headache.
*   Pip wheels provide the runtime `.so` libraries CTranslate2 needs.

### 6) Put `transcribe.py` in your repo folder

Place the script at `~/whisper-project/transcribe.py`.

Then run:
```bash
cd ~/whisper-project
python3 transcribe.py
```

## Usage

### Hotkeys

*   **Hold F9** → Mic transcription → clipboard
*   **Hold F8** → System/desktop audio transcription → clipboard

### Clipboard

The result is copied using `wl-copy`. Paste anywhere with `Ctrl+V`.

### Background usage

This is designed to run continuously. You can start it manually in a terminal, or run it via systemd (recommended).

## Environment variables

These are optional but useful:

*   `GRAB_KEYBOARD=1`
    *   Grabs keyboard device exclusively via `evdev.grab()`
    *   More reliable hotkey detection (no stray escape sequences in terminals)
    *   *Downsides: hotkeys might not reach other apps while running*

*   `START_DELAY_SEC=5`
    *   Adds a delay on startup (helpful on login so audio/session is ready)

*   `SYS_AUDIO_TARGET="<sink>.monitor"` (Linux only)
    *   Force a specific PipeWire/Pulse monitor source.
    *   Usually auto-detected correctly, but forcing can help unusual setups.

### Desktop/System Audio Notes (important)

On PipeWire systems, "desktop audio" is usually captured from a monitor source of the default sink (speaker output), commonly named like:
`alsa_output.pci-0000_00_1f.3.analog-stereo.monitor`

This project tries to auto-detect the default sink monitor using `pactl`. If auto-detection fails, set it manually:

```bash
pactl get-default-sink
# suppose it prints: alsa_output.pci-0000_00_1f.3.analog-stereo

export SYS_AUDIO_TARGET="alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
```

## Running at login (systemd user service on host)

### 1) Create the systemd user service

On the host:

```bash
mkdir -p ~/.config/systemd/user

cat << 'EOF' > ~/.config/systemd/user/whisper-transcribe.service
[Unit]
Description=Hold-to-transcribe (Distrobox + Faster-Whisper)
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple

# Optional delay so the audio session is ready
ExecStartPre=/usr/bin/sleep 5

# Import Wayland/desktop session env so wl-copy and pipewire behave correctly
ExecStart=/usr/bin/bash -lc 'systemctl --user import-environment DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR XDG_CURRENT_DESKTOP; /usr/bin/distrobox enter -n whisper-box -- bash -lc "cd ~/whisper-project && source ~/whisper-env/bin/activate && exec python3 transcribe.py"'

Restart=on-failure
RestartSec=2
KillSignal=SIGINT
TimeoutStopSec=3
KillMode=control-group

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now whisper-transcribe.service
```

### 2) Useful control commands

On the host:
```bash
systemctl --user start whisper-transcribe.service
systemctl --user stop whisper-transcribe.service
systemctl --user restart whisper-transcribe.service
systemctl --user status whisper-transcribe.service --no-pager
journalctl --user -u whisper-transcribe.service -f
```

### 3) Optional convenience scripts

If you want `whisper-on` / `whisper-off` / `whisper-logs`:

```bash
mkdir -p ~/.local/bin

cat << 'EOF' > ~/.local/bin/whisper-on
#!/usr/bin/env bash
systemctl --user start whisper-transcribe.service
EOF

cat << 'EOF' > ~/.local/bin/whisper-off
#!/usr/bin/env bash
systemctl --user stop whisper-transcribe.service
EOF

cat << 'EOF' > ~/.local/bin/whisper-restart
#!/usr/bin/env bash
systemctl --user restart whisper-transcribe.service
EOF

cat << 'EOF' > ~/.local/bin/whisper-status
#!/usr/bin/env bash
systemctl --user status whisper-transcribe.service --no-pager
EOF

cat << 'EOF' > ~/.local/bin/whisper-logs
#!/usr/bin/env bash
journalctl --user -u whisper-transcribe.service -n 200 --no-pager
EOF

chmod +x ~/.local/bin/whisper-*
```

Ensure `~/.local/bin` is on PATH (Bazzite usually does this already):
```bash
echo $PATH | tr ':' '\n' | grep -n '\.local/bin' || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
```

## Troubleshooting

### 1) "It worked, I reopened the container, now GPU crashes with cuDNN errors"

Symptoms look like:
`Unable to load any of {libcudnn_ops.so...}`
`Invalid handle. Cannot load symbol ...`

This usually means the new shell/session doesn’t have the correct dynamic loader search path for the pip-installed NVIDIA .so libraries.

**Fix options:**
*   Run via the systemd service (recommended), which gives a consistent startup environment.
*   Or ensure your launcher sets `LD_LIBRARY_PATH` before Python starts (repo can include a `run_transcribe.sh`).

### 2) Desktop audio doesn’t record / SYS_AUDIO_TARGET not found

*   Install `pulseaudio-utils` in the container.
*   Confirm monitor sources exist:
    ```bash
    pactl list short sources | grep -i monitor
    ```
*   Then set:
    ```bash
    export SYS_AUDIO_TARGET="<name>.monitor"
    ```

### 3) Hotkeys not detected

*   Make sure `KBD_PATH` matches your keyboard device.
*   Check:
    ```bash
    ls -la /dev/input/by-id/
    ```
*   Permissions: you need access to that event device. Usually being in the `input` group helps:
    ```bash
    groups
    ```
*   If detection is flaky, set:
    `GRAB_KEYBOARD=1` (more reliable, but more invasive)

### 4) Clipboard doesn’t update

*   Confirm `wl-copy` exists in the container:
    ```bash
    which wl-copy
    ```
*   Confirm Wayland env is available:
    ```bash
    echo "$WAYLAND_DISPLAY"
    echo "$XDG_RUNTIME_DIR"
    ```

### 5) "My terminal gets spammed with ^[[19~ when pressing F8/F9"

That’s normal when your terminal sees raw function key escape sequences.
If you run with `GRAB_KEYBOARD=1`, it won’t leak into your terminal (because the keyboard is grabbed).

## Security / UX warnings (read this)

*   A process reading `/dev/input/...event-kbd` is powerful. This tool only listens for specific hotkeys, but it still has input access.
*   `GRAB_KEYBOARD=1` will prevent keys reaching other apps while it runs. Use only if you want reliability over non-interference.

## Credits

*   Transcription: **Faster-Whisper** (CTranslate2 backend)
*   Clipboard: **wl-clipboard**
*   Audio stack: **PipeWire / WirePlumber** (Pulse compatibility for monitor capture)

---

# macOS (Apple Silicon / M1)

This script also supports macOS with a slightly different setup (no `evdev`, no PipeWire, no `wl-copy`).

## macOS Requirements

*   Python 3.10+ (pyenv or Homebrew)
*   Homebrew packages:
    ```bash
    brew install portaudio
    ```
*   Pip packages:
    ```bash
    pip install -U numpy sounddevice pynput faster-whisper
    ```

## macOS Setup Notes

*   **Hotkeys**: `pynput` requires Accessibility permissions. Grant access in:
    `System Settings → Privacy & Security → Accessibility`.
*   **Hotkey behavior (macOS)**:
    *   Press **F8** once to start/stop system audio
    *   Press **F9** once to start/stop microphone
    *   Auto-stops after 5 minutes (configurable)
*   **Clipboard**: uses `pbcopy` (built in).
*   **System audio**: macOS cannot capture system audio without a loopback device.
    Install a loopback device like **BlackHole** and set `SYS_AUDIO_DEVICE` to that input.

## macOS Example

```bash
export SYS_AUDIO_DEVICE="BlackHole 2ch"
export WHISPER_DEVICE="cpu"
export WHISPER_COMPUTE_TYPE="int8"
python3 transcription.py
```

## Run on login (macOS launchd)

This starts the script in the background at login without opening a terminal.

### 1) Create a LaunchAgent

```bash
mkdir -p ~/Library/LaunchAgents

cat << 'EOF' > ~/Library/LaunchAgents/com.whisper.transcribe.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.whisper.transcribe</string>

  <key>ProgramArguments</key>
  <array>
    <string>/Users/REPLACE_ME/Documents/code/transcription/bazzite-transcription/.venv/bin/python</string>
    <string>/Users/REPLACE_ME/Documents/code/transcription/bazzite-transcription/transcription.py</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>WHISPER_DEVICE</key>
    <string>cpu</string>
    <key>WHISPER_COMPUTE_TYPE</key>
    <string>int8</string>
    <key>HOTKEY_SYS_KEY</key>
    <string>f8</string>
    <key>HOTKEY_MIC_KEY</key>
    <string>f9</string>
    <!-- Optional: set system audio input device -->
    <!-- <key>SYS_AUDIO_DEVICE</key> <string>BlackHole 2ch</string> -->
  </dict>

  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/Users/REPLACE_ME/Library/Logs/whisper-transcribe.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/REPLACE_ME/Library/Logs/whisper-transcribe.err.log</string>
</dict>
</plist>
EOF
```

Replace `REPLACE_ME` with your macOS username.

### 2) Load and start

```bash
launchctl unload ~/Library/LaunchAgents/com.whisper.transcribe.plist 2>/dev/null
launchctl load ~/Library/LaunchAgents/com.whisper.transcribe.plist
launchctl start com.whisper.transcribe
```

### 3) Stop / restart

```bash
launchctl stop com.whisper.transcribe
launchctl unload ~/Library/LaunchAgents/com.whisper.transcribe.plist
launchctl load ~/Library/LaunchAgents/com.whisper.transcribe.plist
```

### 4) Logs

```bash
tail -f ~/Library/Logs/whisper-transcribe.out.log
tail -f ~/Library/Logs/whisper-transcribe.err.log
```

**Note:** The first time it runs, macOS will still require Accessibility/Input Monitoring permissions. Grant those to the app that hosts this process (the Python binary in `.venv`).

## macOS Environment variables (additional)

*   `SYS_AUDIO_DEVICE="<device name or index>"` (macOS only)
    *   Use a loopback input like **BlackHole** for system audio.
*   `MIC_DEVICE="<device name or index>"`
    *   Selects a specific input device for mic capture.
*   `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE`
    *   Override device and compute type (`cpu` + `int8` works well on M1).
*   `HOTKEY_SYS_KEY="f8"` / `HOTKEY_MIC_KEY="f9"`
    *   Override macOS hotkeys if needed.
*   `MAX_RECORD_SEC="300"`
    *   Max recording length before auto-stop.