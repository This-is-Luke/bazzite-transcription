import os
import re
import sys
import time
import wave
import queue
import signal
import shutil
import logging
import subprocess
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
from evdev import ecodes, InputDevice

# =======================
# CONFIGURATION
# =======================

KBD_PATH = '/dev/input/by-id/usb-Keychron_K4_Keychron_K4-event-kbd'

HOTKEY_MIC_CODE = ecodes.KEY_F9
HOTKEY_SYS_CODE = ecodes.KEY_F8

MODEL_SIZE = "large-v3-turbo"
LANGUAGE = "en"
BEAM_SIZE = 1

SAMPLE_RATE = 16000
MIC_CHANNELS = 1
SYS_CHANNELS = 2  # desktop audio is typically stereo; we downmix to mono

# Default: do NOT grab keyboard (won't interfere). Set GRAB_KEYBOARD=1 for exclusive grab.
GRAB_KEYBOARD = os.environ.get("GRAB_KEYBOARD", "0") == "1"

# Optional: add a startup delay (useful in systemd)
START_DELAY_SEC = float(os.environ.get("START_DELAY_SEC", "0"))

DEFAULT_HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "whisper-hf")

# Optional override for system audio (Pulse monitor name):
# export SYS_AUDIO_TARGET="<default-sink>.monitor"
SYS_AUDIO_TARGET_ENV = "SYS_AUDIO_TARGET"

REQUIRED_BINARIES = ["wl-copy", "wpctl", "pw-record"]
RECOMMENDED_BINARIES = ["pactl", "parec"]

# =======================
# LOGGING
# =======================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("WhisperHotkey")

# =======================
# CUDA / cuDNN LIB PATH FIX
# =======================

def _find_pkg_dir(modname: str) -> Optional[str]:
    try:
        import importlib.util
        spec = importlib.util.find_spec(modname)
        if spec is None:
            return None
        if spec.submodule_search_locations:
            return list(spec.submodule_search_locations)[0]
        if spec.origin:
            return os.path.dirname(spec.origin)
        return None
    except Exception:
        return None

def _prepend_ld_library_path(paths: List[str]) -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    new_parts = []
    for p in paths:
        if p and os.path.isdir(p) and p not in parts:
            new_parts.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + parts)

def _ensure_cuda_runtime_libs_visible():
    """
    After reopening the container/shell, LD_LIBRARY_PATH may be empty.
    We rebuild it from the pip-installed nvidia wheels so cuDNN can be found.
    """
    cudnn_dir = _find_pkg_dir("nvidia.cudnn.lib")
    cublas_dir = _find_pkg_dir("nvidia.cublas.lib")

    _prepend_ld_library_path([cublas_dir, cudnn_dir])

    if cudnn_dir:
        logger.info(f"cuDNN lib dir: {cudnn_dir}")
    else:
        logger.warning("Could not locate nvidia.cudnn.lib. If GPU crashes, install: pip install 'nvidia-cudnn-cu12==9.*'")

    if cublas_dir:
        logger.info(f"cuBLAS lib dir: {cublas_dir}")
    else:
        logger.warning("Could not locate nvidia.cublas.lib. If GPU crashes, install: pip install nvidia-cublas-cu12")

# =======================
# UTIL
# =======================

def ensure_hf_cache_writable():
    hf_home = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def require_binaries():
    missing = [b for b in REQUIRED_BINARIES if not have(b)]
    if missing:
        logger.error(f"Missing required binaries in the container: {', '.join(missing)}")
        logger.error("Install: wl-clipboard, pipewire-utils, wireplumber (or equivalents).")
        sys.exit(2)

    rec_missing = [b for b in RECOMMENDED_BINARIES if not have(b)]
    if rec_missing:
        logger.warning(f"Recommended not found: {', '.join(rec_missing)}")
        logger.warning("For best system audio capture: sudo dnf install -y pulseaudio-utils")

def run(cmd: List[str], timeout: Optional[float] = None):
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=timeout)
        combined = ("\n".join([out or "", err or ""])).strip()
        return p.returncode, combined
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:
        return 999, str(e)

def copy_to_clipboard(text: str):
    try:
        env = os.environ.copy()
        p = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE, env=env)
        p.communicate(input=text.encode("utf-8"))
        if p.returncode != 0:
            raise RuntimeError(f"wl-copy exited with code {p.returncode}")
        logger.info(f"DONE (clipboard): {text}")
    except Exception as e:
        logger.error(f"Clipboard Error: {e}")

def held(device: InputDevice, key_code: int) -> bool:
    try:
        return key_code in device.active_keys()
    except Exception:
        return False

def read_wav_to_float32(path: str) -> Optional[np.ndarray]:
    try:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            _rate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.float32)
            if np.any(np.abs(arr) > 2.0):
                data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                data = arr
        else:
            logger.error(f"Unsupported WAV sample width: {sampwidth} bytes")
            return None

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)

        return data.astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to read WAV '{path}': {e}")
        return None

# =======================
# SYSTEM AUDIO TARGET DISCOVERY
# =======================

def get_default_sink_node_name_wpctl() -> Optional[str]:
    rc, out = run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"])
    if rc != 0:
        return None
    m = re.search(r'node\.name\s*=\s*"([^"]+)"', out)
    return m.group(1).strip() if m else None

def get_default_sink_pactl() -> Optional[str]:
    if not have("pactl"):
        return None
    rc, out = run(["pactl", "get-default-sink"])
    if rc != 0:
        return None
    s = out.strip().splitlines()[-1].strip() if out.strip() else ""
    return s or None

def pulse_monitor_for_default_sink() -> Optional[str]:
    sink = get_default_sink_pactl()
    return (sink + ".monitor") if sink else None

# =======================
# AUDIO RECORDERS
# =======================

class MicRecorder:
    def __init__(self):
        self._queue = queue.Queue()
        self._is_recording = False

    def _callback(self, indata, frames, t, status):
        if self._is_recording:
            self._queue.put(indata.copy())

    def record_until_released(self, device: InputDevice, key_code: int, mic_device: Optional[int] = None) -> Optional[np.ndarray]:
        self._queue = queue.Queue()
        self._is_recording = True
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=mic_device,
                channels=MIC_CHANNELS,
                callback=self._callback,
            ):
                logger.info("--- RECORDING (MIC) ---")
                while held(device, key_code):
                    sd.sleep(25)

            self._is_recording = False
            chunks = []
            while not self._queue.empty():
                chunks.append(self._queue.get())

            if not chunks:
                return None
            return np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Mic recording interrupted: {e}")
            return None
        finally:
            self._is_recording = False


class SystemRecorder:
    def __init__(self):
        forced = os.environ.get(SYS_AUDIO_TARGET_ENV)
        if forced:
            self.target = forced.strip()
            logger.info(f"Using forced system audio target from env {SYS_AUDIO_TARGET_ENV}={self.target}")
        else:
            mon = pulse_monitor_for_default_sink()
            if mon:
                self.target = mon
                logger.info(f"Auto-selected Pulse monitor target: {self.target}")
            else:
                sink_node = get_default_sink_node_name_wpctl()
                if sink_node:
                    self.target = sink_node + ".monitor"
                    logger.warning(f"No pactl; guessing monitor target as: {self.target}")
                else:
                    self.target = None

        if self.target and have("parec") and have("pactl"):
            self.backend = "parec"
        else:
            self.backend = "pw-record"

        if not self.target:
            logger.error("SYSTEM audio not configured.")
            logger.error("Install pulseaudio-utils OR set SYS_AUDIO_TARGET explicitly.")
        else:
            logger.info(f"System audio target: {self.target} (backend={self.backend})")

    def _record_with_parec(self, device: InputDevice, key_code: int) -> Optional[np.ndarray]:
        tmp_raw = os.path.join("/tmp", f"whisper_sys_{int(time.time()*1000)}.raw")
        cmd = [
            "parec",
            "-d", self.target,
            "--rate", str(SAMPLE_RATE),
            "--channels", str(SYS_CHANNELS),
            "--format", "s16le",
        ]

        logger.info("--- RECORDING (SYSTEM) ---")
        p = None
        try:
            with open(tmp_raw, "wb") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)

                while held(device, key_code):
                    if p.poll() is not None:
                        err = (p.stderr.read().decode("utf-8", errors="replace") if p.stderr else "").strip()
                        logger.error(f"parec exited early (code {p.returncode}). stderr:\n{err}")
                        return None
                    time.sleep(0.03)

                try:
                    p.send_signal(signal.SIGINT)
                except Exception:
                    pass

                try:
                    _, errb = p.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    _, errb = p.communicate()

                if p.returncode not in (0, 130):
                    err = (errb.decode("utf-8", errors="replace") if errb else "").strip()
                    logger.error(f"parec failed (code {p.returncode}). stderr:\n{err}")
                    return None

            raw = open(tmp_raw, "rb").read()
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if SYS_CHANNELS > 1:
                data = data.reshape(-1, SYS_CHANNELS).mean(axis=1)
            return data.astype(np.float32)

        except Exception as e:
            logger.error(f"System recording (parec) failed: {e}")
            return None
        finally:
            try:
                if os.path.exists(tmp_raw):
                    os.remove(tmp_raw)
            except Exception:
                pass

    def _record_with_pw_record(self, device: InputDevice, key_code: int) -> Optional[np.ndarray]:
        tmp_wav = os.path.join("/tmp", f"whisper_sys_{int(time.time()*1000)}.wav")
        cmd = [
            "pw-record",
            "--target", self.target,
            "--rate", str(SAMPLE_RATE),
            "--channels", str(SYS_CHANNELS),
            tmp_wav
        ]

        logger.info("--- RECORDING (SYSTEM) ---")
        p = None
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            while held(device, key_code):
                if p.poll() is not None:
                    out, err = p.communicate(timeout=0.2)
                    combined = ("\n".join([out or "", err or ""])).strip()
                    logger.error(f"pw-record exited early (code {p.returncode}). output:\n{combined}")
                    return None
                time.sleep(0.03)

            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass

            try:
                out, err = p.communicate(timeout=2.5)
            except subprocess.TimeoutExpired:
                p.terminate()
                out, err = p.communicate()

            if p.returncode not in (0, 130):
                combined = ("\n".join([out or "", err or ""])).strip()
                logger.error(f"pw-record failed (code {p.returncode}). output:\n{combined}")
                return None

            audio = read_wav_to_float32(tmp_wav)
            if audio is None or audio.size == 0:
                logger.error("System recording produced no audio samples.")
                return None
            return audio
        except Exception as e:
            logger.error(f"System recording (pw-record) failed: {e}")
            return None
        finally:
            try:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
            except Exception:
                pass

    def record_until_released(self, device: InputDevice, key_code: int) -> Optional[np.ndarray]:
        if not self.target:
            logger.error(f"SYSTEM audio not configured. Set {SYS_AUDIO_TARGET_ENV} then retry F8.")
            return None
        if self.backend == "parec":
            return self._record_with_parec(device, key_code)
        return self._record_with_pw_record(device, key_code)

# =======================
# WHISPER INIT (import after LD_LIBRARY_PATH fix)
# =======================

def init_whisper_gpu_or_die():
    logger.info("Initializing Faster-Whisper (GPU-only)...")
    try:
        from faster_whisper import WhisperModel  # delayed import
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        logger.info("GPU Acceleration: ENABLED")
        return model
    except Exception as e:
        logger.error(f"GPU initialization FAILED: {e}")
        logger.error("GPU-only required, exiting.")
        sys.exit(3)

# =======================
# MAIN
# =======================

def main():
    if START_DELAY_SEC > 0:
        time.sleep(START_DELAY_SEC)

    ensure_hf_cache_writable()
    require_binaries()

    # Make GPU libs resolvable even after reopening the container/shell
    _ensure_cuda_runtime_libs_visible()

    if not os.path.exists(KBD_PATH):
        logger.error(f"Keyboard device not found: {KBD_PATH}")
        sys.exit(2)

    kbd = InputDevice(KBD_PATH)

    if GRAB_KEYBOARD:
        try:
            kbd.grab()
            logger.info("Keyboard GRABBED (exclusive). Keys won’t reach other apps while running.")
        except Exception as e:
            logger.error(f"Failed to grab keyboard: {e}")
            sys.exit(2)
    else:
        logger.info("Keyboard NOT grabbed. Listening for key events only (won't interfere with typing).")

    try:
        model = init_whisper_gpu_or_die()
        mic = MicRecorder()
        sysrec = SystemRecorder()

        logger.info("READY.")
        logger.info("Hold F9 for MIC transcription.")
        logger.info("Hold F8 for SYSTEM (desktop) audio transcription.")

        for event in kbd.read_loop():
            if event.type != ecodes.EV_KEY:
                continue
            if event.value != 1:  # key press only
                continue

            audio = None
            if event.code == HOTKEY_MIC_CODE:
                audio = mic.record_until_released(kbd, HOTKEY_MIC_CODE, mic_device=None)
            elif event.code == HOTKEY_SYS_CODE:
                audio = sysrec.record_until_released(kbd, HOTKEY_SYS_CODE)
            else:
                continue

            if audio is None:
                continue

            if audio.shape[0] < int(0.25 * SAMPLE_RATE):
                logger.info("Clip too short; ignoring.")
                continue

            logger.info("AI is thinking...")
            try:
                segments, _ = model.transcribe(audio.astype(np.float32), beam_size=BEAM_SIZE, language=LANGUAGE)
                text = " ".join([s.text for s in segments]).strip()
                if text:
                    copy_to_clipboard(text)
                else:
                    logger.info("No speech detected.")
            except Exception as e:
                logger.error(f"Transcription failed: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        if GRAB_KEYBOARD:
            try:
                kbd.ungrab()
            except Exception:
                pass
        logger.info("Exiting.")

if __name__ == "__main__":
    main()
