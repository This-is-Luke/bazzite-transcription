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
import threading
from typing import Optional, Tuple, List, Callable, Dict

import numpy as np
import sounddevice as sd

IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"

if IS_LINUX:
    from evdev import ecodes, InputDevice
else:
    ecodes = None
    InputDevice = None

# =======================
# CONFIGURATION
# =======================

DEFAULT_KBD_PATH = "/dev/input/by-id/usb-Keychron_K4_Keychron_K4-event-kbd"
KBD_PATH = os.environ.get("KBD_PATH", DEFAULT_KBD_PATH)

HOTKEY_MIC_CODE = ecodes.KEY_F9 if IS_LINUX else None
HOTKEY_SYS_CODE = ecodes.KEY_F8 if IS_LINUX else None

DEFAULT_MODEL_SIZE_LINUX = "large-v3-turbo"
DEFAULT_MODEL_SIZE_MAC = "small"
DEFAULT_LANGUAGE = "en"
DEFAULT_BEAM_SIZE = 1

MODEL_SIZE = os.environ.get("MODEL_SIZE", DEFAULT_MODEL_SIZE_MAC if IS_MAC else DEFAULT_MODEL_SIZE_LINUX)
LANGUAGE = os.environ.get("LANGUAGE", DEFAULT_LANGUAGE)
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", str(DEFAULT_BEAM_SIZE)))

DEFAULT_DEVICE_LINUX = "cuda"
DEFAULT_DEVICE_MAC = "cpu"
DEFAULT_COMPUTE_LINUX = "float16"
DEFAULT_COMPUTE_MAC = "int8"

WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", DEFAULT_DEVICE_MAC if IS_MAC else DEFAULT_DEVICE_LINUX)
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", DEFAULT_COMPUTE_MAC if IS_MAC else DEFAULT_COMPUTE_LINUX)

SAMPLE_RATE = 16000
MIC_CHANNELS = 1
SYS_CHANNELS = 2  # desktop audio is typically stereo; we downmix to mono
MIN_CLIP_SEC = float(os.environ.get("MIN_CLIP_SEC", "0.25"))
POLL_INTERVAL_MS = 25
POLL_INTERVAL_SEC = 0.03
MAX_RECORD_SEC = float(os.environ.get("MAX_RECORD_SEC", "300"))
DEFAULT_SYS_AUDIO_GAIN_DB = 6.0
SYS_AUDIO_GAIN_DB = float(os.environ.get("SYS_AUDIO_GAIN_DB", str(DEFAULT_SYS_AUDIO_GAIN_DB)))

MAC_HOTKEY_MIC_ENV = "HOTKEY_MIC_KEY"
MAC_HOTKEY_SYS_ENV = "HOTKEY_SYS_KEY"
DEFAULT_MAC_HOTKEY_MIC = "f9"
DEFAULT_MAC_HOTKEY_SYS = "f8"

# Default: do NOT grab keyboard (won't interfere). Set GRAB_KEYBOARD=1 for exclusive grab.
GRAB_KEYBOARD = os.environ.get("GRAB_KEYBOARD", "0") == "1"

# Optional: add a startup delay (useful in systemd)
START_DELAY_SEC = float(os.environ.get("START_DELAY_SEC", "0"))

DEFAULT_HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "whisper-hf")

# Optional override for system audio (Pulse monitor name on Linux or device name/index on macOS):
# Linux: export SYS_AUDIO_TARGET="<default-sink>.monitor"
# macOS: export SYS_AUDIO_DEVICE="BlackHole 2ch" (name substring) or index
SYS_AUDIO_TARGET_ENV = "SYS_AUDIO_TARGET"
SYS_AUDIO_DEVICE_ENV = "SYS_AUDIO_DEVICE"
MIC_DEVICE_ENV = "MIC_DEVICE"

REQUIRED_BINARIES = ["wl-copy", "wpctl", "pw-record"] if IS_LINUX else []
RECOMMENDED_BINARIES = ["pactl", "parec"] if IS_LINUX else []

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
    if not IS_LINUX:
        return

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
        logger.error(f"Missing required binaries: {', '.join(missing)}")
        if IS_LINUX:
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
        if IS_MAC:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        else:
            env = os.environ.copy()
            p = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE, env=env)
        p.communicate(input=text.encode("utf-8"))
        if p.returncode != 0:
            raise RuntimeError(f"Clipboard command exited with code {p.returncode}")
        logger.info(f"DONE (clipboard): {text}")
    except Exception as e:
        logger.error(f"Clipboard Error: {e}")

def held(device: InputDevice, key_code: int) -> bool:
    try:
        return key_code in device.active_keys()
    except Exception:
        return False
# =======================
# MACOS HOTKEY LISTENER
# =======================

class MacHotkeyListener:
    def __init__(self, hotkey_map: Dict[str, str]):
        try:
            from pynput import keyboard as pynput_keyboard
        except Exception as e:
            logger.error(f"pynput is required on macOS for hotkeys: {e}")
            logger.error("Install with: pip install pynput")
            sys.exit(2)

        self._keyboard = pynput_keyboard
        self._queue: queue.Queue = queue.Queue()
        self.keys: Dict[str, object] = {}
        self._state: Dict[object, bool] = {}
        for name, key_name in hotkey_map.items():
            if not isinstance(key_name, str) or not key_name:
                logger.error(f"Invalid hotkey name for {name}: {key_name}")
                sys.exit(2)
            try:
                key_obj = getattr(self._keyboard.Key, key_name)
            except AttributeError:
                logger.error(f"Unknown key: {key_name}")
                sys.exit(2)
            self.keys[name] = key_obj
            self._state[key_obj] = False

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    def _on_press(self, key: object) -> None:
        if key in self._state and not self._state[key]:
            self._state[key] = True
            self._queue.put(key)

    def _on_release(self, key: object) -> None:
        if key in self._state:
            self._state[key] = False

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def next_press(self, timeout: Optional[float] = None) -> Optional[object]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_held(self, key: object) -> bool:
        return self._state.get(key, False)


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

def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0.0:
        return audio
    gain = 10 ** (gain_db / 20.0)
    boosted = audio * gain
    return np.clip(boosted, -1.0, 1.0)

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

    def record_while(self, is_held: Callable[[], bool], mic_device: Optional[int] = None) -> Optional[np.ndarray]:
        self._queue = queue.Queue()
        self._is_recording = True
        channels = resolve_input_channels(mic_device, MIC_CHANNELS, "mic")
        if channels == 0:
            return None
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=mic_device,
                channels=channels,
                callback=self._callback,
            ):
                logger.info("--- RECORDING (MIC) ---")
                while is_held():
                    sd.sleep(POLL_INTERVAL_MS)

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

    def record_until_released(self, device: InputDevice, key_code: int, mic_device: Optional[int] = None) -> Optional[np.ndarray]:
        return self.record_while(lambda: held(device, key_code), mic_device=mic_device)

    def record_until_stop(self, stop_event: threading.Event, max_sec: float, mic_device: Optional[int] = None) -> Optional[np.ndarray]:
        self._queue = queue.Queue()
        self._is_recording = True
        channels = resolve_input_channels(mic_device, MIC_CHANNELS, "mic")
        if channels == 0:
            return None
        start_time = time.monotonic()
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=mic_device,
                channels=channels,
                callback=self._callback,
            ):
                logger.info("--- RECORDING (MIC) ---")
                while not stop_event.is_set():
                    if time.monotonic() - start_time >= max_sec:
                        logger.warning("Max record duration reached; stopping.")
                        break
                    sd.sleep(POLL_INTERVAL_MS)

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
        self.target = None
        self.backend = None
        self.device = None

        if IS_MAC:
            self.backend = "sounddevice"
            self.device = resolve_audio_device(SYS_AUDIO_DEVICE_ENV, kind="input")
            if self.device is None:
                logger.error("SYS_AUDIO_DEVICE not set; system audio recording is disabled.")
                logger.error("Set SYS_AUDIO_DEVICE to a loopback input (e.g., BlackHole 2ch).")
            else:
                logger.info(f"Using macOS system audio device: {self.device}")
            return

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
                    time.sleep(POLL_INTERVAL_SEC)

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
                time.sleep(POLL_INTERVAL_SEC)

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
            return apply_gain_db(audio, SYS_AUDIO_GAIN_DB)
        except Exception as e:
            logger.error(f"System recording (pw-record) failed: {e}")
            return None
        finally:
            try:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
            except Exception:
                pass

    def _record_with_sounddevice(self, is_held: Callable[[], bool]) -> Optional[np.ndarray]:
        chunks = []
        device_index = self.device if self.device is not None else get_default_input_device_index()
        channels = resolve_input_channels(device_index, SYS_CHANNELS, "system")
        if channels == 0:
            return None

        def _callback(indata, frames, t, status):
            if is_held():
                chunks.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=device_index,
                channels=channels,
                callback=_callback,
            ):
                logger.info("--- RECORDING (SYSTEM) ---")
                while is_held():
                    sd.sleep(POLL_INTERVAL_MS)

            if not chunks:
                return None
            data = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
            return apply_gain_db(data.astype(np.float32), SYS_AUDIO_GAIN_DB)
        except Exception as e:
            logger.error(f"System recording (sounddevice) failed: {e}")
            return None

    def _record_with_sounddevice_until_stop(self, stop_event: threading.Event, max_sec: float) -> Optional[np.ndarray]:
        chunks = []
        device_index = self.device if self.device is not None else get_default_input_device_index()
        channels = resolve_input_channels(device_index, SYS_CHANNELS, "system")
        if channels == 0:
            return None

        def _callback(indata, frames, t, status):
            if not stop_event.is_set():
                chunks.append(indata.copy())

        start_time = time.monotonic()
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                device=device_index,
                channels=channels,
                callback=_callback,
            ):
                logger.info("--- RECORDING (SYSTEM) ---")
                while not stop_event.is_set():
                    if time.monotonic() - start_time >= max_sec:
                        logger.warning("Max record duration reached; stopping.")
                        break
                    sd.sleep(POLL_INTERVAL_MS)

            if not chunks:
                return None
            data = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
            return apply_gain_db(data.astype(np.float32), SYS_AUDIO_GAIN_DB)
        except Exception as e:
            logger.error(f"System recording (sounddevice) failed: {e}")
            return None

    def record_until_released(self, device: InputDevice, key_code: int) -> Optional[np.ndarray]:
        if IS_MAC:
            return self._record_with_sounddevice(lambda: held(device, key_code))

        if not self.target:
            logger.error(f"SYSTEM audio not configured. Set {SYS_AUDIO_TARGET_ENV} then retry F8.")
            return None
        if self.backend == "parec":
            return self._record_with_parec(device, key_code)
        return self._record_with_pw_record(device, key_code)

    def record_while(self, is_held: Callable[[], bool]) -> Optional[np.ndarray]:
        if IS_MAC:
            return self._record_with_sounddevice(is_held)
        logger.error("record_while is only supported on macOS.")
        return None

    def record_until_stop(self, stop_event: threading.Event, max_sec: float) -> Optional[np.ndarray]:
        if IS_MAC:
            if self.device is None:
                logger.error("System audio device not configured. Set SYS_AUDIO_DEVICE, then retry.")
                return None
            return self._record_with_sounddevice_until_stop(stop_event, max_sec)
        logger.error("record_until_stop is only supported on macOS.")
        return None


# =======================
# WHISPER INIT (import after LD_LIBRARY_PATH fix)
# =======================

def init_whisper_gpu_or_die():
    logger.info("Initializing Faster-Whisper...")
    try:
        from faster_whisper import WhisperModel  # delayed import
        model = WhisperModel(MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        logger.info(f"Device: {WHISPER_DEVICE} | Compute: {WHISPER_COMPUTE_TYPE}")
        return model
    except Exception as e:
        logger.error(f"Whisper initialization FAILED: {e}")
        sys.exit(3)

# =======================
# AUDIO DEVICE RESOLUTION (macOS)
# =======================

def resolve_audio_device(env_key: str, kind: str = "input") -> Optional[int]:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return None

    try:
        idx = int(raw)
    except ValueError:
        idx = None

    devices = sd.query_devices()
    if idx is not None:
        if idx < 0 or idx >= len(devices):
            logger.error(f"{env_key}={raw} is out of range. Available devices: {len(devices)}")
            return None
        return idx

    needle = raw.lower()
    for i, d in enumerate(devices):
        name = (d.get("name") or "").lower()
        if needle in name:
            if kind == "input" and d.get("max_input_channels", 0) > 0:
                return i
            if kind == "output" and d.get("max_output_channels", 0) > 0:
                return i

    logger.error(f"No audio device matches {env_key}={raw}")
    return None

def get_default_input_device_index() -> Optional[int]:
    try:
        default_device = sd.default.device
        if default_device and len(default_device) > 0 and default_device[0] is not None:
            return int(default_device[0])
    except Exception:
        return None
    return None

def resolve_input_channels(device_index: Optional[int], requested_channels: int, label: str) -> int:
    try:
        info = sd.query_devices(device_index, "input") if device_index is not None else sd.query_devices(kind="input")
        max_in = int(info.get("max_input_channels", 0) or 0)
        if max_in <= 0:
            logger.error(f"No input channels available for {label} device.")
            return 0
        channels = min(requested_channels, max_in)
        if channels != requested_channels:
            name = info.get("name") or "unknown"
            logger.warning(f"{label} device '{name}' supports {max_in} channel(s); using {channels}.")
        return channels
    except Exception as e:
        logger.error(f"Failed to query input device for {label}: {e}")
        return 0

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

    if not (IS_LINUX or IS_MAC):
        logger.error(f"Unsupported platform: {sys.platform}")
        sys.exit(2)

    if IS_LINUX:
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
    else:
        kbd = None

    listener = None
    try:
        model = init_whisper_gpu_or_die()
        mic = MicRecorder()
        sysrec = SystemRecorder()

        logger.info("READY.")
        logger.info("Hold F9 for MIC transcription.")
        logger.info("Hold F8 for SYSTEM (desktop) audio transcription.")

        if IS_LINUX:
            for event in kbd.read_loop():
                if event.type != ecodes.EV_KEY:
                    continue
                if event.value != 1:  # key press only
                    continue

                audio = None
                if event.code == HOTKEY_MIC_CODE:
                    audio = mic.record_until_released(
                        kbd,
                        HOTKEY_MIC_CODE,
                        mic_device=resolve_audio_device(MIC_DEVICE_ENV, kind="input"),
                    )
                elif event.code == HOTKEY_SYS_CODE:
                    audio = sysrec.record_until_released(kbd, HOTKEY_SYS_CODE)
                else:
                    continue

                if audio is None:
                    continue

                if audio.shape[0] < int(MIN_CLIP_SEC * SAMPLE_RATE):
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
        else:
            mic_key_name = os.environ.get(MAC_HOTKEY_MIC_ENV, DEFAULT_MAC_HOTKEY_MIC)
            sys_key_name = os.environ.get(MAC_HOTKEY_SYS_ENV, DEFAULT_MAC_HOTKEY_SYS)
            listener = MacHotkeyListener({"mic": mic_key_name, "sys": sys_key_name})
            listener.start()
            logger.info("macOS hotkeys: press F9 to start/stop MIC recording.")
            logger.info("macOS hotkeys: press F8 to start/stop SYSTEM recording.")
            logger.info("Grant Accessibility permission if prompted by macOS.")

            result_queue: queue.Queue = queue.Queue()
            stop_event = threading.Event()
            recording_thread: Optional[threading.Thread] = None
            active_mode: Optional[str] = None
            pending_mode: Optional[str] = None

            def _record(mode: str):
                audio = None
                if mode == "mic":
                    audio = mic.record_until_stop(
                        stop_event,
                        MAX_RECORD_SEC,
                        mic_device=resolve_audio_device(MIC_DEVICE_ENV, kind="input"),
                    )
                elif mode == "system":
                    audio = sysrec.record_until_stop(stop_event, MAX_RECORD_SEC)
                result_queue.put((mode, audio))

            def _start_recording(mode: str):
                nonlocal recording_thread, active_mode
                stop_event.clear()
                active_mode = mode
                recording_thread = threading.Thread(target=_record, args=(mode,), daemon=True)
                recording_thread.start()

            def _stop_recording():
                stop_event.set()

            while True:
                key = listener.next_press(timeout=0.1)
                if key is not None:
                    mic_key = listener.keys["mic"]
                    sys_key = listener.keys["sys"]
                    requested_mode = None
                    if key == mic_key:
                        requested_mode = "mic"
                    elif key == sys_key:
                        if sysrec.device is None:
                            logger.error("System audio device not configured. Set SYS_AUDIO_DEVICE, then retry.")
                            requested_mode = None
                        else:
                            requested_mode = "system"

                    if requested_mode:
                        if active_mode is None:
                            _start_recording(requested_mode)
                        elif active_mode == requested_mode:
                            _stop_recording()
                        else:
                            pending_mode = requested_mode
                            _stop_recording()

                try:
                    mode, audio = result_queue.get_nowait()
                except queue.Empty:
                    mode = None
                    audio = None

                if mode is None:
                    continue

                recording_thread = None
                active_mode = None

                if pending_mode:
                    next_mode = pending_mode
                    pending_mode = None
                    _start_recording(next_mode)
                    continue

                if audio is None:
                    continue

                if audio.shape[0] < int(MIN_CLIP_SEC * SAMPLE_RATE):
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
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if IS_LINUX and GRAB_KEYBOARD:
            try:
                kbd.ungrab()
            except Exception:
                pass
        logger.info("Exiting.")

if __name__ == "__main__":
    main()
