"""
Gemini TTS Rotator — Multi-key TTS generation engine.

Loads Gemini API keys from ~/.gemini-tts-mcp/keys.json or env var,
then rotates across them with model fallback.

All voice/style/pitch/format parameters are explicit — no baked-in persona.
"""
import json
import logging
import os
import subprocess
import time
import wave
from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger("gemini_tts_mcp")

DEFAULT_KEYS_PATH = os.path.expanduser("~/.gemini-tts-mcp/keys.json")
DEFAULT_CACHE_DIR = os.path.expanduser("~/.gemini-tts-mcp/cache")

SUPPORTED_FORMATS = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
    "ogg": ["-c:a", "libopus", "-b:a", "32k"],
    "m4a": ["-c:a", "aac", "-b:a", "128k"],
    "flac": ["-c:a", "flac"],
}


def save_wave_file(filename: str, pcm: bytes, channels: int = 1,
                   rate: int = 24000, sample_width: int = 2) -> bool:
    try:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return True
    except Exception as e:
        logger.error("Error writing WAV: %s", e)
        return False


def apply_pitch_and_format(
    input_wav: str,
    output_path: str,
    pitch_factor: float = 1.0,
    audio_format: str = "wav",
) -> bool:
    """Apply pitch shift and convert to target format using ffmpeg."""
    fmt = audio_format.lower().strip().lstrip(".")
    ffmpeg_args = SUPPORTED_FORMATS.get(fmt, ["-c:a", "pcm_s16le"])

    # Build audio filter
    af_filters = []
    if pitch_factor != 1.0:
        af_filters.append(f"asetrate=24000*{pitch_factor},atempo=1/{pitch_factor}")

    cmd = ["ffmpeg", "-y", "-i", input_wav]
    if af_filters:
        cmd.extend(["-af", ",".join(af_filters)])
    cmd.extend(ffmpeg_args)
    cmd.append(output_path)

    # If no pitch shift and output is wav, no need for ffmpeg if input_wav == output_path
    if pitch_factor == 1.0 and fmt == "wav" and input_wav == output_path:
        return True

    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and os.path.exists(output_path):
            if input_wav != output_path and os.path.exists(input_wav):
                try:
                    os.remove(input_wav)
                except OSError:
                    pass
            return True
        logger.error("ffmpeg conversion failed: %s",
                     result.stderr.decode(errors="ignore"))
        return False
    except Exception as e:
        logger.error("ffmpeg exception: %s", e)
        return False


def load_api_keys(keys_path: Optional[str] = None) -> list[str]:
    path = keys_path or DEFAULT_KEYS_PATH
    keys = []

    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        keys = [k.strip() for k in raw if k and k.strip()]
        if keys:
            logger.info("Loaded %d key(s) from %s", len(keys), path)
            return keys

    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        logger.info("Loaded API key from env var")
        return [env_key]

    logger.warning("No Gemini API keys found (%s or env)", path)
    return []


class GeminiTTSRotator:
    """Rotate across API keys + model fallback for Gemini TTS."""

    MODELS = [
        "gemini-3.1-flash-tts-preview",
        "gemini-2.5-flash-preview-tts",
    ]

    def __init__(self, keys_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.keys_path = keys_path or DEFAULT_KEYS_PATH
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_keys = load_api_keys(self.keys_path)
        self.current_index = 0

    @property
    def pool_size(self) -> int:
        return len(self.api_keys)

    def reload_keys(self) -> int:
        self.api_keys = load_api_keys(self.keys_path)
        return len(self.api_keys)

    def generate_speech(
        self,
        text: str,
        voice_name: str = "Puck",
        style_instruction: str = "",
        pitch_factor: float = 1.0,
        audio_format: str = "wav",
        model: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate TTS audio, rotating across API keys and models.

        Args:
            text: The text to speak.
            voice_name: Gemini voice name (e.g. Puck, Leda, Aoede, Charon).
            style_instruction: Optional speaking-style instruction
                (e.g. "softly", "cheerfully"). Prepended to text.
            pitch_factor: 1.0 = no change. >1 = higher, <1 = lower.
            audio_format: Output format: "wav", "mp3", "ogg", "m4a", "flac".
            model: Specific model override.
            output_path: Where to save the file. Auto-named if omitted.

        Returns: Absolute path to the generated audio file.
        """
        if not self.api_keys:
            reloaded = self.reload_keys()
            if reloaded == 0:
                raise RuntimeError(
                    "No API keys available. Add keys to ~/.gemini-tts-mcp/keys.json "
                    "or set GEMINI_API_KEY / GOOGLE_API_KEY env var."
                )

        fmt = audio_format.lower().strip().lstrip(".")
        if fmt not in SUPPORTED_FORMATS:
            fmt = "wav"

        if not output_path:
            output_path = os.path.join(
                self.cache_dir, f"tts_{int(time.time())}.{fmt}"
            )
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            # Infer format from extension if explicitly provided in path
            ext = os.path.splitext(output_path)[1].lower().lstrip(".")
            if ext in SUPPORTED_FORMATS:
                fmt = ext

        temp_wav = output_path + ".temp.wav" if fmt != "wav" or pitch_factor != 1.0 else output_path

        # Build ordered, deduplicated list of models to try
        preferred = [model] if model else []
        models_to_try = list(dict.fromkeys(preferred + self.MODELS))

        payload = f"{style_instruction.strip()}: {text}" if style_instruction else text

        for target_model in models_to_try:
            total_keys = len(self.api_keys)
            for attempt in range(total_keys):
                idx = (self.current_index + attempt) % total_keys
                api_key = self.api_keys[idx]

                try:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=target_model,
                        contents=payload,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name=voice_name,
                                    )
                                )
                            ),
                        ),
                    )
                    data = response.candidates[0].content.parts[0].inline_data.data

                    if save_wave_file(temp_wav, data):
                        self.current_index = idx
                        apply_pitch_and_format(
                            temp_wav, output_path, pitch_factor, fmt
                        )
                        return os.path.abspath(output_path)

                except APIError as e:
                    logger.warning("Key #%d failed on %s: %s", idx + 1, target_model, e.message)
                    continue
                except Exception as e:
                    logger.warning("Key #%d failed on %s: %s", idx + 1, target_model, e)
                    continue

        raise RuntimeError("All API keys and models exhausted.")
