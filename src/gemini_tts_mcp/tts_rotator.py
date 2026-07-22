"""
Gemini TTS Rotator — Multi-key TTS generation engine.

Loads Gemini API keys from ~/.gemini-tts-mcp/keys.json or env var,
then rotates across them with model fallback.

All voice/style/pitch parameters are explicit — no baked-in persona.
"""
import os
import json
import wave
import subprocess
import logging
from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger("gemini_tts_mcp")

DEFAULT_KEYS_PATH = os.path.expanduser("~/.gemini-tts-mcp/keys.json")
DEFAULT_CACHE_DIR = os.path.expanduser("~/.gemini-tts-mcp/cache")


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


def apply_pitch_shift(input_wav: str, output_wav: str,
                      pitch_factor: float = 1.0) -> bool:
    if pitch_factor == 1.0:
        return True
    try:
        temp = input_wav + ".tmp.wav"
        cmd = [
            "ffmpeg", "-y", "-i", input_wav,
            "-af", f"asetrate=24000*{pitch_factor},atempo=1/{pitch_factor}",
            temp,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and os.path.exists(temp):
            os.replace(temp, output_wav)
            return True
        logger.error("ffmpeg pitch shift failed: %s",
                      result.stderr.decode(errors="ignore"))
        return False
    except Exception as e:
        logger.error("Pitch shift exception: %s", e)
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
        model: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate TTS audio, rotating across API keys and models.

        Args:
            text: The text to speak.
            voice_name: Gemini voice name (Puck, Leda, Aoede, Charon, etc.)
            style_instruction: Optional speaking-style instruction
                (e.g. "softly", "cheerfully"). Prepended to text.
            pitch_factor: 1.0 = no change. >1 = higher, <1 = lower.
            model: Specific model override (e.g. "gemini-3.1-flash-tts-preview").
            output_path: Where to save the WAV. Auto-named if omitted.

        Returns: Absolute path to the generated WAV file.
        """
        if not self.api_keys:
            reloaded = self.reload_keys()
            if reloaded == 0:
                raise RuntimeError(
                    "No API keys available. Add keys to ~/.gemini-tts-mcp/keys.json "
                    "or set GEMINI_API_KEY / GOOGLE_API_KEY env var."
                )

        import time
        if not output_path:
            output_path = os.path.join(
                self.cache_dir, f"tts_{int(time.time())}.wav"
            )
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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

                    if save_wave_file(output_path, data):
                        self.current_index = idx
                        if pitch_factor != 1.0:
                            apply_pitch_shift(output_path, output_path, pitch_factor)
                        return os.path.abspath(output_path)

                except APIError as e:
                    logger.warning("Key #%d failed on %s: %s", idx + 1, target_model, e.message)
                    continue
                except Exception as e:
                    logger.warning("Key #%d failed on %s: %s", idx + 1, target_model, e)
                    continue

        raise RuntimeError("All API keys and models exhausted.")
