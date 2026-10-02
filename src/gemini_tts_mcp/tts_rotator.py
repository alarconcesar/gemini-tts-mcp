"""
Gemini TTS Rotator — Multi-key TTS generation engine.

Loads Gemini API keys from ~/.gemini-tts-mcp/keys.json or env var,
then rotates across them with model fallback.

All voice/style/pitch/format parameters are explicit — no baked-in persona.
"""
import array
import base64
import json
import logging
import math
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
    "mp3": ["-c:a", "libmp3lame", "-b:a", "256k"],
    "ogg": ["-c:a", "libopus", "-b:a", "32k"],
    "m4a": ["-c:a", "aac", "-b:a", "128k"],
    "flac": ["-c:a", "flac"],
}

# Post-processing chain applied to every generation: rumble cleanup, broadcast
# loudness (I=-16 LUFS), a touch of presence, and 48 kHz output so MP3 is not
# stuck at the 24 kHz (MPEG-2) bitrate ceiling.
QUALITY_CHAIN = (
    "highpass=f=60,"
    "loudnorm=I=-16:TP=-1.5:LRA=11,"
    "equalizer=f=3000:t=q:w=1:g=1.5,"
    "equalizer=f=6500:t=h:w=1:g=2.5,"
    "aresample=48000:resampler=soxr:precision=28"
)

FADE_SECONDS = 0.25

# Gemini 3.8 TTS family: new Interactions API. Input text is a verbatim
# transcript; sustained delivery directions go in speech_metadata.style.
INTERACTIONS_MODEL_PREFIX = "gemini-3.8"


def is_interactions_model(model: str) -> bool:
    return model.startswith(INTERACTIONS_MODEL_PREFIX)


def synthesize_interactions(client, model: str, text: str, voice_name: str,
                            style_instruction: str = "") -> bytes:
    """Synthesize with the Interactions API (Gemini 3.8 TTS family).

    Returns the model's default WAV file bytes (24 kHz mono, RIFF header) —
    do NOT wrap these in another WAV header.
    """
    part: dict = {"type": "text", "text": text}
    if style_instruction.strip():
        part["annotations"] = [
            {"type": "speech_metadata", "style": style_instruction.strip()}
        ]
    interaction = client.interactions.create(
        model=model,
        input=[{"type": "user_input", "content": [part]}],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": voice_name}]},
    )
    audio = getattr(interaction, "output_audio", None)
    if audio is None or not audio.data:
        raise RuntimeError("Interactions API returned no audio")
    return base64.b64decode(audio.data)


def synthesize_generate_content(client, model: str, contents: str,
                                voice_name: str) -> bytes:
    """Legacy models (<= gemini-3.1-flash-tts-preview): generateContent API.

    Returns headerless raw PCM (24 kHz mono s16).
    """
    response = client.models.generate_content(
        model=model,
        contents=contents,
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
    return response.candidates[0].content.parts[0].inline_data.data


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


def detect_trailing_artifact(input_wav: str) -> Optional[float]:
    """Detect the full-scale noise burst some Gemini TTS models append at the
    end of the audio (a ~0.13-0.4 s run of clipped noise right before EOF).

    Returns the time (seconds) to cut at, or None when no artifact is found.
    """
    try:
        with wave.open(input_wav) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return None  # only analyze the mono s16 model output
            n = w.getnframes()
            sr = w.getframerate()
            frames = w.readframes(n)
    except Exception as e:
        logger.warning("Artifact detection skipped: %s", e)
        return None

    if sr <= 0:
        return None
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if len(samples) < sr:
        return None

    def _rms(seg) -> float:
        return math.sqrt(sum(x * x for x in seg) / len(seg)) if len(seg) else 0.0

    win = max(1, int(0.010 * sr))  # 10 ms analysis windows
    tail_len = min(len(samples), int(0.5 * sr))
    tail_off = len(samples) - tail_len
    clips = [i for i, x in enumerate(samples[tail_off:]) if abs(x) >= 32000]
    if len(clips) < 10:
        return None
    # Artifact signature: clipped samples running to (nearly) the very end
    if (tail_len - clips[-1]) > int(0.15 * sr):
        return None
    first_clip = tail_off + clips[0]

    # Walk backwards from the first clipped sample while the signal stays loud
    # (through the noise burst and its ramp) until the quiet gap before it.
    loud_rms = 200.0
    quiet_gap_rms = 300.0
    max_back = int(0.6 * sr)
    pos = first_clip
    found_quiet = False
    while pos - win >= 0 and (first_clip - pos) < max_back:
        if _rms(samples[pos - win:pos]) < loud_rms:
            found_quiet = True
            break
        pos -= win
    onset = pos

    cut = None
    if found_quiet:
        gap = samples[max(0, onset - int(0.10 * sr)):onset]
        if _rms(gap) < quiet_gap_rms:
            cut = (onset - int(0.02 * sr)) / sr
    if cut is None:
        # No clear quiet gap: cut just before the clipped run as a fallback
        cut = (first_clip - int(0.05 * sr)) / sr
    return cut if cut > 0.05 else None


def apply_pitch_and_format(
    input_wav: str,
    output_path: str,
    pitch_factor: float = 1.0,
    audio_format: str = "wav",
) -> bool:
    """Clean up (trailing burst), apply loudness/EQ chain, pitch shift and
    convert to the target format using ffmpeg."""
    fmt = audio_format.lower().strip().lstrip(".")
    ffmpeg_args = SUPPORTED_FORMATS.get(fmt, ["-c:a", "pcm_s16le"])

    cut = detect_trailing_artifact(input_wav)

    try:
        with wave.open(input_wav) as w:
            src_rate = w.getframerate()
    except Exception:
        src_rate = 24000

    af_filters = []
    if pitch_factor != 1.0:
        af_filters.append(f"asetrate={src_rate}*{pitch_factor},atempo=1/{pitch_factor}")
    if cut is not None:
        af_filters.append(f"atrim=0:{cut:.3f}")
        af_filters.append("asetpts=N/SR/TB")
    af_filters.append(QUALITY_CHAIN)
    if cut is not None:
        fade_start = max(0.0, cut - FADE_SECONDS)
        af_filters.append(
            f"afade=t=out:st={fade_start:.3f}:d={min(FADE_SECONDS, cut):.3f}"
        )

    cmd = ["ffmpeg", "-y", "-i", input_wav, "-af", ",".join(af_filters)]
    cmd.extend(ffmpeg_args)
    cmd.append(output_path)

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
        "gemini-3.8-flash-tts",
        "gemini-3.8-flash-lite-tts",
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

        temp_wav = output_path + ".temp.wav"

        # Build ordered, deduplicated list of models to try
        preferred = [model] if model else []
        models_to_try = list(dict.fromkeys(preferred + self.MODELS))

        # Older models interpret a style prefix inside plain text; the 3.8
        # family keeps the transcript verbatim and takes style separately.
        payload = f"{style_instruction.strip()}: {text}" if style_instruction else text

        for target_model in models_to_try:
            total_keys = len(self.api_keys)
            for attempt in range(total_keys):
                idx = (self.current_index + attempt) % total_keys
                api_key = self.api_keys[idx]

                try:
                    client = genai.Client(api_key=api_key)
                    if is_interactions_model(target_model):
                        try:
                            wav_bytes = synthesize_interactions(
                                client, target_model, text, voice_name,
                                style_instruction,
                            )
                            with open(temp_wav, "wb") as f:
                                f.write(wav_bytes)
                        except Exception as e:
                            # Hiccup on this key: retry via legacy generate_content
                            logger.warning(
                                "Interactions API failed on %s (%s); "
                                "falling back to generate_content",
                                target_model, e,
                            )
                            data = synthesize_generate_content(
                                client, target_model, text, voice_name
                            )
                            if not save_wave_file(temp_wav, data):
                                raise RuntimeError("Could not write temp WAV")
                    else:
                        data = synthesize_generate_content(
                            client, target_model, payload, voice_name
                        )
                        if not save_wave_file(temp_wav, data):
                            raise RuntimeError("Could not write temp WAV")

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
