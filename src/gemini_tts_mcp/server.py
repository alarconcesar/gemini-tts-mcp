"""
Gemini TTS MCP Server — Multi-key TTS with API key rotation.

Exposes:
  - generate_speech: text -> WAV file path
  - list_voices: all 30 Gemini voices with gender, tone, description
  - list_voices_by_gender: filter by gender (male/female)
  - reload_keys: refresh API key pool from disk/env
  - pool_status: check configured key count

Install: pip install mcp google-genai
Run:     python -m gemini_tts_mcp.server
"""
from __future__ import annotations
import logging
import os
import sys

# Make the package importable when running from source (not installed via pip)
_src = os.path.join(os.path.dirname(__file__), "..")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

try:
    from fastmcp import FastMCP
except ImportError:  # mcp<2, ruta original
    from mcp.server.fastmcp import FastMCP
from gemini_tts_mcp.tts_rotator import GeminiTTSRotator

logger = logging.getLogger("gemini_tts_mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

mcp = FastMCP("gemini-tts", instructions="Multi-key Gemini TTS with API key rotation")
rotator = GeminiTTSRotator()

# === Complete voice catalog ===
# Gemini TTS voices are multilingual — any voice speaks any language the model supports.
# Language is determined by the input text, not by the voice selection.
VOICES = [
    {"name": "Achernar",    "gender": "female",   "tone": "Soft",        "desc": "Gentle, mellow tone"},
    {"name": "Achird",      "gender": "male",     "tone": "Friendly",    "desc": "Warm, approachable tone"},
    {"name": "Algieba",     "gender": "male",     "tone": "Smooth",      "desc": "Polished, fluid delivery"},
    {"name": "Algenib",     "gender": "male",     "tone": "Gravelly",    "desc": "Rough, textured quality"},
    {"name": "Alnilam",     "gender": "male",     "tone": "Firm",        "desc": "Steady, resolute delivery"},
    {"name": "Aoede",       "gender": "female",   "tone": "Breezy",      "desc": "Casual, relaxed delivery"},
    {"name": "Autonoe",     "gender": "female",   "tone": "Bright",      "desc": "Clear, vibrant expression"},
    {"name": "Callirrhoe",  "gender": "female",   "tone": "Easy-going",  "desc": "Laid-back, comfortable style"},
    {"name": "Charon",      "gender": "male",     "tone": "Informative", "desc": "Educational, explanatory style"},
    {"name": "Despina",     "gender": "female",   "tone": "Smooth",      "desc": "Refined, elegant tone"},
    {"name": "Enceladus",   "gender": "male",     "tone": "Breathy",     "desc": "Soft, airy quality"},
    {"name": "Erinome",     "gender": "female",   "tone": "Clear",       "desc": "Crisp, distinct articulation"},
    {"name": "Fenrir",      "gender": "male",     "tone": "Excitable",   "desc": "Energetic, animated expression"},
    {"name": "Gacrux",      "gender": "female",   "tone": "Mature",      "desc": "Experienced, seasoned quality"},
    {"name": "Iapetus",     "gender": "male",     "tone": "Clear",       "desc": "Precise, well-articulated"},
    {"name": "Kore",        "gender": "female",   "tone": "Firm",        "desc": "Assertive, confident delivery"},
    {"name": "Laomedeia",   "gender": "female",   "tone": "Upbeat",      "desc": "Positive, energetic style"},
    {"name": "Leda",        "gender": "female",   "tone": "Youthful",    "desc": "Young-sounding, fresh voice"},
    {"name": "Orus",        "gender": "male",     "tone": "Firm",        "desc": "Strong, authoritative tone"},
    {"name": "Puck",        "gender": "male",     "tone": "Upbeat",      "desc": "Cheerful, enthusiastic tone"},
    {"name": "Pulcherrima", "gender": "female",   "tone": "Forward",     "desc": "Direct, straightforward style"},
    {"name": "Rasalgethi",  "gender": "male",     "tone": "Informative", "desc": "Educational, instructive"},
    {"name": "Sadachbia",   "gender": "male",     "tone": "Lively",      "desc": "Energetic, spirited expression"},
    {"name": "Sadaltager",  "gender": "male",     "tone": "Knowledgeable","desc": "Expert, well-informed tone"},
    {"name": "Schedar",     "gender": "male",     "tone": "Even",        "desc": "Balanced, consistent tone"},
    {"name": "Sulafat",     "gender": "female",   "tone": "Warm",        "desc": "Comforting, affectionate quality"},
    {"name": "Umbriel",     "gender": "male",     "tone": "Easy-going",  "desc": "Relaxed, conversational"},
    {"name": "Vindemiatrix","gender": "female",   "tone": "Gentle",      "desc": "Soft, kind delivery"},
    {"name": "Zephyr",      "gender": "female",   "tone": "Bright",      "desc": "High energy, clear articulation"},
    {"name": "Zubenelgenubi","gender": "male",    "tone": "Casual",      "desc": "Informal, conversational"},
]


def _voices_table(voices: list[dict]) -> str:
    """Format a list of voices as a readable markdown table."""
    lines = ["| Voice | Gender | Tone | Description |", "|-------|--------|------|-------------|"]
    for v in sorted(voices, key=lambda x: x["name"]):
        icon = "♂" if v["gender"] == "male" else "♀"
        lines.append(f"| {v['name']} | {icon} {v['gender'].capitalize()} | {v['tone']} | {v['desc']} |")
    return "\n".join(lines)


@mcp.tool()
def generate_speech(
    text: str,
    voice_name: str = "Puck",
    style_instruction: str = "",
    pitch_factor: float = 1.0,
    audio_format: str = "wav",
    model: str | None = None,
    output_path: str | None = None,
) -> str:
    """Generate speech audio from text using Gemini TTS.

    Audio is post-processed automatically: trailing noise-burst cleanup,
    loudness normalization, subtle presence EQ, and 48 kHz output.

    Rotates across all configured API keys and falls back between models.
    Gemini TTS voices are multilingual — any voice speaks the language
    of the input text automatically.

    Args:
        text: The text content to vocalize.
        voice_name: Gemini voice (use list_voices to browse). Default: Puck.
        style_instruction: Optional speaking-style hint prepended to text,
                           e.g. "speak softly", "cheerfully", "in a calm tone".
        pitch_factor: Pitch adjustment. 1.0 = no change, >1 = higher, <1 = lower.
        audio_format: Target audio format: "wav", "mp3", "ogg", "m4a", "flac". Default: wav.
        model: Model override. Default: gemini-3.8-flash-tts, falls back to
               gemini-3.1-flash-tts-preview and gemini-2.5-flash-preview-tts.
        output_path: Optional custom output path for the audio file.

    Returns:
        Absolute path to the generated audio file, or error description.
    """
    # Validate voice
    valid = {v["name"].lower() for v in VOICES}
    if voice_name.lower() not in valid:
        close = [v["name"] for v in VOICES if v["name"].lower().startswith(voice_name.lower()[:3])]
        hint = f" Did you mean {close[0]}?" if close else f" Use list_voices to see all {len(VOICES)} voices."
        return f"❌ Unknown voice '{voice_name}'.{hint}"

    try:
        result = rotator.generate_speech(
            text=text,
            voice_name=voice_name,
            style_instruction=style_instruction,
            pitch_factor=pitch_factor,
            audio_format=audio_format,
            model=model,
            output_path=output_path,
        )
        return f"✅ Audio generated: {result}"
    except Exception as e:
        return f"❌ {e}"


@mcp.tool()
def list_voices(filter_gender: str | None = None, filter_tone: str | None = None) -> str:
    """List all Gemini TTS voices with gender, tone, and description.

    Optionally filter by gender and/or tone. Gemini TTS voices are
    multilingual — any voice speaks the language of the input text.

    Args:
        filter_gender: Filter by "male", "female" (case-insensitive).
        filter_tone: Filter by tone name, e.g. "Soft", "Bright", "Firm", "Warm",
                     "Upbeat", "Clear", "Smooth", "Informative", "Easy-going", etc.
                     (case-insensitive).

    Returns:
        A formatted table of matching voices.
    """
    result = VOICES

    if filter_gender:
        gender = filter_gender.lower().strip()
        if gender == "male":
            result = [v for v in result if v["gender"] == "male"]
        elif gender == "female":
            result = [v for v in result if v["gender"] == "female"]
        elif gender in ("neutral", "any", ""):
            pass
        else:
            return f"❌ Invalid gender '{filter_gender}'. Use 'male' or 'female'."

    if filter_tone:
        tone = filter_tone.lower().strip()
        result = [v for v in result if v["tone"].lower() == tone]

    if not result:
        msg = "No voices match your filters."
        if filter_tone:
            tones = sorted({v["tone"] for v in VOICES})
            msg += f" Available tones: {', '.join(tones)}"
        return msg

    header = f"**{len(result)} voice(s) found**"
    if filter_gender:
        header += f" (gender: {filter_gender})"
    if filter_tone:
        header += f" (tone: {filter_tone})"
    return header + "\n\n" + _voices_table(result)


@mcp.tool()
def list_voices_by_gender(gender: str) -> str:
    """List Gemini TTS voices filtered by gender.

    Args:
        gender: "male" or "female" (case-insensitive).

    Returns:
        A formatted table of matching voices.
    """
    return list_voices(filter_gender=gender)


@mcp.tool()
def reload_keys() -> str:
    """Reload API keys from ~/.gemini-tts-mcp/keys.json or env vars."""
    count = rotator.reload_keys()
    if count > 0:
        return f"✅ {count} API key(s) loaded."
    return "⚠️  No keys found. Create ~/.gemini-tts-mcp/keys.json or set GEMINI_API_KEY."


@mcp.tool()
def pool_status() -> str:
    """Check how many API keys are currently configured."""
    return f"Pool: {rotator.pool_size} API key(s) configured."


def main():
    mcp.run()


if __name__ == "__main__":
    main()
