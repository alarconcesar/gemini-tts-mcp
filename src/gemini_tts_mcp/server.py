"""
Gemini TTS MCP Server — Multi-key TTS with API key rotation.

Exposes tools:
  - generate_speech: text → WAV file path
  - list_voices: available Gemini voices
  - reload_keys: refresh API key pool from disk/env

Install: pip install mcp google-genai
Run:     python -m gemini_tts_mcp.server
"""
from __future__ import annotations
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcp.server.fastmcp import FastMCP
from gemini_tts_mcp.tts_rotator import GeminiTTSRotator

logger = logging.getLogger("gemini_tts_mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

mcp = FastMCP("gemini-tts", instructions="Multi-key Gemini TTS with API key rotation")
rotator = GeminiTTSRotator()

# Known Gemini voice names
KNOWN_VOICES = [
    "Puck", "Leda", "Aoede", "Charon", "Fenrir",
    "Kore", "Rhea", "Triton", "Sterope",
]


@mcp.tool()
def generate_speech(
    text: str,
    voice_name: str = "Puck",
    style_instruction: str = "",
    pitch_factor: float = 1.0,
    model: str | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate speech audio from text using Gemini TTS.

    Rotates across all configured API keys and falls back between models.

    Args:
        text: The text content to vocalize.
        voice_name: Gemini voice name. Common: Puck, Leda, Aoede, Charon,
                    Fenrir, Kore, Rhea, Triton, Sterope. (default: Puck)
        style_instruction: Optional speaking-style hint prepended to text,
                           e.g. "speak softly", "cheerfully", "in a calm tone".
        pitch_factor: Pitch adjustment. 1.0 = no change, >1 = higher, <1 = lower.
        model: Model override. Default pool: gemini-3.1-flash-tts-preview,
               gemini-2.5-flash-preview-tts.
        output_path: Optional custom output path for the WAV file.

    Returns:
        Absolute path to the generated WAV file, or error description.
    """
    try:
        result = rotator.generate_speech(
            text=text,
            voice_name=voice_name,
            style_instruction=style_instruction,
            pitch_factor=pitch_factor,
            model=model,
            output_path=output_path,
        )
        return f"✅ Audio generated: {result}"
    except Exception as e:
        return f"❌ {e}"


@mcp.tool()
def list_voices() -> str:
    """List available Gemini TTS voice names."""
    return f"Available voices: {', '.join(KNOWN_VOICES)}"


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
