# Gemini TTS MCP

Multi-key Gemini TTS server (MCP) with automatic API key rotation and model fallback.

No baked-in voice, accent, or persona — all parameters are explicit.

## Features

- 🔄 **Key rotation** — pool of Gemini API keys; auto-rotates on quota/error
- 📉 **Model fallback** — tries `gemini-3.8-flash-tts` → `gemini-3.8-flash-lite-tts` → `gemini-3.1-flash-tts-preview` → `gemini-2.5-flash-preview-tts`
- 🎭 **Style the right way per model** — the 3.8 family uses the new Interactions API with `speech_metadata` (transcript stays verbatim, inline vocal tags like `<sigh>` work); older models get the style prepended
- 🧼 **Auto cleanup + mastering** — trims the trailing noise burst some models append, normalizes loudness (I=-16 LUFS), subtle presence EQ, 48 kHz output
- 🗣️ **30 voices** — full catalog with gender, tone, and description
- 🔍 **Filter by gender/tone** — `list_voices(male/female)` or by tone name
- 🎚️ **Pitch control** — `pitch_factor` parameter (ffmpeg-based)
- 🌐 **Multilingual** — any voice speaks the language of your input text
- 🧩 **MCP-native** — register in any MCP host (Claude Desktop, Hermes Agent, etc.)

## Quick Start

### 1. Install

> ⚠️ **Not available on PyPI** — install from source (see below).
> `pip install mcp google-genai` installs **dependencies only**, not this project.

**Option A — Clone the repo (recommended):**

```bash
git clone https://github.com/alarconcesar/gemini-tts-mcp.git
cd gemini-tts-mcp
pip install mcp google-genai
```

**Option B — From anywhere (source must be in PYTHONPATH):**

```bash
pip install mcp google-genai
# then clone & run from the repo directory
```

### 2. Set up API keys

Create `~/.gemini-tts-mcp/keys.json`:

```json
["AIzaSy...key1", "AIzaSy...key2"]
```

Or set an env var: `export GEMINI_API_KEY="AIzaSy...your_key"`

### 3. Run (from the repo directory)

```bash
cd gemini-tts-mcp
python -m gemini_tts_mcp.server
```

### 4. Register in any MCP host

**Hermes Agent** (`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  gemini-tts:
    command: "python"
    args: ["-m", "gemini_tts_mcp.server"]
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gemini-tts": {
      "command": "python",
      "args": ["-m", "gemini_tts_mcp.server"]
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `generate_speech` | Text → audio file (WAV, MP3, OGG, M4A, FLAC) |
| `list_voices` | Browse all 30 voices (filter by gender/tone) |
| `list_voices_by_gender` | Shortcut: `voice "male"` or `voice "female"` |
| `reload_keys` | Refresh API key pool from disk/env |
| `pool_status` | Check how many keys are configured |

### generate_speech parameters

| Param | Default | Description |
|-------|---------|-------------|
| `text` | (required) | Text to vocalize |
| `voice_name` | `Puck` | Any of the 30 voices |
| `style_instruction` | `""` | Speaking style e.g. "softly", "cheerfully", "in a calm tone" |
| `pitch_factor` | `1.0` | >1 = higher pitch, <1 = lower |
| `audio_format` | `wav` | Output format: `wav`, `mp3`, `ogg`, `m4a`, `flac` |
| `model` | `null` | Override model (auto fallback if omitted) |
| `output_path` | `null` | Custom WAV path |

## Voice Catalog

All **30 voices** are multilingual — any voice speaks the language of your input text.

Female voices (16):

| Voice | Tone | Description |
|-------|------|-------------|
| Achernar | Soft | Gentle, mellow tone |
| Aoede | Breezy | Casual, relaxed delivery |
| Autonoe | Bright | Clear, vibrant expression |
| Callirrhoe | Easy-going | Laid-back, comfortable style |
| Despina | Smooth | Refined, elegant tone |
| Erinome | Clear | Crisp, distinct articulation |
| Gacrux | Mature | Experienced, seasoned quality |
| Kore | Firm | Assertive, confident delivery |
| Laomedeia | Upbeat | Positive, energetic style |
| Leda | Youthful | Young-sounding, fresh voice |
| Pulcherrima | Forward | Direct, straightforward style |
| Sterope | Forward | Bold, direct presentation |
| Sulafat | Warm | Comforting, affectionate quality |
| Vindemiatrix | Gentle | Soft, kind delivery |
| Zephyr | Bright | High energy, clear articulation |

Male voices (15):

| Voice | Tone | Description |
|-------|------|-------------|
| Achird | Friendly | Warm, approachable tone |
| Algieba | Smooth | Polished, fluid delivery |
| Algenib | Gravelly | Rough, textured quality |
| Alnilam | Firm | Steady, resolute delivery |
| Charon | Informative | Educational, explanatory style |
| Enceladus | Breathy | Soft, airy quality |
| Fenrir | Excitable | Energetic, animated expression |
| Iapetus | Clear | Precise, well-articulated |
| Orus | Firm | Strong, authoritative tone |
| Puck | Upbeat | Cheerful, enthusiastic tone |
| Rasalgethi | Informative | Educational, instructive |
| Sadachbia | Lively | Energetic, spirited expression |
| Sadaltager | Knowledgeable | Expert, well-informed tone |
| Schedar | Even | Balanced, consistent tone |
| Umbriel | Easy-going | Relaxed, conversational |
| Zubenelgenubi | Casual | Informal, conversational |

## Key management

Keys are loaded from `~/.gemini-tts-mcp/keys.json` (a JSON array of strings), with env var fallback. The server rotates through the pool on each request, skipping failed keys per model. If all keys fail on one model, it falls back to the next model.

Use `reload_keys()` to refresh without restarting.

## Security

- **API keys never leave your machine** — the MCP server runs locally
- `.gitignore` excludes `keys.json`
- `keys.json.example` provided as a template

## License

MIT
