# Gemini TTS MCP

Multi-key Gemini TTS server (MCP) with automatic API key rotation and model fallback.

No baked-in voice, accent, or persona — all parameters are explicit.

## Features

- 🔄 **Key rotation** — pool of Gemini API keys; auto-rotates on quota/error
- 📉 **Model fallback** — tries `gemini-3.1-flash-tts-preview` → `gemini-2.5-flash-preview-tts`
- 🎙️ **Any voice** — choose from 9+ Gemini voices
- 🎚️ **Pitch control** — `pitch_factor` parameter (ffmpeg-based)
- 🧩 **MCP-native** — register in any MCP host (Claude Desktop, Hermes Agent, etc.)

## Quick Start

### 1. Install

```bash
pip install mcp google-genai
# or
uv pip install mcp google-genai
```

### 2. Set up API keys

Create `~/.gemini-tts-mcp/keys.json`:

```json
["AIzaSy...key1", "AIzaSy...key2"]
```

Or set an env var: `export GEMINI_API_KEY="AIzaSy...your_key"`

### 3. Run

```bash
python -m gemini_tts_mcp.server
# or
uv run python -m gemini_tts_mcp.server
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
| `generate_speech` | Text → WAV file |
| `list_voices` | List available Gemini voice names |
| `reload_keys` | Refresh API key pool from disk/env |
| `pool_status` | Check how many keys are configured |

### generate_speech parameters

| Param | Default | Description |
|-------|---------|-------------|
| `text` | (required) | Text to vocalize |
| `voice_name` | `Puck` | One of: Puck, Leda, Aoede, Charon, Fenrir, Kore, Rhea, Triton, Sterope |
| `style_instruction` | `""` | Speaking style e.g. "softly", "cheerfully", "in a calm tone" |
| `pitch_factor` | `1.0` | >1 = higher pitch, <1 = lower |
| `model` | `null` | Override model (auto fallback if omitted) |
| `output_path` | `null` | Custom WAV path |

## Key management

Keys are loaded from `~/.gemini-tts-mcp/keys.json` (a JSON array of strings), with env var fallback. The server rotates through the pool on each request, skipping failed keys per model. If all keys fail on one model, it falls back to the next model.

Use `reload_keys()` to refresh without restarting.

## Security

- **API keys never leave your machine** — the MCP server runs locally
- `.gitignore` excludes `keys.json`
- `keys.json.example` provided as a template

## License

MIT
