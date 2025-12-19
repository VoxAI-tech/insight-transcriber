![Faster Whisper Logo](https://5ccaof7hvfzuzf4p.public.blob.vercel-storage.com/banner-pjbGKw0buxbWGhMVC165Gf9qgqWo7I.jpeg)

[Faster Whisper](https://github.com/guillaumekln/faster-whisper) is designed to process audio files using various Whisper models, with options for transcription formatting, language translation and more.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-faster_whisper)](https://www.runpod.io/console/hub/runpod-workers/worker-faster_whisper)

---

## Models

- tiny
- base
- small
- medium
- large-v1
- large-v2
- large-v3
- distil-large-v2
- distil-large-v3
- turbo

### Custom Models

You can use custom HuggingFace models by setting the `WHISPER_CUSTOM_MODELS` environment variable (comma-separated):

```bash
WHISPER_CUSTOM_MODELS=VoxAI/whisper-large-v3-ej-au-20250922-ct2,VoxAI/whisper-large-v3-bk-pl-20250910
```

## Input

The handler supports two modes: **single audio** and **dual-channel** transcription.

### Single Audio Mode

| Input                               | Type  | Description                                                                                                                                                            |
| ----------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio`                             | Path  | URL to audio file (supports .wav, .mp3, .raw PCM)                                                                                                                      |
| `audio_base64`                      | str   | Base64-encoded audio file                                                                                                                                              |
| `model`                             | str   | Choose a Whisper model. Choices: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "distil-large-v2", "distil-large-v3", "turbo". Default: "base" |
| `transcription`                     | str   | Choose the format for the transcription. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                  |
| `translate`                         | bool  | Translate the text to English when set to True. Default: False                                                                                                         |
| `translation`                       | str   | Choose the format for the translation. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                    |
| `language`                          | str   | Language spoken in the audio, specify None to perform language detection. Default: None                                                                                |
| `temperature`                       | float | Temperature to use for sampling. Default: 0                                                                                                                            |
| `best_of`                           | int   | Number of candidates when sampling with non-zero temperature. Default: 5                                                                                               |
| `beam_size`                         | int   | Number of beams in beam search, only applicable when temperature is zero. Default: 5                                                                                   |
| `patience`                          | float | Optional patience value to use in beam decoding. Default: 1.0                                                                                                          |
| `length_penalty`                    | float | Optional token length penalty coefficient (alpha). Default: 1.0                                                                                                        |
| `suppress_tokens`                   | str   | Comma-separated list of token ids to suppress during sampling. Default: "-1"                                                                                           |
| `initial_prompt`                    | str   | Optional text to provide as a prompt for the first window. Default: None                                                                                               |
| `condition_on_previous_text`        | bool  | If True, provide the previous output of the model as a prompt for the next window. Default: True                                                                       |
| `temperature_increment_on_fallback` | float | Temperature to increase when falling back when the decoding fails. Default: 0.2                                                                                        |
| `compression_ratio_threshold`       | float | If the gzip compression ratio is higher than this value, treat the decoding as failed. Default: 2.4                                                                    |
| `logprob_threshold`                 | float | If the average log probability is lower than this value, treat the decoding as failed. Default: -1.0                                                                   |
| `no_speech_threshold`               | float | If the probability of the token is higher than this value, consider the segment as silence. Default: 0.6                                                               |
| `enable_vad`                        | bool  | If True, use the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model. Default: True                |
| `word_timestamps`                   | bool  | If True, include word timestamps in the output. Default: False                                                                                                         |

### Dual-Channel Mode (Insights)

For transcribing dual-channel audio (SPK/MIC) from S3:

| Input                | Type | Description                                      |
| -------------------- | ---- | ------------------------------------------------ |
| `session_id`         | str  | Session identifier (required for dual-channel)   |
| `brand_id`           | str  | Brand identifier (optional)                      |
| `device_id`          | str  | Device identifier (optional)                     |
| `source_spk_s3_path` | str  | S3 path to speaker channel audio (.raw PCM)      |
| `source_mic_s3_path` | str  | S3 path to microphone channel audio (.raw PCM)   |
| `spk_audio_base64`   | str  | Base64-encoded speaker audio (alternative to S3) |
| `mic_audio_base64`   | str  | Base64-encoded mic audio (alternative to S3)     |

All other parameters from single audio mode are also supported.

## Examples

### Single Audio Mode

```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "model": "base"
  }
}
```

Output:

```json
{
  "segments": [...],
  "detected_language": "en",
  "transcription": "Four score and seven years ago...",
  "translation": null,
  "device": "cuda",
  "model": "base"
}
```

### Dual-Channel Mode

```json
{
  "input": {
    "session_id": "0822c6f2-e866-4bc8-900d-b1c60b1eea88",
    "brand_id": "Burger King",
    "device_id": "0349UD36",
    "source_spk_s3_path": "s3://bucket/path/to/spk.raw",
    "source_mic_s3_path": "s3://bucket/path/to/mic.raw",
    "model": "large-v3"
  }
}
```

Output:

```json
{
  "session_id": "0822c6f2-e866-4bc8-900d-b1c60b1eea88",
  "brand_id": "Burger King",
  "device_id": "0349UD36",
  "model": "large-v3",
  "spk_transcription": {
    "segments": [...],
    "detected_language": "cs",
    "transcription": "...",
    "device": "cuda",
    "model": "large-v3"
  },
  "mic_transcription": {
    "segments": [...],
    "detected_language": "cs",
    "transcription": "...",
    "device": "cuda",
    "model": "large-v3"
  },
  "source_spk_s3_path": "s3://bucket/path/to/spk.raw",
  "source_mic_s3_path": "s3://bucket/path/to/mic.raw"
}
```

## Local Testing

Start the local API server:

```bash
uv run ./src/rp_handler_insights.py --rp_serve_api
```

### Test Single Audio Mode

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
      "model": "base"
    }
  }'
```

### Test Dual-Channel Mode

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "session_id": "0822c6f2-e866-4bc8-900d-b1c60b1eea88",
      "brand_id": "Burger King",
      "device_id": "0349UD36",
      "source_spk_s3_path": "s3://vox-ai-audio/brand=Burger King/provider=NEXEO/device=0349UD36/year=2025/month=12/day=18/hour=8/20251218081607_0822c6f2-e866-4bc8-900d-b1c60b1eea88_lane1_spk.raw",
      "source_mic_s3_path": "s3://vox-ai-audio/brand=Burger King/provider=NEXEO/device=0349UD36/year=2025/month=12/day=18/hour=8/20251218081607_0822c6f2-e866-4bc8-900d-b1c60b1eea88_lane1_mic.raw",
      "model": "base"
    }
  }'
```

Interactive API docs available at: http://localhost:8000/docs
