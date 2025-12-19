"""Input schema for RunPod insights handler.

Supports both:
- Single audio transcription (audio/audio_base64)
- Dual-channel transcription (source_spk_s3_path/source_mic_s3_path)
"""

INPUT_VALIDATIONS = {
    # Single audio input (standard mode)
    "audio": {
        "type": str,
        "required": False,
        "default": None,
    },
    "audio_base64": {
        "type": str,
        "required": False,
        "default": None,
    },
    # Dual-channel input (insights mode)
    "session_id": {
        "type": str,
        "required": False,
        "default": None,
    },
    "brand_id": {
        "type": str,
        "required": False,
        "default": None,
    },
    "device_id": {
        "type": str,
        "required": False,
        "default": None,
    },
    "source_spk_s3_path": {
        "type": str,
        "required": False,
        "default": None,
    },
    "source_mic_s3_path": {
        "type": str,
        "required": False,
        "default": None,
    },
    "spk_audio_base64": {
        "type": str,
        "required": False,
        "default": None,
    },
    "mic_audio_base64": {
        "type": str,
        "required": False,
        "default": None,
    },
    # Common parameters
    "model": {
        "type": str,
        "required": False,
        "default": "base",
    },
    "transcription": {
        "type": str,
        "required": False,
        "default": "plain_text",
    },
    "translate": {
        "type": bool,
        "required": False,
        "default": False,
    },
    "translation": {
        "type": str,
        "required": False,
        "default": "plain_text",
    },
    "language": {
        "type": str,
        "required": False,
        "default": None,
    },
    "temperature": {
        "type": float,
        "required": False,
        "default": 0,
    },
    "best_of": {
        "type": int,
        "required": False,
        "default": 5,
    },
    "beam_size": {
        "type": int,
        "required": False,
        "default": 5,
    },
    "patience": {
        "type": float,
        "required": False,
        "default": 1.0,
    },
    "length_penalty": {
        "type": float,
        "required": False,
        "default": 1.0,
    },
    "suppress_tokens": {
        "type": str,
        "required": False,
        "default": "-1",
    },
    "initial_prompt": {
        "type": str,
        "required": False,
        "default": None,
    },
    "condition_on_previous_text": {
        "type": bool,
        "required": False,
        "default": True,
    },
    "temperature_increment_on_fallback": {
        "type": float,
        "required": False,
        "default": 0.2,
    },
    "compression_ratio_threshold": {
        "type": float,
        "required": False,
        "default": 2.4,
    },
    "logprob_threshold": {
        "type": float,
        "required": False,
        "default": -1.0,
    },
    "no_speech_threshold": {
        "type": float,
        "required": False,
        "default": 0.6,
    },
    "enable_vad": {
        "type": bool,
        "required": False,
        "default": True,
    },
    "word_timestamps": {
        "type": bool,
        "required": False,
        "default": False,
    },
    # Echo removal settings (for dual-channel mode)
    "remove_echo": {
        "type": bool,
        "required": False,
        "default": True,
    },
    "echo_method": {
        "type": str,
        "required": False,
        "default": "audio",  # "audio" (better) or "text" (n-gram)
    },
    "echo_word_threshold": {
        "type": float,
        "required": False,
        "default": 0.5,
    },
    "echo_char_threshold": {
        "type": float,
        "required": False,
        "default": 0.6,
    },
    "echo_time_tolerance": {
        "type": float,
        "required": False,
        "default": 1.0,
    },
}
