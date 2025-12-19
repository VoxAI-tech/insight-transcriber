"""RunPod handler for audio transcription.

Supports both:
- Single audio transcription (audio/audio_base64)
- Dual-channel SPK/MIC transcription (source_spk_s3_path/source_mic_s3_path)
"""

import base64
import os
import subprocess
import tempfile
from typing import Any

import boto3
from botocore.client import BaseClient
import runpod
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

# Raw PCM format settings (s16le = signed 16-bit little-endian)
RAW_PCM_FORMAT = "s16le"
RAW_SAMPLE_RATE = 16000
RAW_CHANNELS = 1

from rp_schema_insights import INPUT_VALIDATIONS
from predict import Predictor
from echo_removal import remove_echo_from_transcription

MODEL = Predictor()
MODEL.setup()

_s3_client: BaseClient | None = None


def get_s3_client() -> BaseClient:
    """Get or create S3 client using 'dev' profile."""
    global _s3_client
    if _s3_client is None:
        session = boto3.Session(profile_name="dev")
        _s3_client = session.client("s3")
    return _s3_client


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse S3 path into bucket and key."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    path = s3_path[5:]
    bucket, *key_parts = path.split("/", 1)
    key = key_parts[0] if key_parts else ""
    return bucket, key


def convert_raw_to_wav(raw_path: str) -> str:
    """Convert raw PCM audio to WAV format using ffmpeg."""
    wav_path = raw_path.replace(".raw", ".wav")
    if wav_path == raw_path:
        wav_path = raw_path + ".wav"

    cmd = [
        "ffmpeg", "-y",
        "-f", RAW_PCM_FORMAT,
        "-ar", str(RAW_SAMPLE_RATE),
        "-ac", str(RAW_CHANNELS),
        "-i", raw_path,
        wav_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.unlink(raw_path)
    return wav_path


def download_from_s3(s3_path: str) -> str:
    """Download file from S3 to a temporary file, converting raw PCM if needed."""
    bucket, key = parse_s3_path(s3_path)
    s3 = get_s3_client()
    ext = os.path.splitext(key)[1] or ".raw"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        s3.download_fileobj(bucket, key, tmp)
        raw_path = tmp.name

    if ext == ".raw":
        return convert_raw_to_wav(raw_path)
    return raw_path


def base64_to_tempfile(base64_data: str, suffix: str = ".wav") -> str:
    """Convert base64 encoded audio to a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(base64.b64decode(base64_data))
        return tmp.name


def run_single_audio_job(
    job_input: dict[str, Any], job_id: str
) -> dict[str, Any]:
    """Handle single audio transcription (standard mode)."""
    if job_input.get("audio"):
        audio_path = download_files_from_urls(job_id, [job_input["audio"]])[0]
        if audio_path.endswith(".raw"):
            audio_path = convert_raw_to_wav(audio_path)
    else:
        audio_path = base64_to_tempfile(job_input["audio_base64"])

    result = MODEL.predict(
        audio=audio_path,
        model_name=job_input["model"],
        transcription=job_input["transcription"],
        translate=job_input["translate"],
        translation=job_input["translation"],
        language=job_input["language"],
        temperature=job_input["temperature"],
        best_of=job_input["best_of"],
        beam_size=job_input["beam_size"],
        patience=job_input["patience"],
        length_penalty=job_input["length_penalty"],
        suppress_tokens=job_input.get("suppress_tokens", "-1"),
        initial_prompt=job_input["initial_prompt"],
        condition_on_previous_text=job_input["condition_on_previous_text"],
        temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
        compression_ratio_threshold=job_input["compression_ratio_threshold"],
        logprob_threshold=job_input["logprob_threshold"],
        no_speech_threshold=job_input["no_speech_threshold"],
        enable_vad=job_input["enable_vad"],
        word_timestamps=job_input["word_timestamps"],
    )

    rp_cleanup.clean(["input_objects"])
    return result


def run_dual_channel_job(job_input: dict[str, Any]) -> dict[str, Any]:
    """Handle dual-channel SPK/MIC transcription (insights mode)."""
    has_spk = job_input.get("source_spk_s3_path") or job_input.get("spk_audio_base64")
    has_mic = job_input.get("source_mic_s3_path") or job_input.get("mic_audio_base64")

    if not has_spk and not has_mic:
        return {"error": "Must provide audio for at least one channel (SPK or MIC)"}

    results: dict[str, Any] = {
        "session_id": job_input.get("session_id"),
        "brand_id": job_input.get("brand_id"),
        "device_id": job_input.get("device_id"),
        "model": job_input["model"],
        "spk_transcription": None,
        "mic_transcription": None,
    }

    temp_files: list[str] = []

    try:
        if has_spk:
            if job_input.get("source_spk_s3_path"):
                spk_path = download_from_s3(job_input["source_spk_s3_path"])
                results["source_spk_s3_path"] = job_input["source_spk_s3_path"]
            else:
                spk_path = base64_to_tempfile(job_input["spk_audio_base64"])
            temp_files.append(spk_path)

            results["spk_transcription"] = MODEL.predict(
                audio=spk_path,
                model_name=job_input["model"],
                transcription=job_input["transcription"],
                translate=job_input["translate"],
                translation=job_input["translation"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"],
            )

        if has_mic:
            if job_input.get("source_mic_s3_path"):
                mic_path = download_from_s3(job_input["source_mic_s3_path"])
                results["source_mic_s3_path"] = job_input["source_mic_s3_path"]
            else:
                mic_path = base64_to_tempfile(job_input["mic_audio_base64"])
            temp_files.append(mic_path)

            results["mic_transcription"] = MODEL.predict(
                audio=mic_path,
                model_name=job_input["model"],
                transcription=job_input["transcription"],
                translate=job_input["translate"],
                translation=job_input["translation"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"],
            )

    finally:
        for tmp in temp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        rp_cleanup.clean(["input_objects"])

    # Apply echo removal if enabled and both channels transcribed
    if (
        job_input.get("remove_echo", True)
        and results.get("spk_transcription")
        and results.get("mic_transcription")
    ):
        results["mic_transcription"] = remove_echo_from_transcription(
            spk_result=results["spk_transcription"],
            mic_result=results["mic_transcription"],
            word_threshold=job_input.get("echo_word_threshold", 0.5),
            char_threshold=job_input.get("echo_char_threshold", 0.6),
            time_tolerance=job_input.get("echo_time_tolerance", 1.0),
        )

    return results


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Main handler - routes to single or dual-channel mode."""
    job_input = job["input"]

    validation_result = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validation_result:
        return {"error": validation_result["errors"]}
    job_input = validation_result["validated_input"]

    has_single_audio = job_input.get("audio") or job_input.get("audio_base64")
    has_dual_channel = (
        job_input.get("source_spk_s3_path")
        or job_input.get("source_mic_s3_path")
        or job_input.get("spk_audio_base64")
        or job_input.get("mic_audio_base64")
    )

    if has_single_audio and has_dual_channel:
        return {"error": "Cannot mix single audio and dual-channel inputs"}

    if has_single_audio:
        return run_single_audio_job(job_input, job["id"])
    elif has_dual_channel:
        return run_dual_channel_job(job_input)
    else:
        return {"error": "Must provide audio input (audio, audio_base64, or S3 paths)"}


runpod.serverless.start({"handler": run_job})
