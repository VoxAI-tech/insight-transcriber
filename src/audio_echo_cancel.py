"""Audio-level echo cancellation using cross-correlation and subtraction.

Uses SPK as reference to remove echo bleed from MIC before transcription.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import tempfile


def read_audio(path: str) -> tuple[int, np.ndarray]:
    """Read audio file and return sample rate and normalized float32 array."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    return sr, audio


def write_audio(path: str, sr: int, audio: np.ndarray) -> None:
    """Write audio array to WAV file."""
    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, audio_int16)


def find_echo_delay(spk: np.ndarray, mic: np.ndarray, max_delay_samples: int = 8000) -> int:
    """
    Find the delay of SPK echo in MIC using cross-correlation.

    Args:
        spk: Speaker audio (reference)
        mic: Microphone audio (contains echo)
        max_delay_samples: Maximum delay to search (default 0.5s at 16kHz)

    Returns:
        Delay in samples (positive = MIC lags behind SPK)
    """
    # Use a chunk from the middle for correlation (more stable)
    chunk_size = min(len(spk), len(mic), 32000)  # ~2 seconds
    start = max(0, len(spk) // 2 - chunk_size // 2)

    spk_chunk = spk[start:start + chunk_size]
    mic_chunk = mic[start:start + chunk_size]

    # Cross-correlate
    correlation = signal.correlate(mic_chunk, spk_chunk, mode='full')

    # Find peak in valid delay range
    mid = len(correlation) // 2
    search_start = mid
    search_end = min(mid + max_delay_samples, len(correlation))

    peak_idx = search_start + np.argmax(np.abs(correlation[search_start:search_end]))
    delay = peak_idx - mid

    return delay


def estimate_echo_gain(
    spk: np.ndarray,
    mic: np.ndarray,
    delay: int,
    window_size: int = 4000
) -> float:
    """
    Estimate the gain of the echo (how loud SPK appears in MIC).

    Args:
        spk: Speaker audio
        mic: Microphone audio
        delay: Echo delay in samples
        window_size: Window for estimation

    Returns:
        Estimated echo gain (0-1)
    """
    if delay < 0:
        return 0.0

    # Align signals
    spk_aligned = spk[:-delay] if delay > 0 else spk
    mic_aligned = mic[delay:] if delay > 0 else mic

    min_len = min(len(spk_aligned), len(mic_aligned))
    spk_aligned = spk_aligned[:min_len]
    mic_aligned = mic_aligned[:min_len]

    # Estimate gain using least squares over windows
    gains = []
    for i in range(0, min_len - window_size, window_size):
        spk_win = spk_aligned[i:i + window_size]
        mic_win = mic_aligned[i:i + window_size]

        spk_power = np.sum(spk_win ** 2)
        if spk_power > 1e-6:  # Avoid division by zero
            # Least squares estimate: gain = (spk · mic) / (spk · spk)
            gain = np.sum(spk_win * mic_win) / spk_power
            if 0 < gain < 1:  # Valid echo gain range
                gains.append(gain)

    return np.median(gains) if gains else 0.0


def cancel_echo(
    spk: np.ndarray,
    mic: np.ndarray,
    delay: int | None = None,
    gain: float | None = None,
    adaptive: bool = True,
) -> np.ndarray:
    """
    Remove echo from MIC using SPK as reference.

    Args:
        spk: Speaker audio (reference)
        mic: Microphone audio (contains echo)
        delay: Echo delay in samples (auto-detect if None)
        gain: Echo gain (auto-estimate if None)
        adaptive: Use adaptive filtering for better results

    Returns:
        Echo-cancelled MIC audio
    """
    # Auto-detect delay
    if delay is None:
        delay = find_echo_delay(spk, mic)

    if delay <= 0:
        return mic  # No echo detected

    # Auto-estimate gain
    if gain is None:
        gain = estimate_echo_gain(spk, mic, delay)

    if gain <= 0:
        return mic  # No significant echo

    # Create delayed SPK signal
    spk_delayed = np.zeros_like(mic)
    if delay < len(mic):
        copy_len = min(len(spk), len(mic) - delay)
        spk_delayed[delay:delay + copy_len] = spk[:copy_len]

    if adaptive:
        # Adaptive NLMS filter for better echo cancellation
        return adaptive_echo_cancel(spk_delayed, mic, gain)
    else:
        # Simple subtraction
        cancelled = mic - gain * spk_delayed
        return cancelled


def adaptive_echo_cancel(
    reference: np.ndarray,
    signal_with_echo: np.ndarray,
    initial_gain: float,
    filter_length: int = 128,
    step_size: float = 0.1,
) -> np.ndarray:
    """
    Adaptive NLMS echo cancellation.

    Args:
        reference: Delayed reference signal (SPK)
        signal_with_echo: Signal containing echo (MIC)
        initial_gain: Initial gain estimate
        filter_length: Adaptive filter length
        step_size: NLMS step size (0-1)

    Returns:
        Echo-cancelled signal
    """
    n = len(signal_with_echo)
    output = np.zeros(n)

    # Initialize filter with initial gain estimate
    w = np.zeros(filter_length)
    w[0] = initial_gain

    eps = 1e-6  # Regularization

    for i in range(filter_length, n):
        # Get reference window
        x = reference[i - filter_length:i][::-1]

        # Filter output (estimated echo)
        echo_estimate = np.dot(w, x)

        # Error (desired signal = mic - echo)
        error = signal_with_echo[i] - echo_estimate
        output[i] = error

        # NLMS update
        norm = np.dot(x, x) + eps
        w = w + step_size * error * x / norm

    # Copy beginning unchanged
    output[:filter_length] = signal_with_echo[:filter_length]

    return output


def process_echo_cancellation(
    spk_path: str,
    mic_path: str,
    output_path: str | None = None,
) -> str:
    """
    Process MIC audio to remove SPK echo.

    Args:
        spk_path: Path to SPK WAV file
        mic_path: Path to MIC WAV file
        output_path: Output path (auto-generate if None)

    Returns:
        Path to echo-cancelled MIC audio
    """
    # Read audio files
    sr_spk, spk = read_audio(spk_path)
    sr_mic, mic = read_audio(mic_path)

    if sr_spk != sr_mic:
        raise ValueError(f"Sample rate mismatch: SPK={sr_spk}, MIC={sr_mic}")

    # Match lengths
    min_len = min(len(spk), len(mic))
    spk = spk[:min_len]
    mic = mic[:min_len]

    # Cancel echo
    mic_clean = cancel_echo(spk, mic)

    # Write output
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

    write_audio(output_path, sr_mic, mic_clean)

    return output_path
