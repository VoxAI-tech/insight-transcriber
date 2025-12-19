"""Echo removal using n-gram comparison between SPK and MIC channels.

SPK (speaker/employee) is clean, MIC (microphone/customer) has echo bleed.
We detect and remove segments from MIC that match SPK using n-gram overlap.
"""

import re
from dataclasses import dataclass


@dataclass
class Segment:
    """Transcription segment with timing."""
    start: float
    end: float
    text: str
    is_echo: bool = False


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text


def extract_word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Extract word-level n-grams from text."""
    words = normalize_text(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def extract_char_ngrams(text: str, n: int = 5) -> set[str]:
    """Extract character-level n-grams from text."""
    text = normalize_text(text).replace(' ', '')
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def ngram_overlap_ratio(ngrams1: set, ngrams2: set) -> float:
    """Calculate Jaccard-like overlap ratio between two n-gram sets."""
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    smaller = min(len(ngrams1), len(ngrams2))
    return intersection / smaller if smaller > 0 else 0.0


def times_overlap(
    start1: float, end1: float,
    start2: float, end2: float,
    tolerance: float = 0.5
) -> bool:
    """Check if two time ranges overlap with tolerance for echo delay."""
    # Extend ranges by tolerance to account for echo delay
    return not (end1 + tolerance < start2 or end2 + tolerance < start1)


def build_spk_ngram_index(
    spk_segments: list[dict],
    word_n: int = 3,
    char_n: int = 5
) -> list[dict]:
    """Build n-gram index for SPK segments for fast lookup."""
    indexed = []
    for seg in spk_segments:
        text = seg.get("text", "")
        indexed.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "word_ngrams": extract_word_ngrams(text, word_n),
            "char_ngrams": extract_char_ngrams(text, char_n),
        })
    return indexed


def detect_echo_segments(
    mic_segments: list[dict],
    spk_index: list[dict],
    word_n: int = 3,
    char_n: int = 5,
    word_threshold: float = 0.5,
    char_threshold: float = 0.6,
    time_tolerance: float = 1.0,
) -> list[dict]:
    """
    Detect echo segments in MIC by comparing with SPK.

    Args:
        mic_segments: MIC transcription segments
        spk_index: Pre-indexed SPK segments with n-grams
        word_n: Word n-gram size
        char_n: Character n-gram size
        word_threshold: Min word n-gram overlap to consider echo
        char_threshold: Min char n-gram overlap to consider echo
        time_tolerance: Time overlap tolerance in seconds

    Returns:
        MIC segments with 'is_echo' field added
    """
    results = []

    for mic_seg in mic_segments:
        mic_start = mic_seg["start"]
        mic_end = mic_seg["end"]
        mic_text = mic_seg.get("text", "")

        # Extract n-grams for this MIC segment
        mic_word_ngrams = extract_word_ngrams(mic_text, word_n)
        mic_char_ngrams = extract_char_ngrams(mic_text, char_n)

        is_echo = False
        max_word_overlap = 0.0
        max_char_overlap = 0.0

        # Compare with time-overlapping SPK segments
        for spk_seg in spk_index:
            if not times_overlap(
                mic_start, mic_end,
                spk_seg["start"], spk_seg["end"],
                time_tolerance
            ):
                continue

            # Calculate n-gram overlaps
            word_overlap = ngram_overlap_ratio(mic_word_ngrams, spk_seg["word_ngrams"])
            char_overlap = ngram_overlap_ratio(mic_char_ngrams, spk_seg["char_ngrams"])

            max_word_overlap = max(max_word_overlap, word_overlap)
            max_char_overlap = max(max_char_overlap, char_overlap)

            # Echo if either threshold exceeded
            if word_overlap >= word_threshold or char_overlap >= char_threshold:
                is_echo = True
                break

        results.append({
            **mic_seg,
            "is_echo": is_echo,
            "echo_word_overlap": max_word_overlap,
            "echo_char_overlap": max_char_overlap,
        })

    return results


def remove_echo_from_transcription(
    spk_result: dict,
    mic_result: dict,
    word_threshold: float = 0.5,
    char_threshold: float = 0.6,
    time_tolerance: float = 1.0,
) -> dict:
    """
    Remove echo segments from MIC transcription.

    Args:
        spk_result: SPK transcription result from Whisper
        mic_result: MIC transcription result from Whisper
        word_threshold: Min word n-gram overlap to consider echo
        char_threshold: Min char n-gram overlap to consider echo
        time_tolerance: Time overlap tolerance in seconds

    Returns:
        Updated MIC result with echo segments marked and filtered transcription
    """
    spk_segments = spk_result.get("segments", [])
    mic_segments = mic_result.get("segments", [])

    if not spk_segments or not mic_segments:
        return mic_result

    # Build SPK index
    spk_index = build_spk_ngram_index(spk_segments)

    # Detect echo in MIC segments
    mic_with_echo = detect_echo_segments(
        mic_segments,
        spk_index,
        word_threshold=word_threshold,
        char_threshold=char_threshold,
        time_tolerance=time_tolerance,
    )

    # Filter out echo segments for clean transcription
    clean_segments = [seg for seg in mic_with_echo if not seg["is_echo"]]
    clean_text = " ".join(seg.get("text", "").strip() for seg in clean_segments)

    return {
        **mic_result,
        "segments": mic_with_echo,  # All segments with is_echo flag
        "segments_clean": clean_segments,  # Only non-echo segments
        "transcription": mic_result.get("transcription", ""),  # Original
        "transcription_clean": clean_text,  # Echo removed
        "echo_segments_removed": sum(1 for seg in mic_with_echo if seg["is_echo"]),
    }
