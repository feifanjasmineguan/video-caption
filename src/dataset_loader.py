from datasets import load_dataset
from typing import Iterator


def stream_activitynet(split: str = "train") -> Iterator[dict]:
    """
    Stream ActivityNet Captions without downloading the full 42GB.
    Yields one sample at a time with keys: video_id, timestamps, sentence, duration.
    """
    ds = load_dataset(
        "HuggingFaceM4/ActivityNet_Captions",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    for sample in ds:
        yield {
            "video_id": sample["video_id"],
            "timestamps": sample["timestamps"],
            "sentences": sample["sentence"],   # list of strings
            "duration": sample.get("duration"),
            # "video" column contains raw bytes — only access if you need it
        }


def get_video_bytes(sample: dict) -> bytes:
    """Extract raw video bytes from a sample (only call when needed)."""
    return sample["video"]["bytes"]