import os
import av
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ExtractionConfig:
    frames_per_segment: int = 10          
    max_frames_per_segment: int = 16     # ← cap for very long segments
    output_format: str = "jpg"
    jpg_quality: int = 85
    min_segment_duration: float = 0.5
    output_dir: str = "output/frames"


@dataclass
class SegmentFrames:
    video_id: str
    segment_index: int
    timestamp_start: float
    timestamp_end: float
    sentence: str
    frame_paths: list = field(default_factory=list)
    frame_timestamps: list = field(default_factory=list)


class FrameExtractor:
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def extract_from_video(self, video_path, video_id, timestamps, sentences):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        total_duration = self._get_duration(video_path)
        results = []

        for seg_idx, (ts, sentence) in enumerate(zip(timestamps, sentences)):
            t_start = max(0.0, float(ts[0]))
            t_end   = min(float(ts[1]), total_duration)
            duration = t_end - t_start

            if duration < self.config.min_segment_duration:
                print(f"  [skip] Segment {seg_idx} too short ({duration:.2f}s)")
                continue

            frame_times = self._sample_frame_times(t_start, t_end)
            frames      = self._extract_frames_at(video_path, frame_times)

            seg_dir = Path(self.config.output_dir) / video_id / f"seg_{seg_idx:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            segment = SegmentFrames(
                video_id=video_id,
                segment_index=seg_idx,
                timestamp_start=t_start,
                timestamp_end=t_end,
                sentence=sentence,
            )

            for frame_num, (frame, ft) in enumerate(zip(frames, frame_times)):
                path = self._save_frame(frame, seg_dir, frame_num)
                segment.frame_paths.append(str(path))
                segment.frame_timestamps.append(ft)

            results.append(segment)

        return results

    def _get_duration(self, video_path):
        with av.open(video_path) as container:
            # Try stream duration first, fall back to container duration
            stream = container.streams.video[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            return float(container.duration) / 1_000_000

    def _extract_frames_at(self, video_path, target_times):
        frames = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]

            for t in target_times:
                # Seek to just before the target time
                seek_ts = int(t / stream.time_base)
                container.seek(seek_ts, stream=stream, backward=True)

                frame_found = None
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        frame_t = float(frame.pts * stream.time_base)
                        if frame_t >= t - 0.1:
                            frame_found = frame.to_ndarray(format="rgb24")
                            break
                    if frame_found is not None:
                        break

                if frame_found is not None:
                    frames.append(frame_found)
                else:
                    print(f"  [warn] No frame found near {t:.2f}s")

        return frames

    def _sample_frame_times(self, t_start, t_end):
        duration = t_end - t_start
        n = max(1, min(int(duration / 4), 8))
        if n == 1:
            return [(t_start + t_end) / 2]
        step = (t_end - t_start) / (n - 1)
        return [t_start + i * step for i in range(n)]

    def _save_frame(self, frame_rgb, output_dir, frame_num):
        ext  = self.config.output_format
        path = output_dir / f"frame_{frame_num:03d}.{ext}"
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if ext == "jpg":
            cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpg_quality])
        else:
            cv2.imwrite(str(path), frame_bgr)
        return path