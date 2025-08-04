import cv2
import base64
import threading
import argparse
from openai import OpenAI
from typing import List, Tuple
import scenedetect as sd
from moviepy.editor import VideoFileClip, AudioFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math
from dataclasses import dataclass
import moviepy.editor as mpe
import numpy as np
import pytesseract
from PIL import Image
import random
import time
import functools

def debug_log(message: str):
    if os.getenv('EDIT_DEBUG') == "true":
        print(f"[DEBUG] {message}")

def retry_exponential(
    max_retries=5,
    base_delay=0.5,
    max_delay=10,
    jitter=True,
    multiplier=2.0,
):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except:
                    if attempt == max_retries:
                        raise
                    
                    next_delay = min(delay, max_delay)                    
                    if jitter:
                        next_delay = random.uniform(0, next_delay)
                    
                    debug_log(f"Retrying {fn.__name__} after {next_delay:.2f} seconds (attempt {attempt}/{max_retries})")    
                    time.sleep(next_delay)
                    delay *= multiplier
                
            raise RuntimeError("unreachable")
        return wrapped
    return decorator

@dataclass(frozen=True)
class SubtitleTimestamp:
    def __init__(self, start: int, end: int, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text        

class TimestampSegmenter:
    def __init__(self, segments: List) -> None:
        self.timestamps = []
        self._start = None
        self._current_segment = []
        self._end = 0
        self._sum_char_length = 0

        self.segments = segments

    def _finalize_current_segment(self, is_last=False):
        if len(self._current_segment) == 0:
            return

        start = 0 if not self._start else self._start
        if self._end - start < 1.5 and not is_last:
            # too short, keep going.
            return

        if len(self.timestamps) > 0 and len(self._current_segment) <= 3 and not is_last:
            return

        text_segment = " ".join(self._current_segment)

        self._sum_char_length += len(text_segment)
        self.timestamps.append((text_segment, self._start, self._end))
        self._current_segment = []
        self._start = None

    def _process_word(self, word):
        text = word["text"].upper()

        if "." in text or "?" in text or "," in text:
            self._finalize_current_segment()

        self._current_segment.append(text)
        self._end = word["end"]

    def process(self) -> List[SubtitleTimestamp]:
        for segment in self.segments:
            for w in segment["words"]:
                if not self._start:
                    self._start = w["start"]

                if w["end"] + 1 >= self._end:
                    self._finalize_current_segment()

                self._process_word(w)

            self._finalize_current_segment()

        self._finalize_current_segment(is_last=True)

        return self.timestamps

def purify_audio_timestamps(
    timestamps, time_offset: int
) -> List[SubtitleTimestamp]:
    final = []
    for idx, t in enumerate(timestamps):
        text, start, end = t

        if idx > 0 and not start:
            start = timestamps[idx - 1][2]
            print("changed start", start)

        final.append(
            SubtitleTimestamp(
                start + time_offset, end + time_offset, text
            )
        )

    return final

def timestamps_from_paths(audio_paths: List[str], silence = 0.5) -> List[SubtitleTimestamp]:
    timestamps = []
    offset = 0
    for path in audio_paths:
        try:
            timestamps += purify_audio_timestamps(
                audio_to_timestamps(path), offset
            )
        except Exception as e:
            print("failed for some reason", e)
            timestamps += purify_audio_timestamps(
                [], offset
            )

        audio_clip = mpe.AudioFileClip(path)
        offset += audio_clip.duration + silence

    return timestamps

@retry_exponential(max_retries=1000, max_delay=60*20)
def audio_to_timestamps(file_path: str):
    import whisper_timestamped as whisper
    
    audio = whisper.load_audio(file_path)
    model = whisper.load_model("base")

    result = whisper.transcribe(model, audio, language="en", beam_size=5, best_of=5)

    return TimestampSegmenter(result["segments"]).process()

@retry_exponential(max_retries=10, max_delay=60*20)
def gather_frames_at_seconds(video_path: str, timestamps: List[int]) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = []

    for timestamp in timestamps:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            raise ValueError(f"Failed to capture frame at {timestamp} sec.")

        _, buffer = cv2.imencode('.jpg', frame)
        ret.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    return ret

def save_audio_segment(video_path: str, timestamp: tuple[int, int], output_dir: str = ".") -> list[str]:
    clip = VideoFileClip(video_path)
    
    start, end = timestamp

    if end <= start:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    
    audio_clip = clip.subclip(start, end).audio
    output_path = f"{output_dir}/segment.wav"
    audio_clip.write_audiofile(output_path, codec="pcm_s16le", verbose=False, logger=None)
    
    clip.close()
    return output_path

def timestamp_video_scenes(video_path: str):
    video = sd.open_video(video_path)
    sm = sd.SceneManager()
    sm.add_detector(sd.ContentDetector(threshold=27.0))
    sm.detect_scenes(video)

    scenes = sm.get_scene_list()
    if len(scenes) == 0:
        return [(0, math.floor(video.duration.get_seconds()))]
    
    return [(math.floor(start.get_seconds()), math.floor(end.get_seconds())) for start, end in scenes]

def detect_motion_events(video_path: str, scene_timestamp: tuple[int, int], fps: float = 1.0, threshold: float = 30.0) -> List[tuple[float, float, float]]:    
    start_s, end_s = scene_timestamp
    clip = VideoFileClip(video_path).subclip(start_s, end_s).resize(width=320)
    prev_gray = None
    events: List[tuple[float, float, float]] = []

    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8')):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff > threshold:
                t0 = i / fps
                t1 = min((i + 1) / fps, end_s - start_s)
                events.append((t0, t1, mean_diff))
        prev_gray = gray
    
    try:
        clip.reader.close()
        clip.audio.reader.close_proc()
    except Exception:
        pass
    
    return events

def detect_audio_spikes(
    video_path: str,
    scene_timestamp: Tuple[float, float],
    spike_ratio: float = 3.0,
    width: float = 1.5,
) -> List[Tuple[float, float]]:    
    start_s, end_s = scene_timestamp
    fps = 22050.0

    audioclip = AudioFileClip(video_path).subclip(start_s, end_s)
    try:
        arr = audioclip.to_soundarray(fps=int(fps))
    except Exception:
        audioclip.close()
        return []

    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    mag = np.abs(arr)
    if mag.size == 0:
        audioclip.close()
        return []
    
    smooth_window = int(0.1 * fps)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        mag = np.convolve(mag, kernel, mode="same")

    median = float(np.median(mag))
    std = float(np.std(mag))
    thresh = median + spike_ratio * std

    above = mag > thresh
    idxs = np.where(above)[0]
    events: List[Tuple[float, float]] = []
    if idxs.size:
        segments = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
        for seg in segments:
            i0, i1 = seg[0], seg[-1]
            t0 = i0 / fps
            t1 = (i1 + 1) / fps
            events.append((t0, t1))

    if not events:
        audioclip.close()
        return []

    merged: List[Tuple[float, float]] = []
    current_start, current_end = events[0]
    for next_start, next_end in events[1:]:
        if next_start - current_end <= width:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    audioclip.close()
    return merged

def detect_ocr_events(video_path: str, scene_timestamp: tuple[int, int], fps: float = 0.5) -> List[tuple[float, float, str]]:    
    start_s, end_s = scene_timestamp
    clip = VideoFileClip(video_path).subclip(start_s, end_s).resize(width=640)
    events: List[tuple[float, float, str]] = []
    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8')):
        img = Image.fromarray(frame)
        text = pytesseract.image_to_string(img).strip().replace('\n', ' ')
        if text:
            t0 = i / fps
            t1 = min((i + 1) / fps, end_s - start_s)
            events.append((t0, t1, text))
    try:
        clip.reader.close()
        clip.audio.reader.close_proc()
    except Exception:
        pass
    return events

@retry_exponential(max_retries=20, max_delay=60*2)
def openai_describe_scene(start_frame_base64: str, end_frame_base64: str, scene_transcribed_audio: str) -> str:
    client = OpenAI()

    prompt = """
you are a world class video scene classifier.

given the following two frames of the same scene, determine what is going on in the larger scene.

only return what is going on in the scene. below, is some additional context from the audio in the scene
---
""" + scene_transcribed_audio

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": prompt },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{start_frame_base64}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{end_frame_base64}",
                    },
                ],
            }
        ],
    )

    return response.output_text

@retry_exponential(max_retries=20, max_delay=60*2)
def openai_label_event(start_b64: str, end_b64: str, audio_ctx: str, model: str) -> str:    
    client = OpenAI()
    prompt = (
        """
You are a video event classifier. Given two frames from a short video clip and the transcript of any speech in it (or empty), return a concise label (1-4 words) describing the key action, e.g. 'gunshot', 'explosion', 'character running', or 'nothing' if there's no identifiable event.\nTranscript:\n"""
        + audio_ctx + "\n"""
    )
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{start_b64}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{end_b64}"},
            ]
        }]
    )
    label = response.output_text.strip()
    return '' if label.lower() in ('nothing', 'none', '') else label

@dataclass(frozen=True)
class SceneDescriptorEvent:
    start: int
    end: int
    label: str

@dataclass(frozen=True)
class SceneDescriptor:
    scene_start: int
    scene_end: int
    video_path: str
    description: str
    events: List[SceneDescriptorEvent]
    dialog: List[SceneDescriptorEvent]

def parse_video_scene(scene_timestamp, args, cuda_semaphore) -> SceneDescriptor:
    if scene_timestamp[1] is None or scene_timestamp[0] is None:
        debug_log(f"something is None.. weird {scene_timestamp[1]}:{scene_timestamp[0]}")
        return None

    if abs(scene_timestamp[1] - scene_timestamp[0]) < 0.1:
        debug_log(f"skipping timestamp less than 1s {scene_timestamp[1]}:{scene_timestamp[0]}")
        return None
    
    frames_base64 = gather_frames_at_seconds(args.video_path, [scene_timestamp[0], scene_timestamp[1]-1])
        
    motion_events = detect_motion_events(args.video_path, scene_timestamp) if not args.skip_motion_events else []
    audio_events = detect_audio_spikes(args.video_path, scene_timestamp) if not args.skip_audio_events else []
    ocr_events = detect_ocr_events(args.video_path, scene_timestamp) if not args.skip_ocr_events else []

    audio_segment_path = save_audio_segment(args.video_path, scene_timestamp, "./segments")
    
    dialog = []
    subtitles = []
    if not args.skip_dialog:        
        with cuda_semaphore:
            subtitles = audio_to_timestamps(audio_segment_path)
        
        for subtitle in subtitles:
            if subtitle[1] is None or scene_timestamp[0] is None:
                print("something is None", subtitle[1], scene_timestamp[0])
                continue
            
            dialog.append(SceneDescriptorEvent(subtitle[1]+scene_timestamp[0], subtitle[2]+scene_timestamp[0], subtitle[0]))
    
    def get_event_transcript(start_abs, end_abs):
        texts = []
        for subtitle in subtitles:
            text, start_rel, end_rel = subtitle
            if start_rel is None:
                debug_log(f"start_rel somehow None {subtitle}")
                continue
            seg_start = start_rel + scene_timestamp[0]
            seg_end = end_rel + scene_timestamp[0]
            if seg_start < end_abs and seg_end > start_abs:
                texts.append(text)
        
        return " ".join(texts)

    labeled_events: List[SceneDescriptorEvent] = []

    for val in motion_events + audio_events + ocr_events:
        t0 = val[0]
        t1 = val[1]
        
        start_abs = t0 + scene_timestamp[0]
        end_abs = t1 + scene_timestamp[0]
        frames = gather_frames_at_seconds(args.video_path, [start_abs, max(start_abs + 0.1, end_abs - 0.01)])
        audio_ctx = get_event_transcript(start_abs, end_abs)
        label = openai_label_event(frames[0], frames[1], audio_ctx, args.event_model)
        
        if label:
            labeled_events.append(SceneDescriptorEvent(start_abs, end_abs, label))
    
    audio_transcribed = ""
    for subtitle in subtitles:
        audio_transcribed += subtitle[0]+" "

    scene_description = openai_describe_scene(frames_base64[0], frames_base64[1], audio_transcribed)
    
    return SceneDescriptor(scene_timestamp[0], scene_timestamp[1], args.video_path, scene_description, sorted(labeled_events, key=lambda x: x.start), dialog)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Editor - preprocess step")
    parser.add_argument("video_path")
    parser.add_argument('--out_dir', default="./", help="The out directory for saving the indexed file")
    parser.add_argument('--skip_ocr_events', default=False, action='store_true', help="Flag used for skipping finding events found from on screen text")
    parser.add_argument('--skip_motion_events', default=False, action='store_true', help="Flag used for skipping events derived from high motion")
    parser.add_argument('--skip_audio_events', default=False, action='store_true', help="Flag used for skipping events found from audio spikes")
    parser.add_argument('--skip_dialog', default=False, action='store_true', help="Flag used for skipping transcribing dialog from the video")
    parser.add_argument('--max_scene_workers', type=int, default=1, help="Number of concurrent workers that can be used for scene processing")
    parser.add_argument('--subset_audio_workers', type=int, default=1, help="Number of workers from the scene worker pool that can access audio transcription resources (uses CUDA)")
    parser.add_argument('--event_model', default="gpt-4.1", help="The GPT OpenAI (image supported) used for labeling events")

    return parser.parse_args()

def main():
    args = parse_args()
    
    scenes = timestamp_video_scenes(args.video_path)
    audio_cuda_semaphore = threading.Semaphore(args.subset_audio_workers)

    completed_scenes: List[SceneDescriptor] = []
    with ThreadPoolExecutor(max_workers=args.max_scene_workers) as executor:
        futures = {executor.submit(parse_video_scene, scene, args, audio_cuda_semaphore): scene for scene in scenes}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                completed_scenes.append(res)

    save_file = os.path.splitext(os.path.basename(args.video_path))[0]
    with open(os.path.join(args.out_dir, f"{save_file}.txt"), "w+") as file:
        file.write(f"[File: {args.video_path}]\n\n")
        
        for scene in sorted(completed_scenes, key=lambda x: x.scene_start):
            if scene is None:
                continue
            
            file.write(f"[Scene: {scene.scene_start}s - {scene.scene_end}s]\n")
            file.write(f"{scene.description}\n\n")
            
            if len(scene.events): file.write("EVENTS\n")
            for event in scene.events:
                file.write(f"[{event.start:.2f} - {event.end:.2f}] {event.label}\n")

            if len(scene.events): file.write("\n")        
            if len(scene.dialog): file.write("DIALOG\n")
            
            for dialog in scene.dialog:
                file.write(f"[{dialog.start:.2f} - {dialog.end:.2f}] {dialog.label}\n")
        
            file.write(f"\n")

if __name__ == "__main__":
    main()