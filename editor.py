import subprocess
import os.path
import moviepy.editor as mpe
from typing import List
import argparse

base_prompt = """
you are a world class video editor. your job is to take the media contained in the provided directory and create 
a "final cut" edit satisfying the given prompt. the media contained in the provided directory will already be converted
to useable text, including dialog, and screen descriptors. the final cut you create should be in the same directory as the
segments saved in text form as "_final.txt". the text form should use the same template outlined in `/example`. please also feel free to reference /example to see an example of how to go about editing a video.
---
prompt:
---
"""

def extract_between(text: str, start: str, end: str) -> List[str]:
    results = []
    i = 0
    while i < len(text):
        start_idx = text.find(start, i)
        if start_idx == -1:
            break
        end_idx = text.find(end, start_idx + 1)
        if end_idx == -1:
            break
        results.append(text[start_idx + 1:end_idx])
        i = end_idx + 1
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Editor - edit step")
    parser.add_argument("prompt")
    parser.add_argument('--index_dir', default="./", help="The input directory of indexed videos")
    parser.add_argument('--out', default="./out.mp4", help="The out directory for the edited video")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(os.path.join(args.index_dir, "_final.txt")):
        subprocess.run([
            "codex",
            "--approval-mode", "full-auto",
            "--quiet",
            "--dangerously-auto-approve-everything",
            base_prompt+args.prompt+f" ---- directory: {args.index_dir}"
        ])

    final_cut = open(os.path.join(args.index_dir, "_final.txt"), "r")
    
    audio = []
    video = []

    context = ""
    for line in final_cut.readlines():
        if "Audio" in line or "Video" in line or "[Edited]" in line:
            context = line
            continue

        if not "[" in line:
            continue

        file_path = os.path.join(args.index_dir, extract_between(line, "]", "[")[0].strip() + ".txt")
        video_path = ""
        with open(file_path, "r") as raw_file:
            video_path = raw_file.readlines()[0].strip()[7:-1]

        timestamps = extract_between(line, "[", "]")
        master_timestamp = timestamps[0].strip().split("-")
        subclip_timestart = timestamps[1].strip()

        duration = round(float(master_timestamp[1].strip())-float(master_timestamp[0].strip()),2)
        
        if "Audio" in context:
            audio.append(mpe.AudioFileClip(video_path).subclip(float(subclip_timestart), float(subclip_timestart)+duration))
        
        if "Video" in context:
            print(float(subclip_timestart), float(subclip_timestart)+duration)
            video.append(mpe.VideoFileClip(video_path).subclip(float(subclip_timestart), float(subclip_timestart)+duration))

    final = mpe.concatenate_videoclips(video, method='compose')
    if audio:
        final = final.set_audio(mpe.concatenate_audioclips(audio))

    final.write_videofile(args.out)

if __name__ == "__main__":
    main()