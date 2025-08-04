# LLM Video Editor

This repository adapts off-the-shelf OpenAI LLMs for video editing purposes. This includes a simple technique for indexing video content enabling an LLM to reason about video and a _prompt-able_ video editor that uses indexed video content.

## Quick Start

```bash
python3 index.py "./video.mp4"
python3 editor.py "Create a highlight reel of the most exciting parts"
```

## Background

Back in April, I wrote a node for [Nura Flow](https://flow.nuraml.com) that could [_"find interesting segments"_](https://flow.nuraml.com/docs/changelog/index#May%2028,%202025) by transcribing the dialog found in audio and prompting an LLM to return a list of timestamps for the most interesting parts. A user could then pipe the timestamps to a node that could trim the segments out, then join them together-- resulting in an edited video of only the most interesting parts in the provided source material.

![Nura Flow Interesting Segments](https://flow.nuraml.com/images/changelog-agent-1.png)

This approach worked well with content that was dialog rich, such as podcasts or Ted talks, but would yield poor results with video that lacked dialog such as video game gameplay or nature shots.

After spending some time working on [Vibe-1](https://vibeone.gg), an asynchronous coding agent that uses reasoning models for code generation. I wanted to try adapting reasoning models for video editing.

This repository is split into two parts:

1) **Indexing**: Extracting key features from video, in text form to a specified directory. These features are derived from interpreted motion, audio and frame queues (OCR & visuals) found in video.

2) **Editing**: After the video content is indexed, we employ a reasoning model to reason about a "final cut" of the indexed video(s) given a prompt. Then apply a simple parser to the text form of the reasoning models final cut to edit the video together.

## Installation

This project uses OpenAI GPT models for both the indexing and editing process. You can [register an API key here](https://platform.openai.com/settings).

After you are registered, configure your API key by setting in an environment variable.

```bash
export OPENAI_API_KEY=<your registered key>
```

This project also relies on [OpenAI Codex CLI](https://github.com/openai/codex) for directory reasoning.

```bash
npm install -g @openai/codex
```

Lastly install the python requirements.txt.

```bash
pip install -r requirements.txt
```

I strongly recommend creating a python virtual-environment for installation.

```bash
python -m venv ./venv
source ./venv/bin/activate
```

## Usage

### Indexing

Before editing a video, you must first index the video(s) with the following command.

```bash
python index.py <path_to_video>
```

#### Changing output directory

By default this script will index the current directory. To modify the output directory, you can specify `--out_dir`:

```bash
python index.py <path_to_video> --out_dir=./indexed
```

#### Concurrent Processing

By default this script will process scenes sequentially. To modify the number of concurrent scene processes, you can specify `--max_scene_workers`:

```bash
python index.py <path_to_video> --max_scene_workers=10
```

Keep in mind the indexing code relies on a CUDA, so while the processing of the scenes can be processed concurrently, the default behavior for the transcriptions is sequential.

To modify this you can provide the `--subset_audio_workers` to the script. I recommend monitoring VRAM resources to find the correct number of audio workers.

```bash
python index.py <path_to_video> --max_scene_workers=10 --subset_audio_workers=5
```

### Specifying a GPT Model

The indexing process uses OpenAI GPT models for scene event inference. This process by default uses `gpt-4.1`. This can be modified through the `--event_model` flag using any OpenAI GPT model that [supports image understanding](https://platform.openai.com/docs/models).

```bash
python index.py --event_model=gpt-4.1-nano
```

For more information on usage, you can use the `--help` command.

### Editing

After video(s) is indexed, you can edit the video together with the following command.

```bash
python editor.py "<prompt>"
```

Where the file output will be `out.mp4` to modify this, you can provide the flag `--out=<outfile.mp4>`.

#### Changing Input Directory

If the indexing step included modifying the indexing directory, that directory can be specified with `--index_dir`.

```bash
python editor.py "<prompt>" --index_dir="./indexed"
```

By default this value will be set to the current directory.

For more information on usage, you can use the `--help` command.

## Contributing

We welcome contributions. Feel free to submit a PR or open an issue!

## License

This project is licensed under the MIT License.
