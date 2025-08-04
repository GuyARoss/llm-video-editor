# Final Cut Template

Use this template to write the final edit script. It consists of:

1. Header block
2. Audio block
3. Video block
4. (Optional) Edge-case annotations

This should always be saved as `_final.txt` in the project directory.

## 1. Header

```txt
[Edited]
A short description of tone/purpose (5–10 words)
```

## 2. Audio

```txt
Audio
[SS - EE] clip_name[source_start]
[SS - EE] clip_name[source_start]
```

- All timestamps (`SS`) mark the start time in the final edit.
- All timestamps (`EE`) mark the end time in the final edit.
- `source_start` (also `SS`) indicates the clip’s original start (in seconds).
- The source end is calculated by matching the duration of the final cut.

## 3. Video

```txt
Video
[SS - EE] clip_name[source_start]
[SS - EE] clip_name[source_start]
```

- Same conventions as Audio.
- Clips run until their end time or until the next clip’s start.
