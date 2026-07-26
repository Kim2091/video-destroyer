# Video Destroyer

Video Destroyer builds validated paired HR/LR frame datasets for training. It has two canonical workflows and requires FFmpeg on `PATH`.

## Install

```bash
pip install .
```

`python -m video_destroyer` is equivalent to the installed `video-destroyer` command.

## Desktop App

Install the optional desktop interface, then launch it:

```bash
pip install ".[gui]"
video-destroyer-gui
```

The GUI provides folder/file pickers and guided create/import forms. The Create screen includes a drag-sortable degradation pipeline: toggle a visual stage, set its probability, and drag it into the order it should run. Codec encoding remains locked as the final required step. It runs the same canonical commands as the terminal and shows their live output, so every GUI run has the same validated run directory, logs, reports, and resumable state. Configuration files remain optional and can be selected as a base for advanced settings.

## Create A Dataset

Generate degraded LR clips from a source video or source directory, then extract and validate frames:

```bash
video-destroyer create D:/source-videos --output D:/datasets/generated
```

## Import Matched Clips

Import an existing recursive HR/LR clip collection. Files are referenced by default and are never changed:

```bash
video-destroyer import-pairs --hr D:/clips/hr --lr D:/clips/lr --output D:/datasets/imported
```

Pairing uses the normalized relative path without its final extension, so `show/scene.mkv` pairs with `show/scene.mp4`. Unmatched paths, duplicate keys, and case-folded collisions fail before extraction. Use `--materialize copy` or `--materialize hardlink` to make run-owned clips.

## Run Layout

Each command creates a run directory containing immutable `run.yaml`, resumable `state.json`, `pairs.jsonl`, `sequences.jsonl`, reports, logs, and application-owned `.work` data. Training files are only published after validation under:

```text
RUN/dataset/hr/
RUN/dataset/lr/
```

Use `video-destroyer resume RUN` for interrupted work, `video-destroyer validate RUN` to recheck output, and `video-destroyer report RUN` to rewrite the summary. A run is `completed`, `completed_with_rejections`, `failed`, or `interrupted`. Rejections are documented in manifests and do not fail a valid run unless `--fail-on-rejection` is supplied.

## Configuration

Both starting commands work with built-in defaults. Optional processing configuration is versioned YAML and must begin with `version: 2`; paths belong on the command line. The configuration can customize `create`, `extract`, `curate`, `validation`, and `runtime` settings. `import-pairs` does not require a `create` section.

The legacy v1 `config.yaml`, `main.py`, `frame_extractor.py`, and `post_process.py` remain temporarily available as deprecated wrappers. Replace `use_existing_chunks: true` with `video-destroyer import-pairs --hr ... --lr ... --output ...`.
