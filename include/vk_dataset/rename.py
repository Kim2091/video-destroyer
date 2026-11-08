"""
Renumbering the sequences of a dataset after some were culled, so the show
numbers run 1, 2, 3... again and each sequence's frames run 1, 2, 3....

    python -m vk_dataset rename <dataset folder>

The LR folder says which frames a sequence has; the HR folder follows it, so
a dataset with only the centre HR frame of each sequence stays in step. Both
folders are renamed together, in two passes through temporary names so no
file is overwritten halfway.
"""

import os

from .dataset_folders import IMAGE_EXTENSIONS, SHOW_FRAME_STEM, side_dir


def _sequence_groups(hr_dir, lr_dir):
    """[(show number, LR file names in frame order, HR file names that exist)] by show number."""
    grouped = {}
    for name in os.listdir(lr_dir):
        stem, ext = os.path.splitext(name)
        match = SHOW_FRAME_STEM.match(stem)
        if match and ext.lower() in IMAGE_EXTENSIONS and os.path.isfile(os.path.join(lr_dir, name)):
            grouped.setdefault(int(match.group(1)), []).append((int(match.group(2)), name))
    groups = []
    for show, frames in sorted(grouped.items()):
        lr_names = [name for _, name in sorted(frames)]
        hr_names = [name for name in lr_names if os.path.isfile(os.path.join(hr_dir, name))]
        if hr_names:
            groups.append((show, lr_names, hr_names))
    return groups


def rename_sequences(dataset, emit=None):
    """Renumbers the dataset's sequences in place and returns what it did."""
    if not dataset or not os.path.isdir(dataset):
        raise ValueError(f"the dataset folder {dataset!r} does not exist")
    hr_dir, lr_dir = side_dir(dataset, "hr"), side_dir(dataset, "lr")
    if hr_dir is None or lr_dir is None:
        raise ValueError(f"{dataset} needs both an hr and an lr folder")
    groups = _sequence_groups(hr_dir, lr_dir)
    if not groups:
        raise ValueError("no paired show<N>_Frame<M> image sequences were found")

    moves = []
    sources, targets = set(), set()
    for new_show, (old_show, lr_names, hr_names) in enumerate(groups, start=1):
        new_names = [f"show{new_show}_Frame{n}{os.path.splitext(name)[1]}"
                     for n, name in enumerate(lr_names, start=1)]
        if lr_names == new_names and old_show == new_show:
            continue
        new_name_of = dict(zip(lr_names, new_names))
        for folder, names in ((lr_dir, lr_names), (hr_dir, hr_names)):
            for name in names:
                old, temporary = os.path.join(folder, name), os.path.join(folder, name + ".renumber_tmp")
                new = os.path.join(folder, new_name_of[name])
                if os.path.exists(temporary):
                    raise RuntimeError(f"the temporary name {temporary} is already taken")
                moves.append((old, temporary, new))
                sources.add(old)
                targets.add(new)
    for target in targets:
        if os.path.exists(target) and target not in sources:
            raise RuntimeError(f"cannot renumber over the unrelated file {target}")

    for old, temporary, _ in moves:
        os.replace(old, temporary)
    for _, temporary, new in moves:
        os.replace(temporary, new)
    if emit:
        emit({"message": f"packed {len(groups)} sequences, renamed {len(moves)} files"})
    return {"sequences": len(groups), "renamed_files": len(moves), "hr_dir": hr_dir, "lr_dir": lr_dir}
