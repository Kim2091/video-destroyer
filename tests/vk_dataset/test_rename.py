"""Tests for renumbering dataset sequences after culling, mirroring VSR-Dataset-Creator's."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

from vk_dataset.__main__ import main  # noqa: E402
from vk_dataset.rename import rename_sequences  # noqa: E402


def touch(root, side, name):
    path = root / side / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def names(root, side):
    return sorted(p.name for p in (root / side).iterdir())


def test_remaining_sequences_become_contiguous(tmp_path):
    for name in ("show1_Frame1.png", "show1_Frame2.png", "show3_Frame1.png", "show3_Frame2.png"):
        touch(tmp_path, "HR", name)
        touch(tmp_path, "LR", name)
    result = rename_sequences(str(tmp_path))
    assert (result["sequences"], result["renamed_files"]) == (2, 4)
    expected = ["show1_Frame1.png", "show1_Frame2.png", "show2_Frame1.png", "show2_Frame2.png"]
    assert names(tmp_path, "HR") == names(tmp_path, "LR") == expected


def test_frames_are_renumbered_in_order(tmp_path):
    for name in ("show5_Frame2.png", "show5_Frame4.png"):
        touch(tmp_path, "hr", name)
        touch(tmp_path, "lr", name)
    rename_sequences(str(tmp_path))
    assert names(tmp_path, "hr") == ["show1_Frame1.png", "show1_Frame2.png"]


def test_centre_only_hr_follows_the_full_lr_sequence(tmp_path):
    for frame in range(1, 8):
        touch(tmp_path, "lr", f"show4_Frame{frame}.png")
    touch(tmp_path, "hr", "show4_Frame4.png")
    assert rename_sequences(str(tmp_path))["sequences"] == 1
    assert names(tmp_path, "lr") == [f"show1_Frame{n}.png" for n in range(1, 8)]
    assert names(tmp_path, "hr") == ["show1_Frame4.png"]


def test_swapped_numbers_do_not_overwrite(tmp_path):
    # show2 and show4 pack to show1 and show2, which is a name show2 already holds.
    for show in (2, 4):
        for side in ("hr", "lr"):
            touch(tmp_path, side, f"show{show}_Frame1.png")
    (tmp_path / "hr" / "show4_Frame1.png").write_bytes(b"four")
    rename_sequences(str(tmp_path))
    assert (tmp_path / "hr" / "show2_Frame1.png").read_bytes() == b"four"


def test_nothing_to_rename_is_an_error(tmp_path):
    touch(tmp_path, "hr", "a.png")
    touch(tmp_path, "lr", "a.png")
    with pytest.raises(ValueError, match="no paired"):
        rename_sequences(str(tmp_path))


def test_cli_subcommand(tmp_path, capsys):
    touch(tmp_path, "hr", "show7_Frame1.png")
    touch(tmp_path, "lr", "show7_Frame1.png")
    assert main(["rename", str(tmp_path)]) == 0
    assert '"sequences": 1' in capsys.readouterr().out
    assert names(tmp_path, "hr") == ["show1_Frame1.png"]
