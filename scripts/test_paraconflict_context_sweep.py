"""Smoke tests for ParaConflict prompt decomposition (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PCD = Path(__file__).resolve().parents[1] / "ic_factual" / "paraconflict_decompose.py"
_spec = importlib.util.spec_from_file_location("paraconflict_decompose", _PCD)
_pcd = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pcd)
decompose_row = _pcd.decompose_row
assemble_prompt = _pcd.assemble_prompt


def test_decompose_athlete_row():
    row = {
        "Subject": "Conor McGregor",
        "Distracted Token": "basketball",
        "Clean Prompt": "Conor McGregor plays the sport of",
        "Substitution Conflict": (
            "Conor McGregor plays the sport of basketball. "
            "Conor McGregor plays the sport of"
        ),
        "Coherent Conflict": (
            "Conor McGregor plays the sport of basketball. "
            "Recognized by peers and fans alike, his journey continues."
        ),
    }
    clause, passage, cont = decompose_row(row)
    assert "basketball" in clause
    assert cont == "Conor McGregor plays the sport of"
    assert "Recognized" in passage
    prompt = assemble_prompt(clause, "FILLER", cont)
    assert "FILLER" in prompt
    assert prompt.endswith(cont)


def test_assemble_no_filler():
    row = {
        "Subject": "France",
        "Distracted Token": "Tokyo",
        "Clean Prompt": "The capital of France is",
        "Substitution Conflict": "The capital of France is Tokyo. The capital of France is",
        "Coherent Conflict": "The capital of France is Tokyo. Tokyo is a vibrant city.",
    }
    clause, _, cont = decompose_row(row)
    built = assemble_prompt(clause, "", cont)
    assert built.replace("  ", " ") == row["Substitution Conflict"].replace("  ", " ")


if __name__ == "__main__":
    test_decompose_athlete_row()
    test_assemble_no_filler()
    print("ok")
