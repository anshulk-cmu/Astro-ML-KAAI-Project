"""Convention tests for the input-space operators.

These pin the rotation sign, the array-versus-catalog frame handedness, and the algebraic
properties of the major-axis flip before any expensive re-encoding is run. The operators
live in lib/transforms.py and are introduced with Diagnostics 2, 3, 5 and 9; the module is
skipped until they exist so a missing implementation can never be mistaken for a pass.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import pytest

transforms = pytest.importorskip("transforms")

pytestmark = pytest.mark.skipif(
    not hasattr(transforms, "rotate"),
    reason="input-space operators are introduced with Diagnostic 2; contract in lib/transforms.py",
)


def test_placeholder_until_operators_exist():
    raise AssertionError("implement with Diagnostic 2")
