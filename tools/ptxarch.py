"""PTX target selection for the local GPU harness.

The submission PTX declares ``.target sm_89``. The hosted runner assembles that
file with ptxas for ``sm_120`` unchanged. Older local GPUs (``sm_80``, ``sm_86``)
reject a target newer than the device, so only that case rewrites the target.
"""

from __future__ import annotations


def arch_number(arch: str) -> int:
    digits = "".join(ch for ch in arch.rsplit("_", 1)[-1] if ch.isdigit())
    return int(digits) if digits else 0


def retarget_ptx(ptx: str, arch: str) -> str:
    if arch_number(arch) < 89:
        return ptx.replace(".target sm_89", f".target {arch}")
    return ptx
