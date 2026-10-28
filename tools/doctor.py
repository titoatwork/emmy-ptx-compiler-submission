#!/usr/bin/env python3
"""Check whether this machine can build, test and benchmark the compiler.

    ../venv/bin/python tools/doctor.py
    ../venv/bin/python tools/doctor.py --runner
    ../venv/bin/python tools/doctor.py --task ../task

Exit code 0 means no hard failures. Warnings describe optional pieces, such as
a local GPU or a generated corpus, that are useful but not required to compile.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import ctypes
import ctypes.util
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from irfiles import resolve_ir  # noqa: E402
RUNNER = os.environ.get("RUNNER_URL", "").rstrip("/")
CORPORA = ("stress", "blocks", "fixed", "ln", "small", "f16", "fuzz",
           "attngen", "attnedge", "attnadv", "ops", "dtypeedge", "adv")
ROWS = []


def add(status, name, detail=""):
    ROWS.append((status, name, detail))


def run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_python():
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 12):
        add("OK", "Python", version)
    else:
        add("FAIL", "Python", f"{version}; this project needs 3.12 or newer")


def check_repo():
    missing = [name for name in ("compile.py", "ptxc", "tests", "tools")
               if not (ROOT / name).exists()]
    if missing:
        add("FAIL", "Repository", f"missing {', '.join(missing)} under {ROOT}")
    else:
        add("OK", "Repository", str(ROOT))


def check_emmy():
    if importlib.util.find_spec("emmy") is None:
        add("FAIL", "emmy", "not importable; install requirements.txt in the compiler venv")
        return
    try:
        version = importlib.metadata.version("emmy-ml")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    add("OK", "emmy", f"emmy-ml {version}")


def check_tools_deps():
    missing = [name for name in ("requests", "numpy") if importlib.util.find_spec(name) is None]
    if missing:
        add("WARN", "tool dependencies", f"missing {', '.join(missing)}; install requirements-tools.txt")
    else:
        add("OK", "tool dependencies", "requests and numpy")
    if importlib.util.find_spec("torch") is None:
        add("INFO", "torch", "not installed; only needed to regenerate corpora through the emmy CLI")
    else:
        add("OK", "torch", "installed")


def check_ptxas():
    explicit = os.environ.get("PTXAS")
    if explicit:
        path = explicit if os.path.sep in explicit else shutil.which(explicit)
        result = run([path, "--version"], timeout=10) if path and Path(path).exists() else None
        if result is None or result.returncode != 0:
            add("FAIL", "ptxas", f"PTXAS={explicit} is not a working ptxas")
            return
        first = (result.stdout or result.stderr).strip().splitlines()[0]
        add("OK", "ptxas", f"{path}: {first}")
        return
    candidates = []
    found = shutil.which("ptxas")
    if found:
        candidates.append(found)
    candidates += ["/usr/local/cuda/bin/ptxas", "/usr/local/cuda-12.6/bin/ptxas",
                   "/usr/local/cuda-12.5/bin/ptxas", "/usr/local/cuda-12.4/bin/ptxas"]
    for path in candidates:
        if not path or not Path(path).exists():
            continue
        result = run([path, "--version"], timeout=10)
        if result is not None and result.returncode == 0:
            first = (result.stdout or result.stderr).strip().splitlines()[0]
            add("OK", "ptxas", f"{path}: {first}")
            return
    detail = "not found; local variants need it. Set PTXAS or install the CUDA toolkit"
    add("WARN", "ptxas", detail)


def load_cuda():
    explicit = os.environ.get("CUDA_LIB")
    candidates = [explicit] if explicit else []
    candidates += ["/usr/lib/wsl/lib/libcuda.so", "libcuda.so.1",
                   "/usr/lib/x86_64-linux-gnu/libcuda.so.1"]
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name), name
        except OSError:
            continue
    fallback = ctypes.util.find_library("cuda")
    if fallback:
        try:
            return ctypes.CDLL(fallback), fallback
        except OSError:
            pass
    return None, explicit


def check_cuda():
    smi = shutil.which("nvidia-smi")
    name = None
    cap = None
    if smi:
        result = run([smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"], timeout=10)
        if result is not None and result.returncode == 0 and result.stdout.strip():
            first = result.stdout.strip().splitlines()[0].split(",", 1)
            if len(first) == 2:
                name, cap = first[0].strip(), first[1].strip()
    if name:
        add("OK", "GPU", f"{name}, compute {cap}")
    else:
        add("WARN", "GPU", "nvidia-smi did not report a device; local variants will be skipped")

    lib, source = load_cuda()
    if lib is None:
        add("WARN", "libcuda", "not loadable; set CUDA_LIB for localrun.py")
        return
    try:
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        rc = lib.cuInit(0)
        if rc != 0:
            add("WARN", "libcuda", f"loaded from {source} but cuInit returned {rc}")
            return
        dev = ctypes.c_int()
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype = ctypes.c_int
        rc = lib.cuDeviceGet(ctypes.byref(dev), 0)
        if rc != 0:
            add("WARN", "libcuda", f"loaded from {source} but cuDeviceGet returned {rc}")
            return
        buf = ctypes.create_string_buffer(256)
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.cuDeviceGetName.restype = ctypes.c_int
        lib.cuDeviceGetName(buf, 256, dev)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        lib.cuDeviceGetAttribute.restype = ctypes.c_int
        lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
        lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)
        arch = f"sm_{major.value}{minor.value}"
        add("OK", "libcuda", f"{source}; device {buf.value.decode(errors='replace')}; LOCAL_ARCH={arch}")
    except (AttributeError, OSError) as exc:
        add("WARN", "libcuda", f"loaded from {source} but could not query it: {exc}")


def check_task(task):
    if not (task / "submit.py").exists():
        level = "FAIL" if os.environ.get("TASK") else "WARN"
        add(level, "task repository", f"{task} has no submit.py; clone emmy-hackathon-2026 there")
        return
    problems = sorted(p for p in (task / "problems").glob("*/meta.json"))
    add("OK", "task repository", f"{task} ({len(problems)} problems)")


def check_compile(task):
    if not (task / "submit.py").exists():
        add("SKIP", "compile smoke test", "no task repository")
        return
    problem = next((p for p in sorted((task / "problems").iterdir()) if (p / "meta.json").exists()), None)
    if problem is None:
        add("SKIP", "compile smoke test", "no problem packages")
        return
    meta = json.loads((problem / "meta.json").read_text())
    if not meta.get("kernels"):
        add("SKIP", "compile smoke test", f"{problem.name} has no kernels")
        return
    entry = meta["kernels"][0]["entry"]
    try:
        ir = resolve_ir(problem, entry)
    except Exception:
        add("SKIP", "compile smoke test", f"{problem.name} has no IR for {entry}")
        return
    result = run([sys.executable, str(ROOT / "compile.py"), str(ir)], timeout=60)
    if result is None:
        add("FAIL", "compile smoke test", "could not run compile.py")
    elif result.returncode != 0:
        add("FAIL", "compile smoke test", (result.stderr or "").strip()[-300:])
    elif f".visible .entry {entry}(" not in result.stdout:
        add("FAIL", "compile smoke test", f"PTX does not declare entry {entry}")
    else:
        add("OK", "compile smoke test", f"{problem.name} -> {entry}")


def check_assets():
    variants = len(list((ROOT / "tests" / "variants").glob("*/meta.json")))
    add("OK" if variants else "WARN", "variants", f"{variants} packages in tests/variants")

    found = []
    missing = []
    for name in CORPORA:
        path = ROOT.parent / name
        count = len(list(path.glob("*/meta.json"))) if path.exists() else 0
        if count:
            found.append(f"{name}={count}")
        else:
            missing.append(name)
    if found:
        add("OK", "corpora", ", ".join(found))
    else:
        add("WARN", "corpora", "none found; run tools/generate_corpus.sh")
    if missing:
        add("INFO", "corpora missing", ", ".join(missing))


def check_dashes():
    result = run([sys.executable, str(ROOT / "tools" / "check_dashes.py")], timeout=60)
    if result is not None and result.returncode == 0:
        add("OK", "dash check", result.stdout.strip())
    else:
        detail = (result.stdout or "").strip()[-500:] if result else "could not run"
        add("FAIL", "dash check", detail)


def check_runner():
    if not RUNNER:
        add("SKIP", "runner", "set RUNNER_URL to the event-provided runner address")
        return
    request = urllib.request.Request(RUNNER, headers={"User-Agent": "ptxc-doctor"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            add("OK", "runner", f"{RUNNER} (HTTP {response.status})")
    except urllib.error.HTTPError as exc:
        add("OK", "runner", f"{RUNNER} (HTTP {exc.code}; reachable)")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        add("FAIL", "runner", f"{RUNNER}: {exc}")


def print_report():
    width = max(len(name) for _, name, _ in ROWS)
    for status, name, detail in ROWS:
        print(f"{status:4s} {name:{width}s}  {detail}")
    counts = {key: sum(1 for row in ROWS if row[0] == key) for key in ("OK", "WARN", "FAIL", "INFO", "SKIP")}
    print()
    print("summary: " + ", ".join(f"{key} {counts[key]}" for key in ("OK", "WARN", "FAIL", "INFO", "SKIP")))
    problems = [(status, name, detail) for status, name, detail in ROWS if status in ("WARN", "FAIL")]
    if problems:
        print("needs attention:")
        for status, name, detail in problems:
            print(f"  {status} {name}: {detail}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default=os.environ.get("TASK", str(ROOT.parent / "task")))
    parser.add_argument("--runner", action="store_true", help="also check the hosted runner")
    args = parser.parse_args(argv)

    check_python()
    check_repo()
    check_emmy()
    check_tools_deps()
    check_ptxas()
    check_cuda()
    check_task(Path(args.task))
    check_compile(Path(args.task))
    check_assets()
    check_dashes()
    if args.runner:
        check_runner()
    elif RUNNER:
        add("INFO", "runner", f"{RUNNER}; not checked (use --runner)")
    else:
        add("INFO", "runner", "set RUNNER_URL to the event runner address")
    print_report()
    return 1 if any(row[0] == "FAIL" for row in ROWS) else 0


if __name__ == "__main__":
    sys.exit(main())
