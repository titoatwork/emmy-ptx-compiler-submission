#!/usr/bin/env python3
"""Local GPU execution harness for fast numeric debugging.

Requires a local CUDA device and ptxas. The driver library, device architecture
and ptxas are detected automatically on Linux and WSL; override with CUDA_LIB,
LOCAL_ARCH and PTXAS.
"""

import sys
sys.dont_write_bytecode = True
import ctypes, ctypes.util, json, math, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

from irfiles import resolve_ir
from ptxarch import retarget_ptx


def _load_cuda():
    candidates = [os.environ.get("CUDA_LIB")]
    candidates += ["/usr/lib/wsl/lib/libcuda.so", "libcuda.so.1",
                   "/usr/lib/x86_64-linux-gnu/libcuda.so.1"]
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    fallback = ctypes.util.find_library("cuda")
    if fallback:
        try:
            return ctypes.CDLL(fallback)
        except OSError:
            pass
    raise OSError("cannot load libcuda; set CUDA_LIB")


cuda = _load_cuda()
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
cuda.cuMemFree_v2.argtypes = [ctypes.c_void_p]
cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
cuda.cuInit(0)
dev = ctypes.c_int()
cuda.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)

NP = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "f16": np.float16}

def check(rc, what):
    if rc != 0:
        s = ctypes.c_char_p()
        cuda.cuGetErrorString(rc, ctypes.byref(s))
        raise RuntimeError(f"{what}: {rc} {s.value}")


def _detect_arch():
    major = ctypes.c_int()
    minor = ctypes.c_int()
    if cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) != 0:
        return "sm_86"
    if cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) != 0:
        return "sm_86"
    return f"sm_{major.value}{minor.value}"


def _find_ptxas():
    explicit = os.environ.get("PTXAS")
    if explicit:
        return explicit
    found = shutil.which("ptxas")
    if found:
        return found
    for path in ("/usr/local/cuda/bin/ptxas", "/usr/local/cuda-12.6/bin/ptxas"):
        if os.path.exists(path):
            return path
    return "ptxas"


LOCAL_ARCH = os.environ.get("LOCAL_ARCH") or _detect_arch()
PTXAS = _find_ptxas()


def compile_ptx(ptx, arch=LOCAL_ARCH):
    ptx = retarget_ptx(ptx, arch)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "k.ptx"); c = os.path.join(d, "k.cubin")
    open(p, "w").write(ptx)
    r = subprocess.run([PTXAS, f"-arch={arch}", p, "-o", c, "-v"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ptxas: " + r.stderr)
    return open(c, "rb").read(), r.stderr

def run_kernel(cubin, entry, grid, block, smem, args):
    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(cubin)), "load")
    fn = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, entry.encode()), "getfn")
    if smem > 48 * 1024:
        check(cuda.cuFuncSetAttribute(fn, 8, ctypes.c_int(smem)), "smemattr")
    devptrs = []
    holders = []
    for a in args:
        if a.get("output"):
            buf = np.zeros(math.prod(a["shape"]), dtype=NP[a["dtype"]])
        elif "data" in a:
            buf = np.array(a["data"], dtype=NP[a["dtype"]]).reshape(-1)
        else:
            rng = np.random.default_rng(a.get("seed", 0))
            buf = rng.standard_normal(math.prod(a["shape"])).astype(NP[a["dtype"]])
        holders.append(buf)
        p = ctypes.c_void_p()
        check(cuda.cuMemAlloc_v2(ctypes.byref(p), max(4, buf.nbytes)), "alloc")
        if buf.nbytes:
            check(cuda.cuMemcpyHtoD_v2(p, ctypes.c_void_p(buf.ctypes.data), buf.nbytes), "htod")
        devptrs.append(p)
    params = (ctypes.c_void_p * len(args))()
    kv = []
    for i, a in enumerate(args):
        v = ctypes.c_uint64(devptrs[i].value)
        kv.append(v)
        params[i] = ctypes.cast(ctypes.byref(kv[i]), ctypes.c_void_p)
    check(cuda.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                              smem, None, params, None), "launch")
    check(cuda.cuCtxSynchronize(), "sync")
    out = {}
    for a, p, buf in zip(args, devptrs, holders):
        if a.get("output"):
            got = np.empty_like(buf)
            check(cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(got.ctypes.data), p, buf.nbytes), "dtoh")
            out[a["name"]] = got.reshape(a["shape"])
        cuda.cuMemFree_v2(p)
    check(cuda.cuModuleUnload(mod), "unload")
    return out

def main():
    pdir = Path(sys.argv[1])
    ptx_path = Path(sys.argv[2])
    meta = json.loads((pdir / "meta.json").read_text())
    expected_path = pdir / "expected.npz"
    if not expected_path.exists():
        print(f"[{pdir}] FAIL: missing expected.npz")
        return 1
    expected = dict(np.load(expected_path))
    if not expected:
        print(f"[{pdir}] FAIL: expected.npz has no arrays")
        return 1
    if not meta.get("kernels"):
        print(f"[{pdir}] FAIL: meta.json has no kernels")
        return 1
    produced = {}
    ok = True
    compared = 0
    for k in meta["kernels"]:
        if ptx_path.suffix == ".py":
            ir = resolve_ir(pdir, k["entry"])
            r = subprocess.run([sys.executable, str(ptx_path), str(ir)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{k['entry']}] COMPILER FAILED: {r.stderr[-800:]}")
                return 1
            ptx = r.stdout
        else:
            ptx = ptx_path.read_text()
        cubin, info = compile_ptx(ptx)
        args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            args.append(a)
        out = run_kernel(cubin, k["entry"], k["grid"], k["block"], k["smem_bytes"], args)
        for name, got in out.items():
            if name in expected:
                dt = expected[name].dtype
                got = got.astype(dt)
            produced[name] = got.reshape(-1).tolist()
            if name not in expected:
                if k.get("final", True):
                    print(f"[{k['entry']}] FAIL {name}: no reference array in expected.npz")
                    ok = False
                continue
            if name in expected:
                ref = expected[name]
                compared += 1
                if got.size != ref.size:
                    print(f"[{k['entry']}] FAIL {name}: {got.size} values, expected {ref.size}")
                    ok = False
                    continue
                # The runner returns flat values and reshapes them to the
                # expected shape; a scalar output arrives as shape (1,).
                got = got.reshape(ref.shape)
                close = np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"])
                err = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)))) if got.size else 0.0
                print(f"[{k['entry']}] {'PASS' if close else 'FAIL'} {name}: max abs err {err:.3e}")
                if not close:
                    ok = False
                    print("   first mismatches:")
                    flat_got = got.reshape(-1); flat_ref = ref.reshape(-1)
                    idx = np.argsort(-np.abs(flat_got.astype(np.float64)-flat_ref.astype(np.float64)))[:8]
                    for i in idx:
                        print(f"    idx {i}: got {flat_got[i]!r} ref {flat_ref[i]!r}")
        if info.strip():
            for line in info.strip().splitlines():
                if "Used" in line:
                    print("   ", line.strip())
    if compared == 0:
        print(f"[{pdir}] FAIL: no output was compared against expected.npz")
        ok = False
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
