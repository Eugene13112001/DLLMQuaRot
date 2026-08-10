"""Build an offline wheel bundle so a machine without PyPI can get a
different transformers than the one its venv already has.

Why this exists
---------------
The two LLaDA families need transformers versions whose windows do not
overlap (1.5 wants 4.38-4.46, 2.0 wants 4.56+), the GPU box has no route to
PyPI, and a second venv would have to re-download torch -- 2.5 GB that is
already sitting in the first venv.

So instead of a venv, install the new transformers into a *directory* and put
that directory on ``PYTHONPATH``.  Path entries are searched before
site-packages, so the newer library shadows the older one for exactly the
processes that ask for it, and torch/numpy keep coming from the venv:

    pip install --no-index --no-deps --target=$TF457 bundle/core/*.whl
    PYTHONPATH=$TF457 python scripts/selfcheck.py --model-type llada2_moe ...

Two tiers, because shadowing is not free
----------------------------------------
Everything on ``PYTHONPATH`` shadows for the whole process, including
libraries that only came along as dependencies.  A newer ``fsspec`` can upset
``datasets``, which the eval path needs.  So ``core/`` holds the four wheels
that actually have to change, and ``extra/`` holds the rest of the dependency
closure -- installed only if the core set turns out to need a newer one.

``numpy`` is deliberately excluded from both: it is a compiled extension that
torch is linked against, and swapping it under a working torch is the one
substitution here that can break something unrelated.

Run this on a machine that *does* have PyPI, then copy the zip over.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Wheels whose version is the point of the exercise.  Everything else in the
# closure is a dependency that the target venv very likely already satisfies.
CORE = ("transformers", "tokenizers", "huggingface_hub", "safetensors")

# Never ship these: compiled, and something already in the target links
# against them.
EXCLUDE = ("numpy",)


def wheel_name(path: Path) -> str:
    """Distribution name of a wheel file, normalised."""
    return path.name.split("-")[0].lower().replace("-", "_")


def download(spec: str, dest: Path, python_version: str, platform: str) -> None:
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:",
        f"--platform={platform}",
        f"--python-version={python_version}",
        "--implementation=cp",
        "-d", str(dest),
        spec,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="transformers==4.57.1",
                    help="requirement to resolve, e.g. 'transformers==4.57.1'")
    ap.add_argument("--python-version", default="3.11",
                    help="python of the *target* machine, not this one")
    ap.add_argument("--platform", default="manylinux2014_x86_64")
    ap.add_argument("--out", type=Path, required=True,
                    help="directory to build in; a .zip is written beside it")
    args = ap.parse_args()

    staging = args.out / "_download"
    if args.out.exists():
        shutil.rmtree(args.out)
    staging.mkdir(parents=True)

    download(args.spec, staging, args.python_version, args.platform)

    core_dir = args.out / "core"
    extra_dir = args.out / "extra"
    core_dir.mkdir()
    extra_dir.mkdir()

    dropped = []
    for whl in sorted(staging.glob("*.whl")):
        name = wheel_name(whl)
        if name in EXCLUDE:
            dropped.append(whl.name)
            continue
        target = core_dir if name in CORE else extra_dir
        shutil.move(str(whl), target / whl.name)
    shutil.rmtree(staging)

    (args.out / "INSTALL.txt").write_text(INSTALL_TEXT, encoding="utf-8")

    zip_path = args.out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(args.out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(args.out))

    def show(d: Path) -> None:
        total = sum(f.stat().st_size for f in d.glob("*.whl"))
        print(f"\n{d.name}/  ({total / 1e6:.1f} MB)")
        for f in sorted(d.glob("*.whl")):
            print(f"  {f.name}")

    show(core_dir)
    show(extra_dir)
    if dropped:
        print(f"\nexcluded (compiled, torch links against them): {', '.join(dropped)}")
    print(f"\nbundle: {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return 0


INSTALL_TEXT = """\
Offline install of a shadowing transformers
===========================================

On the target machine, with the existing venv active:

    export TF=$HOME/tf457
    pip install --no-index --no-deps --target=$TF core/*.whl

--no-deps is not laziness: it pins the installed set to exactly these wheels,
so pip cannot decide it also wants a different numpy.

Check it before running anything long:

    PYTHONPATH=$TF python -c "import transformers, torch; \\
        print(transformers.__version__, transformers.__file__); \\
        print(torch.__version__, torch.cuda.device_count())"

Expect the new version, a path under $TF, and torch still from the venv.

If that import fails on a dependency being too old, add the one it names:

    pip install --no-index --no-deps --target=$TF extra/<that>.whl

Add them one at a time.  Everything on PYTHONPATH shadows for the whole
process, and `datasets` in the eval path is sensitive to fsspec's version.

Then run with the path in front, per command:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TF python scripts/selfcheck.py \\
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe --rotate

Leave PYTHONPATH out for LLaDA-1.5 work: that model needs the old
transformers, which is still exactly where it was.
"""


if __name__ == "__main__":
    raise SystemExit(main())
