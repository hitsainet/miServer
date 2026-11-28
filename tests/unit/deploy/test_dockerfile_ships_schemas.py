"""The image must carry the vendored schemas that production code reads (DEBT_TASKS 6.2, DEBT-043-b).

Every reader resolves `Path(__file__).parents[k] / "docs" / "schemas"`, outside the package, so a wheel installed
alone carries none of them: the image works only because it runs from source (`PYTHONPATH=/app`, `WORKDIR /app`) with
`docs/schemas/` copied beside `miserver/`. That arrangement is the contract this file holds:

* the runtime stage sets `PYTHONPATH=/app` and `WORKDIR /app`, and copies `miserver/` and `docs/schemas/` there;
* the runtime-read set is DERIVED, never listed: every module-level path a `miserver` module holds into
  `docs/schemas/` (seven files and two directory constants on 2026-11-26, where this file used to name two);
* in a scratch root built from the runtime stage's own `COPY` lines, every one of those paths exists.

Control: drop the `docs/schemas/` COPY, or `PYTHONPATH=/app`, and this goes red.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO / "Dockerfile"
RUNTIME_STAGE = "FROM python:3.13-slim AS runtime"
#: Optional runtime dependencies a module may lack in a test environment (CI installs wasmtime; GGUF is the image's).
OPTIONAL_IMPORTS = frozenset({"wasmtime", "llama_cpp"})

#: Imports every `miserver` module from ROOT and writes `{module.attr: {rel, exists, dir}}` for every module-level path
#: into ROOT/docs/schemas, plus the import failures. Run in a subprocess: the import is the test's subject.
COLLECT = r"""
import importlib, json, pathlib, pkgutil, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import miserver
schemas = (root / "docs" / "schemas").resolve()
found, failed = {}, []
for m in pkgutil.walk_packages(miserver.__path__, "miserver."):
    if ".migrations." in m.name:
        continue
    try:
        mod = importlib.import_module(m.name)
    except ImportError as e:
        failed.append([m.name, getattr(e, "name", None) or ""])
        continue
    for k, v in vars(mod).items():
        if isinstance(v, pathlib.PurePath):
            p = pathlib.Path(v)
            try:
                rel = p.resolve().relative_to(schemas)
            except ValueError:
                continue
            found[f"{m.name}.{k}"] = {"rel": str(rel), "exists": p.exists(), "dir": p.is_dir()}
open(sys.argv[2], "w").write(json.dumps({"found": found, "failed": failed, "file": miserver.__file__}))
"""


def runtime_stage() -> list[str]:
    """The Dockerfile lines of the runtime stage, with continuation lines joined."""
    joined, buf = [], ""
    for line in DOCKERFILE.read_text().splitlines():
        buf += line.rstrip("\\").rstrip() + " " if line.rstrip().endswith("\\") else line
        if not line.rstrip().endswith("\\"):
            joined.append(buf.strip())
            buf = ""
    start = joined.index(RUNTIME_STAGE)
    stage = []
    for line in joined[start + 1:]:
        if line.startswith("FROM "):
            break
        stage.append(line)
    return stage


def _env(stage: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stage:
        if line.startswith("ENV "):
            for part in line[4:].split():
                if "=" in part:
                    key, _, value = part.partition("=")
                    out[key] = value
    return out


def _copies(stage: list[str]) -> list[tuple[str, str]]:
    return [tuple(line.split()[1:3]) for line in stage  # type: ignore[misc]
            if line.startswith("COPY ") and len(line.split()) == 3]


def test_the_runtime_stage_runs_from_source_under_app():
    stage = runtime_stage()
    assert _env(stage).get("PYTHONPATH") == "/app", "the image imports miserver from /app, not from a wheel"
    assert "WORKDIR /app" in stage
    copies = set(_copies(stage))
    assert ("miserver/", "/app/miserver/") in copies
    assert ("docs/schemas/", "/app/docs/schemas/") in copies


def _collect(root: Path, out: Path) -> dict:
    env = {**os.environ, "SECRET_KEY": os.environ.get("SECRET_KEY", "ci-test-secret-key-minimum-32-characters-long")}
    subprocess.run([sys.executable, "-I", "-c", COLLECT, str(root), str(out)], cwd=out.parent, env=env,
                   check=True, capture_output=True, timeout=600)
    doc = json.loads(out.read_text())
    bad = [f"{mod} (missing {missing!r})" for mod, missing in doc["failed"] if missing not in OPTIONAL_IMPORTS]
    assert not bad, f"modules that do not import from {root}: {bad}"
    return doc


def _image_root(tmp: Path) -> Path:
    """A scratch root laid out by the runtime stage's own COPY lines (`/app/<dest>` -> `<root>/<dest>`)."""
    root = tmp / "app"
    root.mkdir()
    for src, dest in _copies(runtime_stage()):
        source = REPO / src
        rel = dest[len("/app/"):] if dest.startswith("/app/") else dest
        target = root / rel
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif source.is_file():
            if dest.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                target = target / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return root


@pytest.fixture(scope="module")
def readers(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("schemas")
    checkout = _collect(REPO, tmp / "checkout.json")
    image = _collect(_image_root(tmp), tmp / "image.json")
    return {"checkout": checkout, "image": image, "image_root": tmp / "app"}


def test_the_walk_finds_the_runtime_readers(readers):
    """Guard the guard: a walk that found nothing would pass every check below."""
    found = readers["checkout"]["found"]
    files = {v["rel"] for v in found.values() if not v["dir"]}
    assert len(files) >= 7, sorted(files)
    assert {"adapter-manifest-v1.json", "checkpoint-manifest-v1.json", "behaviour-verdict-v1.json",
            "behaviour-verdict-cases.json", "midataworks-rule-set-v1.json", "midataworks-runner-v1.json"} <= files


def test_every_runtime_schema_exists_in_the_image_layout(readers):
    image = readers["image"]
    assert Path(image["file"]).resolve().is_relative_to(readers["image_root"].resolve()), (
        f"the image-layout run imported miserver from {image['file']}, not from the scratch root"
    )
    assert {k: v["rel"] for k, v in image["found"].items()} == {
        k: v["rel"] for k, v in readers["checkout"]["found"].items()}
    missing = sorted(f"{k} -> docs/schemas/{v['rel']}" for k, v in image["found"].items() if not v["exists"])
    assert not missing, f"read at runtime but absent from the image: {missing}"
