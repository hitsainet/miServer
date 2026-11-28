"""Deny-by-default brand guard (040 FR-040.15, FR-040.38; FTDD §6.7; TID §3.7).

Scans the brand surfaces (PD-4: code strings, UI, manual, front-door docs, schemas' values, deploy
files, integrations, file names) for the old names `millm` / `mistudio` in any case, and for every
match of an ACTIVE `deny` guard record of the export. A match is allowed only by a record of
`docs/legacy-identifiers.md` (a frozen/scheduled entry's pattern, a transition env/host/repo old
name, a history tool, an active allow guard); the guard keeps no list of its own.

Report-only until P4 closes: `ENFORCED` follows the export header's `closed` (CF-21), checked by
`test_enforced_follows_closed`. In report mode every finding is written to
`tests/.artifacts/brand_report.txt` (ID-7) and the test passes; enforced, a READ-text finding fails
and identifier findings are still only reported (Q4, R17). Enforcement happens in the PRIVATE
repository only (R23, 2026-11-11): on the public mirror, whose export variant withholds the old
hostnames and the private repository names, the guard runs in report mode (`judged_here`). The export itself is excluded with the
glob-correct pathspec (CF-20). Comments are not scanned (PD-4).

P9 (miServer 042 FR-042.2, decisions P9-D1 to P9-D7, 2026-11-14) adds four surfaces that P4 left out,
each tied to the residue group that cleans it: `comments` (RG-1: `#`, `//`, `/* */` comments and
docstrings), `agent_text` (RG-2: `CLAUDE.md` minus its dated sections, the admin UI README), `tests`
(RG-3/RG-6: test names, text, values and file names) and `names` (RG-5: the package's code names).
They read old names with `STEM`, which also sees a camel-case compound (`MiLLMError`) the P4 `TOKEN`
does not. A finding on a P9 surface, and an IDENTIFIER finding on a P4 surface, fails only once its
group is in `CLOSED_GROUPS`; a group is added in the commit that cleans it. A line that keeps an old
name on purpose (a sentence about the rename, a retired name a test must name, a value the register
has not covered yet) says so with `p9: keep` (or P8's `p8: keep`) and a reason, on that line;
`test_keep_marks_carry_a_reason` reads every one.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.support import legacy_identifiers as li
from tests.support import source_scan
from tests.support.package_root import package_rel

ROOT = Path(__file__).resolve().parents[2]
#: The package directory, read from pyproject.toml (041 FR-041.4): the python surface and the
#: planted trees below follow it, so this guard reads the package before and after P8.
PKG = package_rel()
REPORT = ROOT / "tests" / ".artifacts" / "brand_report.txt"
EXPORT_EXCLUDE = ":(exclude,glob)**/docs/legacy-identifiers*.md"

#: Flipped by P4-E (FR-040.38), with the export whose header says `closed: "P4"`;
#: `test_enforced_follows_closed` requires it to equal `closed_at_least("P4")` of the committed export.
ENFORCED = True

TOKEN = re.compile(r"(?i)(?<![a-z0-9])mi(llm|studio)(?![a-z0-9])")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
#: P9 surfaces: TOKEN, plus the start of a camel-case word (`FakeMillm`) and a following capital
#: (`MiLLMError`), which TOKEN's case-blind boundaries miss.
STEM = re.compile(r"(?:(?<![A-Za-z0-9])|(?<=[a-z])(?=M))(?:mi|Mi|MI|mI)(?:llm|LLM|Llm|LLm|studio|Studio|STUDIO)"
                  r"s?(?![a-z0-9])")
#: The guard's own file plants and names old names by design, as the export does; the P9 surfaces
#: skip it (the P4 surfaces never read tests).
GUARD_FILE = "tests/unit/test_brand_names.py"
#: A deliberate keep on a P9 surface: the mark and a reason after it (`p8: keep` is P8's mark).
KEEP = re.compile(r"\bp[89]: keep\b")

#: P9 (042 FR-042.2): the residue groups whose surfaces fail. A group joins in the commit that
#: cleans it; `test_closed_groups_are_enforced` proves each one bites.
CLOSED_GROUPS: frozenset[str] = frozenset({"RG-1", "RG-2", "RG-3", "RG-5", "RG-7"})
#: Which residue group judges each surface. A P4 surface's READ findings fail whatever this says
#: (FR-040.15); its IDENTIFIER findings, and every finding of a P9 surface, fail once the group closes.
GROUP_OF_SURFACE = {"comments": "RG-1", "agent_text": "RG-2", "test_names": "RG-3", "test_values": "RG-6",
                    "names": "RG-5",
                    "python": "RG-5", "typescript": "RG-5", "deploy": "RG-7"}
P9_SURFACES = frozenset({"comments", "agent_text", "test_names", "test_values", "names"})
#: P9-D6 (2026-11-14): a test value changes only with the code value it asserts, and a value that pins a
#: frozen identifier stays, so RG-6 is never closed: its surface reports what is left (mostly frozen).
NEVER_CLOSED = frozenset({"RG-6"})
#: `CLAUDE.md` sections that are dated history and stay as written (P9-D2, P9-D9): `_agent_text`
#: refuses a `CLAUDE.md` that lacks one of them, so a reorganised file fails loudly.
KEPT_SECTIONS = ("### Standing lessons", "### Status history", "## Document Inventory",
                 "## Session History Log")


@dataclass(frozen=True)
class Surface:
    name: str
    globs: tuple[str, ...]
    extract: Callable[[str, str], Iterator[tuple[int, str]]]
    exclude: tuple[str, ...] = ()


def _python(rel: str, source: str) -> Iterator[tuple[int, str]]:
    return source_scan.python_strings(source, docstrings=rel.startswith(f"{PKG}/api/"))


def _typescript(rel: str, source: str) -> Iterator[tuple[int, str]]:
    if rel.endswith(".html"):
        return source_scan.markdown_items(source, prose=True)
    return source_scan.typescript_items(source)


def _by_kind(rel: str, source: str) -> Iterator[tuple[int, str]]:
    if rel.endswith(".toml"):
        return source_scan.data_items(tomllib.loads(source), prose=True)
    if rel.endswith(".py"):
        return _python(rel, source)
    if rel.endswith((".ts", ".tsx", ".js", ".html")):
        return _typescript(rel, source)
    # prose=True: a `description` or `title` value is text people read (a dashboard, a schema, a manual
    # page's front matter). The shared scanner skips those values for the generality guard.
    return source_scan.items_of(rel, source, prose=True)


def _file_name(rel: str, _source: str) -> Iterator[tuple[int, str]]:
    return source_scan.filename_items(rel)


def _comments(rel: str, source: str) -> Iterator[tuple[int, str]]:
    """RG-1: comments and docstrings. A docstring under the package's `api/` is the `python`
    surface's (it feeds the OpenAPI text), so it is read there and not twice."""
    if rel.endswith(".py"):
        return source_scan.python_comments(source, docstrings=not rel.startswith(f"{PKG}/api/"))
    if rel.endswith((".ts", ".tsx", ".js", ".mjs", ".css")):
        return source_scan.typescript_comments(source)
    return source_scan.hash_comments(source)


def _agent_text(rel: str, source: str) -> Iterator[tuple[int, str]]:
    """RG-2: agent instructions and READMEs as read text; `CLAUDE.md` without its dated sections."""
    if rel != "CLAUDE.md":
        return source_scan.markdown_items(source, prose=True)
    return _claude_md_current(source)


def _claude_md_current(source: str) -> Iterator[tuple[int, str]]:
    lines = source.splitlines()
    headings = {ln.strip() for ln in lines if ln.startswith("#")}
    missing = [k for k in KEPT_SECTIONS if not any(h.startswith(k) for h in headings)]
    if missing:
        raise AssertionError(f"CLAUDE.md lost the kept section(s) {missing}: the agent_text surface would "
                             "read dated history as current text (042 FR-042.12)")
    skip_level, fenced = 0, False
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        heading = not fenced and re.match(r"(#{1,6}) ", line)
        if heading:
            level = len(heading.group(1))
            if skip_level and level <= skip_level:
                skip_level = 0
            if any(line.strip().startswith(k) for k in KEPT_SECTIONS):
                skip_level = level
        if not skip_level and line.strip():
            yield number, line


def _test_names(rel: str, source: str) -> Iterator[tuple[int, str]]:
    """RG-3: a test file's name and its code names (functions, classes, constants, variables)."""
    yield from source_scan.filename_items(rel)
    if rel.endswith(".py"):
        yield from source_scan.python_names(source)
    elif rel.endswith((".ts", ".tsx")):
        yield from source_scan.typescript_code(source)


def _test_values(rel: str, source: str) -> Iterator[tuple[int, str]]:
    """RG-6 (and RG-3's message text): the strings and data of tests (comments are `comments`')."""
    if rel.endswith(".py"):
        yield from source_scan.python_strings(source)
    elif rel.endswith((".ts", ".tsx")):
        yield from source_scan.typescript_strings(source)
    elif rel.endswith((".json", ".yaml", ".yml", ".md")):
        yield from source_scan.items_of(rel, source, prose=True)
    else:
        yield from ((n, ln) for n, ln in enumerate(source.splitlines(), 1) if ln.strip())


def _names(rel: str, source: str) -> Iterator[tuple[int, str]]:
    """RG-5: the code names of the package (strings are the `python` surface's)."""
    return source_scan.python_names(source)


TEST_GLOBS = ("tests/**/*.py", "tests/**/*.json", "tests/**/*.yaml", "tests/**/*.yml", "tests/**/*.txt",
              "admin-ui/src/**/__tests__/**/*.ts", "admin-ui/src/**/__tests__/**/*.tsx",
              "admin-ui/src/**/*.test.ts", "admin-ui/src/**/*.test.tsx", "admin-ui/src/test/**")
TEST_EXCLUDE = ("tests/fixtures/goldens/*", "tests/.artifacts/*", GUARD_FILE)

SURFACES = (
    Surface("python", (f"{PKG}/**/*.py",), _python, exclude=(f"{PKG}/db/migrations/*",)),
    Surface("typescript", ("admin-ui/src/**/*.ts", "admin-ui/src/**/*.tsx", "admin-ui/index.html"),
            _typescript, exclude=("*__tests__*", "*.test.ts", "*.test.tsx", "*/test/*",
                                  "*setupTests.ts", "*.d.ts")),
    Surface("ui-files", ("admin-ui/public/**", "manual/static/img/**"), _file_name),
    Surface("manual", ("manual/docusaurus.config.ts", "manual/sidebars.*", "manual/src/**/*.ts",
                       "manual/src/**/*.tsx", "manual/src/**/*.js", "manual/docs/**/*.md",
                       "manual/docs/**/*.mdx"), _by_kind),
    Surface("docs", ("README.md", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md", "docs/*.md",
                     "docs/schemas/*.json"), _by_kind),
    Surface("deploy", ("k8s/**/*.yaml", "k8s/**/*.yml", "k8s/**/*.json", "docker-compose*.yml",
                       ".env.example", ".github/workflows/*.yml",
                       # P9 (042 FR-042.17, RG-7): the package metadata people and tools read.
                       "pyproject.toml", "manual/package.json", "admin-ui/package.json"), _by_kind),
    Surface("integrations", ("integrations/openwebui/*.py",), _python),
    # ── P9 (042 FR-042.2) ──
    Surface("comments", (f"{PKG}/**/*.py", "tests/**/*.py", "integrations/**/*.py", "docker/**/*.py",
                         "scripts/**/*.py", "admin-ui/src/**/*.ts", "admin-ui/src/**/*.tsx",
                         "admin-ui/scripts/**/*.mjs", "admin-ui/*.js", "admin-ui/*.ts", "admin-ui/src/**/*.css",
                         "manual/src/**/*.css", "manual/src/**/*.ts", "manual/src/**/*.tsx", "manual/src/**/*.js",
                         "manual/*.ts", "manual/*.js", "k8s/**/*.yaml", "k8s/**/*.yml", "k8s/**/*.sh",
                         "tests/**/*.yaml", "tests/**/*.yml", ".github/**/*.yml", "docker-compose*.yml",
                         "Dockerfile", "admin-ui/Dockerfile", "docker-entrypoint.sh", "nginx/*.conf",
                         "admin-ui/nginx.conf", "pyproject.toml", "alembic.ini", ".env.example",
                         "constraints.txt"),
            _comments, exclude=(f"{PKG}/db/migrations/*", "tests/.artifacts/*",
                                GUARD_FILE)),
    Surface("agent_text", ("CLAUDE.md", "admin-ui/README.md", ".claude/**/*.md"), _agent_text),
    Surface("test_names", TEST_GLOBS, _test_names, exclude=TEST_EXCLUDE),
    Surface("test_values", TEST_GLOBS, _test_values, exclude=TEST_EXCLUDE),
    Surface("names", (f"{PKG}/**/*.py", "integrations/**/*.py"), _names,
            exclude=(f"{PKG}/db/migrations/*",)),
)


def _listed(surface: Surface, root: Path) -> list[str] | None:
    try:
        out = subprocess.run(["git", "ls-files", "--", *(f":(glob){g}" for g in surface.globs),
                              EXPORT_EXCLUDE], cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return sorted(rel for rel in out.stdout.splitlines()
                  if (root / rel).is_file() and not _excluded(surface, rel))


def _excluded(surface: Surface, rel: str) -> bool:
    return (any(fnmatch.fnmatch(rel, pat) for pat in surface.exclude) or "node_modules" in rel
            or re.search(r"(^|/)docs/legacy-identifiers[^/]*\.md$", rel) is not None)


def _walked(surface: Surface, root: Path) -> list[str]:
    found = set()
    for glob in surface.globs:
        for path in root.glob(glob):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and not _excluded(surface, rel):
                found.add(rel)
    return sorted(found)


@dataclass(frozen=True)
class Finding:
    tree: str
    path: str
    line: int
    token: str
    kind: str  # "read" | "identifier"
    text: str
    kept: bool = False  # the source line carries a `p9: keep` / `p8: keep` mark
    marks: tuple[int, ...] = ()  # the marked line(s) that keep it (a parsed value may stand on several)

    def render(self) -> str:
        kept = " (kept)" if self.kept else ""
        return (f"{self.kind:10s} {self.tree:12s} {self.path}:{self.line}: {self.token!r} in "
                f"{self.text[:100]!r}{kept}")


def classify(text: str, start: int, end: int) -> str:
    """Identifier when the whole literal is identifier-shaped and the token touches `_` or `.`."""
    touches = (start > 0 and text[start - 1] in "_.") or (end < len(text) and text[end] in "_.")
    return "identifier" if IDENTIFIER.fullmatch(text) and touches else "read"


def scan_text(export: li.Export, text: str, stem: bool = False) -> list[tuple[int, int, str]]:
    """`(start, end, token)` of every disallowed match in one extracted text (`stem`: the P9
    surfaces' wider `STEM` as well as `TOKEN`)."""
    spans = [(m.start(), m.end()) for m in TOKEN.finditer(text)]
    if stem:
        spans += [(m.start(), m.end()) for m in STEM.finditer(text)]
    for guard in export.active_guards():
        if guard["effect"] == "deny":
            spans += [(m.start(), m.end()) for m in re.finditer(guard["pattern"], text)]
    out = []
    for start, end in sorted(set(spans)):
        if not export.allowed(text, start, end):
            out.append((start, end, text[start:end]))
    return out


def _segment(text: str, start: int, end: int) -> tuple[str, int]:
    """The token's own line of the extracted text, stripped, and the token's offset in it."""
    seg_start = text.rfind("\n", 0, start) + 1
    seg_end = text.find("\n", end)
    raw = text[seg_start:len(text) if seg_end == -1 else seg_end]
    lead = len(raw) - len(raw.lstrip())
    return raw.strip(), start - seg_start - lead


def _source_lines_of(lines: list[str], line: int, text: str, start: int, end: int) -> list[int]:
    """The 1-based source line(s) the token sits on. An extractor that reports a line gives it (moved
    down by the newlines before the token inside a multi-line literal). A parsed YAML/JSON key or value
    has no line: every source line holding the token's own text line, plain or JSON-escaped."""
    if line >= 1:
        return [line + text.count("\n", 0, start)]
    segment, _offset = _segment(text, start, end)
    whole = _whole(segment, json.dumps(segment)[1:-1])
    return [n for n, source_line in enumerate(lines, 1)
            if not source_line.lstrip().startswith("#") and whole.search(source_line)]  # comments: PD-4


def _whole(*needles: str) -> re.Pattern[str]:
    """A parsed value as it stands in its file: the text, not a piece of a longer name."""
    return re.compile(r"(?<![\w.-])(?:" + "|".join(re.escape(n) for n in needles if n) + r")(?![\w.-])")


def _token_spans(source_line: str, line: int, text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Where on `source_line` the extracted token can be, narrowest reading first: the whole literal
    between quotes (so the `millm` of `"millm": _millm.dump()` is the key, not the variable); else the
    token's own text line wherever the source line holds it; else EVERY occurrence of the token."""
    token, (segment, offset) = text[start:end], _segment(text, start, end)
    hits: list[int] = []
    if line >= 1 and "\n" not in text:
        for quote in "\"'`":
            hits += [m.start() + 1 + start for m in re.finditer(re.escape(quote + text + quote), source_line)]
    if not hits and segment:
        found = _whole(segment) if line < 1 else re.compile(re.escape(segment))
        hits = [m.start() + offset for m in found.finditer(source_line)]
    if not hits:
        hits = [m.start() for m in re.finditer(re.escape(token), source_line)]
    return [(hit, hit + len(token)) for hit in hits]


def source_line_allows(export: li.Export, lines: list[str], line: int, text: str, start: int,
                        end: int) -> bool:
    """FTDD §6.7: the token is also allowed by a record that matches its SOURCE line (the extractor
    strips the quotes, key or call a pattern anchors on, and the register's patterns are written for
    source lines). Fails closed: the line must be found, and on EVERY candidate line every place the
    token can be (`_token_spans`) must lie inside a record's match."""
    numbers = _source_lines_of(lines, line, text, start, end)
    if not numbers or any(number > len(lines) for number in numbers):
        return False
    for number in numbers:
        source_line = lines[number - 1]
        spans = _token_spans(source_line, line, text, start, end)
        if not spans or not all(export.allowed(source_line, s, e) for s, e in spans):
            return False
    return True


def scan_surface(surface: Surface, export: li.Export, root: Path = ROOT) -> tuple[int, list[Finding]]:
    files = _walked(surface, root)
    findings = []
    for rel in files:
        source = "" if surface.extract is _file_name else (root / rel).read_text(encoding="utf-8")
        lines = source.splitlines()
        for line, text in surface.extract(rel, source):
            for start, end, token in scan_text(export, text, stem=surface.name in P9_SURFACES):
                if source_line_allows(export, lines, line, text, start, end):
                    continue
                at = line + text.count("\n", 0, start) if line >= 1 else 0
                # A parsed YAML/JSON value has no line: kept when every source line holding it is marked.
                marked = [at] if line >= 1 else _source_lines_of(lines, line, text, start, end)
                kept = bool(marked) and all(1 <= n <= len(lines) and KEEP.search(lines[n - 1]) for n in marked)
                findings.append(Finding(surface.name, rel, at if line >= 1 else line, token,
                                        classify(text, start, end), text, kept, tuple(marked) if kept else ()))
    return len(files), findings


def scan_all(export: li.Export, root: Path = ROOT) -> dict[str, tuple[int, list[Finding]]]:
    return {s.name: scan_surface(s, export, root) for s in SURFACES}


_SCAN_ALL = scan_all


def judged_here(export: li.Export) -> bool:
    """R23 (operator, 2026-11-11; answers Q-F2): the guard can FAIL only where the full export is,
    which is the private repository (its CI is the enforcement point: a commit syncs to the mirror
    only after it passes). The public mirror carries the public variant, which withholds the old
    hostnames (DR-29) and the private repository names; it cannot allow them, so there the guard
    reports and never fails."""
    return export.header["variant"] == "full"


def mode(export: li.Export, *, enforced: bool) -> str:
    return "enforce" if enforced and judged_here(export) else "report"


def fails(surface: str, finding: Finding, closed: frozenset[str]) -> bool:
    """One finding's verdict in enforce mode. P4 (Q4, R17): a READ finding on a P4 surface fails.
    P9 (042 FR-042.2): an identifier finding on a P4 surface, and any finding on a P9 surface, fails
    once the surface's group is closed, unless its line carries a keep mark."""
    p9 = surface in P9_SURFACES
    if not p9 and finding.kind == "read":
        return True
    if finding.kept:
        return False
    return GROUP_OF_SURFACE.get(surface) in closed


def failures(results: dict[str, tuple[int, list[Finding]]], export: li.Export, *,
             enforced: bool, closed: frozenset[str] | None = None) -> list[Finding]:
    """The findings that fail the guard, in enforce mode only (`fails`)."""
    if mode(export, enforced=enforced) != "enforce":
        return []
    closed = CLOSED_GROUPS if closed is None else closed
    return [f for name, (_p, fs) in results.items() for f in fs if fails(name, f, closed)]


def render_report(results: dict[str, tuple[int, list[Finding]]], export: li.Export, *,
                  enforced: bool) -> str:
    lines = [f"brand guard report — mode {mode(export, enforced=enforced)}, export "
             f"{export.header['variant']} phase {export.phase()}, closed {export.closed()}, "
             f"ENFORCED={enforced}, CLOSED_GROUPS={sorted(CLOSED_GROUPS)}", ""]
    for name, (parsed, findings) in results.items():
        counts = Counter(f.kind for f in findings)
        kept = sum(f.kept for f in findings)
        group = GROUP_OF_SURFACE.get(name, "-")
        state = "closed" if group in CLOSED_GROUPS else "open"
        lines.append(f"{name:12s} files {parsed:5d}  read {counts['read']:5d}  "
                     f"identifier {counts['identifier']:5d}  kept {kept:4d}  {group} {state}")
    lines.append("")
    for _name, (_parsed, findings) in results.items():
        lines += [f.render() for f in findings]
    return "\n".join(lines) + "\n"


def write_report(results: dict[str, tuple[int, list[Finding]]], export: li.Export) -> str:
    text = render_report(results, export, enforced=ENFORCED)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    return text


# ── the guard ─────────────────────────────────────────────────────────────────────────


#: The real tree's scan, left by the guard test for `test_keep_marks_keep_something` (one scan, not two).
_REAL_SCAN: dict[str, object] = {}


def test_no_unlisted_old_name_on_brand_surfaces():
    export = li.load()
    results = scan_all(export)
    if scan_all is _SCAN_ALL:
        _REAL_SCAN.update(export=export, results=results)
    report = write_report(results, export)
    print("\n" + "\n".join(report.splitlines()[:len(SURFACES) + 2]))
    failing = failures(results, export, enforced=ENFORCED)
    assert not failing, ("old names with no record allowing them (FR-040.15; P9 surfaces of closed "
                         "groups, FR-042.2):\n  "
                         + "\n  ".join(f.render() for f in failing[:200]))


def test_a_clean_group_is_closed():
    """042 acceptance F-1 (2026-11-26): `CLOSED_GROUPS`' membership was pinned by nothing, so dropping a cleaned
    group (`"RG-1"`, control M01) switched its enforcement off with the whole module green. Derived from the real
    scan, not a second list: a group whose every surface has no unkept finding at HEAD is clean, and a clean group
    (bar `NEVER_CLOSED`) must be closed. Control: drop `"RG-1"` from `CLOSED_GROUPS` -> red."""
    if _REAL_SCAN:
        results = _REAL_SCAN["results"]
    else:
        results = scan_all(li.load())
    unkept: dict[str, int] = {}
    for name, (_parsed, findings) in results.items():
        group = GROUP_OF_SURFACE.get(name)
        if group is not None:
            unkept[group] = unkept.get(group, 0) + sum(1 for f in findings if not f.kept)
    judged = set(GROUP_OF_SURFACE.values()) - NEVER_CLOSED
    assert judged and set(unkept) >= judged, "the scan judged no group: the check would be vacuous"
    clean = {g for g in judged if unkept[g] == 0}
    assert clean <= CLOSED_GROUPS, f"clean groups not closed (their enforcement is off): {sorted(clean - CLOSED_GROUPS)}"


@pytest.mark.parametrize("surface", SURFACES, ids=[s.name for s in SURFACES])
def test_fail_open_floors(surface):
    parsed = len(_walked(surface, ROOT))
    listed = _listed(surface, ROOT)
    floor = len(listed) if listed is not None else parsed
    assert parsed >= floor, f"{surface.name}: walked {parsed} files, the index lists {floor}"
    assert parsed > 0, f"{surface.name}: parsed nothing — a broken lister must not pass"


def enforce_mode(export: li.Export) -> bool:
    """Enforced exactly when P4 is CLOSED (CF-21: closed, never phase)."""
    return export.closed_at_least("P4")


def test_enforced_follows_closed(tmp_path):
    assert ENFORCED == enforce_mode(li.load()), "ENFORCED must equal the committed export's mode"
    for phase, closed, expected in (("P4", "P3", False), ("P5", "P4", True), ("P1", "P0", False)):
        variant = _export_variant(tmp_path, header={"phase": phase, "closed": closed})
        assert enforce_mode(variant) is expected, (phase, closed)


def _as_variant(tmp_path: Path, variant: str, closed: str = "P4") -> li.Export:
    """A committed export re-headed with P4 closed: the public file for `public`; for `full` the
    full file, or on the mirror (where it is absent) the public file's records under a full header."""
    name = "legacy-identifiers.public.md" if variant == "public" else "legacy-identifiers.md"
    source = li.PUBLIC if variant == "public" or not li.FULL.exists() else li.FULL
    out = []
    for line in source.read_text().splitlines():
        if line.startswith("{") and "format" in json.loads(line):
            header = json.loads(line)
            header.update(variant=variant, phase="P4", closed=closed,
                          withheld=header["withheld"] if variant == "public" else [])
            line = json.dumps(header)
        out.append(line)
    path = tmp_path / variant / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    return li.load(path)


def test_enforced_in_the_private_tree_reported_on_the_mirror(tmp_path):
    """R23 (operator, 2026-11-11; answers Q-F2). With P4 closed the guard FAILS on a read-text
    finding where the full export is (the private repository: a commit syncs to the mirror only after
    its CI passes) and only REPORTS where the public variant is (the mirror: that variant withholds
    the old hostnames, DR-29, and the private repository names, so it cannot allow them there)."""
    full, public = _as_variant(tmp_path, "full"), _as_variant(tmp_path, "public")
    assert (judged_here(full), judged_here(public)) == (True, False)
    assert mode(full, enforced=True) == "enforce" and mode(public, enforced=True) == "report"
    assert mode(full, enforced=False) == mode(public, enforced=False) == "report", "before the flip"

    root = _git_tree(tmp_path, {"admin-ui/src/Zz.tsx": "const el = <p>miLLM settings</p>;\n",
                                f"{PKG}/zz.py": 'NAME = "millm_requests_total"\n'})
    ui = next(s for s in SURFACES if s.name == "typescript")
    for export, expected in ((full, ["admin-ui/src/Zz.tsx"]), (public, [])):
        results = {ui.name: scan_surface(ui, export, root)}
        assert [f.path for f in results[ui.name][1]] == ["admin-ui/src/Zz.tsx"], "found in both trees"
        assert [f.path for f in failures(results, export, enforced=True)] == expected
        assert failures(results, export, enforced=False) == []
        report = render_report(results, export, enforced=True)
        assert "admin-ui/src/Zz.tsx:1: 'miLLM'" in report, "reported in both modes"
        assert f"mode {mode(export, enforced=True)}" in report.splitlines()[0]


def test_the_guard_itself_fails_only_where_it_is_enforced(tmp_path, monkeypatch):
    """Wiring: the guard test's own verdict, run with one planted finding. Flipped and on the full
    export it FAILS; flipped and on the public variant it passes and reports; unflipped it passes."""
    me = __import__(__name__, fromlist=["x"])
    plant = Finding("typescript", "admin-ui/src/Zz.tsx", 1, "miLLM", "read", "miLLM settings")
    monkeypatch.setattr(me, "scan_all", lambda export: {"typescript": (1, [plant])})
    monkeypatch.setattr(me, "REPORT", tmp_path / "brand_report.txt")
    full, public = _as_variant(tmp_path, "full"), _as_variant(tmp_path, "public")

    monkeypatch.setattr(me, "ENFORCED", True)
    monkeypatch.setattr(li, "load", lambda path=None: full)
    with pytest.raises(AssertionError, match="admin-ui/src/Zz.tsx"):
        test_no_unlisted_old_name_on_brand_surfaces()
    monkeypatch.setattr(li, "load", lambda path=None: public)
    test_no_unlisted_old_name_on_brand_surfaces()
    written = (tmp_path / "brand_report.txt").read_text()
    assert "mode report" in written.splitlines()[0] and "admin-ui/src/Zz.tsx:1" in written

    monkeypatch.setattr(me, "ENFORCED", False)
    monkeypatch.setattr(li, "load", lambda path=None: full)
    test_no_unlisted_old_name_on_brand_surfaces()


# ── self-tests ───────────────────────────────────────────────────────────────────────

PLANTS = {
    "string": (f"{PKG}/zz.py", 'LABEL = "Open miLLM now"\n', _python),
    "template_literal": ("admin-ui/src/zz.ts", "const t = `Welcome to miLLM ${x}`;\n", _typescript),
    "jsx_text": ("admin-ui/src/Zz.tsx", "const el = <p>miLLM settings</p>;\n", _typescript),
    "yaml_value": ("k8s/zz.yaml", "metadata:\n  label: miLLM server\n", _by_kind),
    "yaml_key": ("k8s/zz.yaml", "miLLM-key: 1\n", _by_kind),
    "json_value": ("docs/schemas/zz.json", json.dumps({"title2": "the miLLM server"}), _by_kind),
    # Prose keys (040 P4 follow-up): the shared scanner skipped these values, so the Grafana dashboard's
    # description and two manual page titles named the old brand with no finding.
    "json_description": ("k8s/zz.json", json.dumps({"panels": [{"description": "the miLLM dashboard"}]}),
                         _by_kind),
    "yaml_description": ("k8s/zz.yaml", "spec:\n  description: miStudio workbench project\n", _by_kind),
    "json_title": ("docs/schemas/zz.json", json.dumps({"title": "A miLLM document"}), _by_kind),
    "front_matter_title": ("manual/docs/zz.md", "---\ntitle: Introduction to miLLM\n---\n\nBody.\n",
                           _by_kind),
    "front_matter_description": ("manual/docs/zz.mdx",
                                 '---\ndescription: "A detector trained in miStudio"\n---\n\nBody.\n', _by_kind),
    "markdown_text": ("docs/zz.md", "# Using miLLM\n", _by_kind),
    "file_name": ("admin-ui/public/millm-logo.svg", "", _file_name),
    "package_name": ("manual/package.json", json.dumps({"name": "millm-manual"}), _by_kind),
    "toml_value": ("pyproject.toml", '[project.urls]\nHomepage = "https://github.com/miLLM/miLLM"\n', _by_kind),
    "workflow_if": (".github/workflows/zz.yml",
                    "jobs:\n  a:\n    if: github.repository == 'someone/miLLM-fork'\n", _by_kind),
    # P9 surfaces (042 FR-042.2): one plant per extractor path each new surface takes.
    "python_comment": (f"{PKG}/zz.py", "x = 1  # miLLM serves this\n", _comments),
    "python_docstring": ("tests/unit/zz.py", '"""What miStudio sends."""\n', _comments),
    "hash_comment": ("k8s/zz.yaml", "a: 1  # the miLLM pod\n", _comments),
    "typescript_comment": ("admin-ui/src/zz.ts", "const a = 1; // read by miLLM\n", _comments),
    "css_comment": ("manual/src/css/zz.css", "/* matches miLLM's theme */\n", _comments),
    "agent_text": ("admin-ui/README.md", "# miLLM Admin UI\n", _agent_text),
    "test_name": ("tests/unit/zz.py", "def test_millm_answers():\n    pass\n", _test_names),
    "test_name_ts": ("admin-ui/src/__tests__/zz.test.ts", "const millmKeys = [];\n", _test_names),
    "test_value": ("tests/unit/zz.py", 'EXPECTED = "miStudio"\n', _test_values),
    "test_value_ts": ("admin-ui/src/__tests__/zz.test.ts", "expect(x).toBe('miLLM');\n", _test_values),
    "test_file_name": ("tests/unit/test_mistudio_zz.py", "", _test_names),
    "code_name": (f"{PKG}/zz.py", "class MiLLMError(Exception):\n    pass\n", _names),
}
P9_EXTRACTORS = (_comments, _agent_text, _test_names, _test_values, _names)


@pytest.mark.parametrize("category", sorted(PLANTS))
def test_plants_found_in_every_category(category):
    rel, source, extract = PLANTS[category]
    export = li.load()
    stem = extract in P9_EXTRACTORS
    found = [t for _l, text in extract(rel, source) for _s, _e, t in scan_text(export, text, stem=stem)]
    assert found, f"{category}: the planted old name was not found"


def _git_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("add", ".")
    return root


def test_export_files_are_excluded(tmp_path):
    root = _git_tree(tmp_path, {"docs/legacy-identifiers.md": "Use miLLM\n",
                                "x/docs/legacy-identifiers.public.md": "Use miLLM\n",
                                "docs/other.md": "Use miLLM\n"})
    docs = next(s for s in SURFACES if s.name == "docs")
    nested = Surface("nested", ("**/*.md",), _by_kind)
    assert _listed(docs, root) == ["docs/other.md"]
    assert _listed(nested, root) == ["docs/other.md"]
    _parsed, findings = scan_surface(docs, li.load(), root)
    assert [f.path for f in findings] == ["docs/other.md"]


def _export_variant(tmp_path: Path, drop=None, header=None, change=None) -> li.Export:
    lines = li.default_path().read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith("{"):
            rec = json.loads(line)
            if drop and drop(rec):
                continue
            if header and "format" in rec:
                rec.update(header)
            if change:
                change(rec)
            line = json.dumps(rec)
        out.append(line)
    path = tmp_path / li.default_path().name
    path.write_text("\n".join(out) + "\n")
    return li.load(path)


def test_allowances_come_from_records_only(tmp_path):
    text = '"mistudio.probe-definition/v1"'
    assert scan_text(li.load(), text) == [], "allowed by DR-01's record"
    without = _export_variant(tmp_path, drop=lambda r: r.get("id") == "DR-01")
    assert scan_text(without, text), "with the record gone the same text is a finding"


def test_allowance_reads_the_source_line(tmp_path):
    """FTDD §6.7: a token is allowed when its span lies inside a record's match "on the same extracted
    text (or the source line when the extractor reports one)". An extractor strips the quotes and the
    key a record's pattern anchors on, so `"millm": {...}` (DR-34) arrives as the bare text `millm`."""
    python = next(s for s in SURFACES if s.name == "python")
    files = {
        f"{PKG}/keep.py": 'BODY = {\n    "millm": 1,\n}\n',                       # DR-34: the response key
        f"{PKG}/label.py": 'LABEL = "millm"\n',                                    # no record: a finding
        f"{PKG}/both.py": 'PAIR = {\n    "millm": "millm",\n}\n',                  # one occurrence uncovered
        f"{PKG}/var.py": 'BODY = {\n    "millm": _millm.dump(),\n}\n',              # the key, not the variable
        f"{PKG}/doc.py": 'TEXT = """first line\n    "millm": documented\nOpen millm now\n"""\n',
    }
    root = _git_tree(tmp_path, files)
    _parsed, findings = scan_surface(python, li.load(), root)
    by_file = Counter(f.path for f in findings)
    assert by_file == {f"{PKG}/label.py": 1, f"{PKG}/both.py": 2, f"{PKG}/doc.py": 1}, by_file
    assert [f.text for f in findings if f.path == f"{PKG}/doc.py"][0].count("millm") == 2, (
        "inside a multi-line literal each token is judged on its own source line")

    # A parsed YAML value has no line: it is judged on every source line that holds it.
    deploy = next(s for s in SURFACES if s.name == "deploy")
    compose = ("# millm: a comment is not a place a value stands\n"
               "services:\n  db:\n    container_name: millm-db\n    volumes:\n      - millm-pg:/data\n"
               "    environment:\n      - name: PGUSER\n        value: millm\n"     # DR-19: the role
               "      - name: OTHER\n        value: millm-x\n"
               "volumes:\n  millm-pg:\n    name: millm-postgres-data\n")          # DR-41: the volume's name
    root = _git_tree(tmp_path / "deploy", {"docker-compose.yml": compose})
    export = li.load()
    _parsed, findings = scan_surface(deploy, export, root)
    # The public variant withholds DR-19 (CF-2), so on the mirror the role is a finding too.
    role = ["millm"] if "DR-19" in export.withheld() else []
    assert sorted(f.text for f in findings) == sorted(
        ["millm-db", "millm-pg", "millm-pg:/data", "millm-x", *role]), findings


def test_old_repo_name_fails_once_its_record_is_retired(tmp_path):
    """T33 (040 FR-040.40, CF-13), on the real export: since P7 every `repo` record is `retired`, and
    each old repository name the export carries is a finding in text. The record decides, not a list
    here: NC-94 marks this repository's private record back to `transition` and this goes red. A name
    the public variant withholds is not judged on the mirror (its record is not there to decide)."""
    export = li.load()
    records = export.repos()
    assert records, "the export carries the five repo records until S001 drops them"
    for record in records:
        text = f"git clone https://github.com/{record['old']}.git"
        assert scan_text(export, text), f"{record['old']} ({record['status']}): the record allows it"
    assert {r["status"] for r in records} == {"retired"}, records
    # The other half: a `transition` record allows the same text (so the red above is the status's).
    transition = _export_variant(tmp_path, change=lambda r: r.update(status="transition")
                                 if r.get("section") == "repo" else None)
    for record in records:
        assert scan_text(transition, f"git clone https://github.com/{record['old']}.git") == [], record


@pytest.mark.parametrize("phase", ["P4", "P5"])
def test_suite_phrase_rules(tmp_path, phase):
    closed = {"P4": "P3", "P5": "P4"}[phase]
    export = _export_variant(tmp_path, header={"phase": phase, "closed": closed})
    flagged = lambda text: bool(scan_text(export, text))  # noqa: E731
    assert flagged("Open miStudio"), "a bare miStudio is a finding in every phase"
    assert flagged("part of the miStudio suite") is (phase == "P4")
    assert flagged("miSuite") is (phase == "P5"), "G-MISUITE denies from P5 only"
    li.skip_if_withheld(export, "G-STUDIO-PATH")  # the public variant withholds the path rule (CF-2)
    assert flagged("see ~/app/miStudio/scripts/x.sh") is (phase == "P4")
    assert flagged("memory -home-x-sean-app-miStudio") is (phase == "P4")


def test_suite_phrase_rules_read_the_record(tmp_path):
    """CF-22 control: the path allowance is the G-STUDIO-PATH record, not a list here."""
    li.skip_if_withheld(li.load(), "G-STUDIO-PATH")
    export = _export_variant(tmp_path, header={"phase": "P5", "closed": "P4"},
                             drop=lambda r: r.get("id") == "G-STUDIO-PATH")
    assert scan_text(export, "see ~/app/miStudio/scripts/x.sh")


def test_prose_values_are_read_except_a_title_that_is_only_its_property_name():
    """The prose keys' values reach the guard (`prose=True`), with one mechanical exception: the
    `title` a schema generator derives from the property's own name. That title is the name re-cased;
    the name is yielded as the key one level up and judged there (an identifier, or a frozen entry).
    A title somebody wrote is read, on the same property."""
    export = li.load()

    def read(document) -> list[str]:
        return [token for _l, text in _by_kind("docs/schemas/zz.json", json.dumps(document))
                for _s, _e, token in scan_text(export, text) if " " in text]

    prop = "mi" + "studio_zz_id"
    derived = source_scan.derived_title(prop)
    assert derived == "Mi" + "studio Zz Id"
    assert read({"properties": {prop: {"title": derived, "type": "string"}}}) == []
    assert read({"properties": {prop: {"title": derived + " field", "type": "string"}}}) == ["Mi" + "studio"]
    assert read({"properties": {"other": {"title": derived, "type": "string"}}}) == ["Mi" + "studio"]
    assert read({"properties": {prop: {"description": derived, "type": "string"}}}) == ["Mi" + "studio"]
    # The key itself is still yielded, so the name is judged where it is written.
    keys = [text for _l, text in _by_kind("docs/schemas/zz.json", json.dumps({"properties": {prop: {}}}))]
    assert prop in keys
    # The generality guard's reading is unchanged: no prose value at all.
    plain = [t for _l, t in source_scan.items_of("k8s/zz.json", json.dumps({"description": "zq words"}))]
    assert plain == ["description"]
    front = [t for _l, t in source_scan.markdown_items("---\ntitle: zq words\n---\nbody\n")]
    assert front == ["title", "body"]


def test_the_real_dashboard_and_manual_prose_is_read():
    """Wiring on the real tree: the guard's own extractor yields the dashboard's description and a manual
    page's front-matter title (a plant proves the extractor; this proves the surface is fed by it)."""
    deploy = next(s for s in SURFACES if s.name == "deploy")
    manual = next(s for s in SURFACES if s.name == "manual")
    board = "k8s/components/otel-reference/dashboard.json"
    assert board in _walked(deploy, ROOT)
    document = json.loads((ROOT / board).read_text())
    texts = [text for _l, text in deploy.extract(board, (ROOT / board).read_text())]
    assert document["description"] in texts
    panel_notes = [p["description"] for p in document["panels"] if p.get("description")]
    assert panel_notes and all(note in texts for note in panel_notes)
    intro = "manual/docs/intro.md"
    assert intro in _walked(manual, ROOT)
    source = (ROOT / intro).read_text()
    title = next(ln.split(":", 1)[1].strip() for ln in source.splitlines() if ln.startswith("title:"))
    assert title and title in [text for _l, text in manual.extract(intro, source)]


def test_identifier_and_read_classes():
    assert classify("millm_requests_total", 0, 5) == "identifier"
    assert classify("millm.engine", 0, 5) == "identifier"
    assert classify("miLLM", 0, 5) == "read"
    assert classify("Open miLLM", 5, 10) == "read"


def test_scanner_extracts_markdown_and_file_names(tmp_path):
    """040 FR-040.15 additions to the moved scanner: Markdown text (code spans and fenced blocks
    included, front matter by parsed value), a file name as one item, and a lister pathspec."""
    md = "---\nlabel: zq Title\n---\n# Heading `code span`\n\n```\nfenced line\n```\n"
    items = list(source_scan.markdown_items(md))
    texts = [t for _l, t in items]
    assert "zq Title" in texts
    assert (4, "# Heading `code span`") in items
    assert (7, "fenced line") in items
    assert list(source_scan.items_of("docs/x.md", "plain\n")) == [(1, "plain")]
    assert list(source_scan.filename_items("admin-ui/public/millm-icon.svg")) == [
        (0, "admin-ui/public/millm-icon.svg")]

    root = _git_tree(tmp_path, {"docs/a.md": "t\n", "docs/legacy-identifiers.md": "t\n",
                                "x/docs/legacy-identifiers.public.md": "t\n"})
    tree = source_scan.Tree("docs", (".",), (".md",))
    assert source_scan._git_listed(tree, root) == [
        "docs/a.md", "docs/legacy-identifiers.md", "x/docs/legacy-identifiers.public.md"]
    assert source_scan._git_listed(tree, root, (EXPORT_EXCLUDE,)) == ["docs/a.md"]


# ── P9 (042 FR-042.2): closed groups, keep marks, the agent text ───────────────────────


def test_closed_groups_are_enforced(tmp_path):
    """Wiring of `CLOSED_GROUPS`, on a planted tree: each P9 surface's finding fails exactly when its
    group is closed, a P4 surface's identifier finding likewise, a kept line never, and a P4 read
    finding always. Control: `fails` returning False for a closed group turns this red."""
    root = _git_tree(tmp_path, {
        f"{PKG}/zz.py": 'class MiLLMError(Exception):\n    pass\n\n\nKEY = "millm_zz_key"  # miLLM owns this\n',
        "tests/unit/test_zz.py": 'def test_zz():\n    assert "miStudio"  # p9: keep, a planted keep\n\n\n'
                                 'def test_millm_zz():\n    pass\n',
        "CLAUDE.md": "# miLLM\n\n" + "".join(f"{k} x\n\nold miLLM\n\n" for k in KEPT_SECTIONS),
    })
    export = li.load()
    results = {s.name: scan_surface(s, export, root) for s in SURFACES
               if s.name in ("python", "comments", "test_names", "test_values", "names", "agent_text")}
    by = {name: fs for name, (_n, fs) in results.items()}
    assert [f.text for f in by["names"]] == ["MiLLMError"]
    assert [f.kind for f in by["python"]] == ["identifier"]
    assert [f.line for f in by["agent_text"]] == [1], "only the heading; kept sections are not read"
    assert [f.kept for f in by["test_values"]] == [True]
    assert [(f.text, f.line) for f in by["test_names"]] == [("test_millm_zz", 5)]
    assert len(by["comments"]) == 1
    full = _as_variant(tmp_path, "full")
    assert failures(results, full, enforced=True, closed=frozenset()) == []
    every = frozenset(GROUP_OF_SURFACE.values())
    failing = failures(results, full, enforced=True, closed=every)
    assert sorted((f.tree, f.line) for f in failing) == [
        ("agent_text", 1), ("comments", 5), ("names", 1), ("python", 5), ("test_names", 5)], failing
    for name in ("comments", "names", "python", "agent_text", "test_names"):
        group = GROUP_OF_SURFACE[name]
        alone = failures(results, full, enforced=True, closed=frozenset({group}))
        assert {f.tree for f in alone} == {s for s, g in GROUP_OF_SURFACE.items() if g == group} & set(by)
    assert failures(results, _as_variant(tmp_path, "public"), enforced=True, closed=every) == [], "R23"
    assert CLOSED_GROUPS <= every, f"a closed group no surface judges: {sorted(CLOSED_GROUPS - every)}"
    assert not CLOSED_GROUPS & NEVER_CLOSED, "P9-D6: RG-6 values change only with their code values"


def test_keep_marks_keep_something():
    """Every `p9: keep` mark carries a reason and sits on a line where the guard found an old name it
    then kept: a mark that keeps nothing is stale and is removed, so marks cannot pile up unread."""
    if not _REAL_SCAN:
        export = li.load()
        _REAL_SCAN.update(export=export, results=scan_all(export))
    results = _REAL_SCAN["results"]
    kept = {(f.path, n) for _n, (_p, fs) in results.items() for f in fs if f.kept for n in f.marks}
    files = {rel for s in SURFACES if s.name in P9_SURFACES for rel in _walked(s, ROOT)}
    marks, bare = set(), []
    for rel in sorted(files):
        for number, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            if "p9: keep" in line:
                marks.add((rel, number))
                if not re.search(r"p9: keep, \S", line):
                    bare.append(f"{rel}:{number}")
    assert not bare, f"a keep mark without its reason: {bare}"
    stale = sorted(marks - kept)
    assert not stale, f"keep marks that keep nothing (remove them): {stale[:40]}"


def test_agent_text_skips_the_kept_sections_and_refuses_without_them():
    text = "# Project: x\n\nmiLLM now\n\n" + "".join(f"{k}\n\nmiLLM then\n\n" for k in KEPT_SECTIONS)
    read = [line for _n, line in _agent_text("CLAUDE.md", text)]
    assert "miLLM now" in read and "miLLM then" not in read
    fenced = "## A\n```\n## Document Inventory\n```\nafter\n" + "".join(f"{k}\n" for k in KEPT_SECTIONS)
    assert "after" in [line for _n, line in _agent_text("CLAUDE.md", fenced)], "a heading in a fence is text"
    with pytest.raises(AssertionError, match="Status history"):
        list(_agent_text("CLAUDE.md", text.replace("### Status history", "### Status notes")))


def test_comment_extractors():
    py = '"""Module miLLM."""\nx = "# not a comment"  # a comment\ndef f():\n    """Doc."""\n'
    assert sorted(source_scan.python_comments(py)) == [(1, "Module miLLM."), (2, "# a comment"), (4, "Doc.")]
    assert list(source_scan.python_comments(py, docstrings=False)) == [(2, "# a comment")]
    hashed = 'a: 1  # yes\nurl: https://x/y#readme\nb: "#no"\n# whole\n'
    assert list(source_scan.hash_comments(hashed)) == [(1, "# yes"), (4, "# whole")]
    ts = 'const a = "// no"; // yes\n/* two\nlines */ const t = `${b /* in */}`;\n'
    assert list(source_scan.typescript_comments(ts)) == [(1, "// yes"), (2, "/* two\nlines */"), (3, "/* in */")]
    assert [t for _l, t in source_scan.typescript_items(ts)] == [
        t for _l, t in source_scan.typescript_items(ts)], "the lexer is shared"
    assert "// no" in [t for _l, t in source_scan.typescript_items(ts)]
    names = [t for _l, t in source_scan.python_names('import a as b\nclass C(D):\n    def f(self, x=y): "s"\n')]
    assert sorted(names) == ["C", "D", "a", "b", "f", "self", "x", "y"]


def test_stem_sees_camel_compounds_token_does_not():
    for text in ("MiLLMError", "FakeMillm", "TestIsMiStudios", "x_millm_lease", "miSTUDIO's"):
        assert STEM.search(text), text
    assert not TOKEN.search("MiLLMError") and not TOKEN.search("FakeMillm")
    for text in ("similar", "semillmore", "Mistral", "minimilliseconds"):
        assert not STEM.search(text), text
