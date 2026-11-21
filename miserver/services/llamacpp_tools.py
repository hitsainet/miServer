"""Native function calling on the llama.cpp engine (GGUF models), 2026-11-21.

THE DEFECT IT REMOVES. `tools` on a GGUF model was refused before any load ("function calling is
served on the transformers engine only"), so a client that speaks only native tool calls (miForge's
`weightspace` provider: `tools` out, `tool_calls` back, `tool` messages in the history) could not
use a GGUF editor model such as Qwen3.6-27B-GGUF:Q4_K_M.

WHY NOT llama-cpp-python's OWN TOOL SUPPORT (0.3.36, read from its source):
`create_chat_completion(tools=...)` renders the tools through the GGUF template, then for
`tool_choice: "auto"` or `"required"` applies NO constraint and parses nothing (the reply comes back
as content), and for a named function forces a grammar of the bare arguments JSON — not the
template's call syntax — and returns that text as `arguments` unvalidated. Neither is usable.

WHAT THIS MODULE DOES INSTEAD:

* **Prompt.** The model's own chat template (`tokenizer.chat_template` in the GGUF metadata) is
  rendered by `render_chat_prompt`, a copy of llama-cpp-python's `Jinja2ChatFormatter` (the
  renderer its chat handler uses for the same model: an `ImmutableSandboxedEnvironment` with
  `trim_blocks`/`lstrip_blocks`, the `generation` tag and loop controls, the same `tojson`), given
  `tools`, the tool-call history (`tool_calls.template_messages`) and the request's
  `chat_template_kwargs`. A copy, not an import, so it runs where the wheel is not installed;
  `tests/unit/services/test_gguf_tool_calls_real.py` proves it renders byte-identically to the
  class wherever the wheel is present. The prompt is tokenized as that handler tokenizes it
  (`add_bos=False, special=True`).
* **Constraint: a grammar for the call, a lazy trigger for "auto".** llama-cpp-python 0.3.36's
  `Llama` takes one grammar for the whole completion and exposes no lazy (triggered) grammar, and a
  grammar over free text would have to exclude the call marker from every answer (and be applied to
  every token of it). So generation runs in two phases:
    1. FREE: the reply is generated unconstrained and streamed while it is plain text (a reasoning
       block included). In `auto`, generation is cut the moment the template's call marker
       (`<tool_call>`) appears in the ANSWER (after any `<think>` block — a marker inside reasoning
       is not a call). In `required` / a named function, this phase is only the reasoning block of
       a template that opened one, cut at `</think>`; without one there is no free phase.
    2. CALL: generation resumes after the marker under a GBNF grammar that admits exactly the
       template's call syntax for the allowed functions, with each argument constrained by its
       parameter schema (llama-cpp-python's own `SchemaConverter`; a Qwen3.5/3.6 string argument
       is raw text that cannot contain `</parameter`). After the last call the grammar admits only
       end of generation; `parallel_tool_calls: false` admits one call.
  A call is never parsed out of free text and never served malformed: the grammar fixes its
  shape, the parser reads it back strictly, and the arguments are validated with `jsonschema`
  against the FULL parameter schema (a failure is a 500 `constrained_output_invalid`, never a 200).
* **Bounds.** `minimum`/`maximum` in a tool's parameters are ACCEPTED (unlike `response_format`,
  where they are refused): the model is shown them in the rendered tools, the grammar enforces the
  type, and the validation enforces the bound. `pattern`, and every keyword the grammar would drop,
  stay refused (`json_schema_subset.check_llamacpp_tools`).
* **Thinking.** A template that opens `<think>` in the generation prompt (Qwen3.6 does unless
  `chat_template_kwargs.enable_thinking` is false) gets its reasoning generated freely and returned
  in `reasoning_content`: miServer rendered the prompt, so it knows the block was opened. The grammar
  starts only after `</think>`, so it never fights the reasoning. The block is bounded by the
  reasoning allowance (`ToolGeneration._reason`, the same rule as the document path below; live
  failure WSP run `wsp_84dddc47ecaf`, where an `auto` turn spent all 2048 tokens reasoning), and a
  call the budget cuts short is never returned as text.

STRUCTURED OUTPUT WITH THINKING (2026-11-21). A `response_format` json_schema / json_object request
on a GGUF model used to go to llama-cpp-python's chat handler with its grammar applied from the
FIRST generated token. On a template that opens `<think>` in the generation prompt (Qwen3.6 does by
default) that forced the JSON inside the reasoning block: the model could not think, and
`chat_template_kwargs.enable_thinking` never reached the template. Such a request is now rendered
here too (mode "document"), with its `chat_template_kwargs`:
  * prompt opens `<think>`: the reasoning is generated freely (phase 1, cut at `</think>`) within
    the REASONING ALLOWANCE (`reasoning_allowance`: `max_reasoning_tokens`, else half of
    `max_tokens`; the block's closing tag counts against it). If the allowance ends the block,
    miServer closes it (`\n</think>`), never failing or truncating the reply. The document then
    follows under the grammar, starting after `</think>\n\n` (the template's own separator), with
    what is left of `max_tokens`. `reasoning_content` is the reasoning; `content` the JSON only.
  * prompt does not open one (`enable_thinking: false` renders Qwen3.6's closed, empty
    `<think>\n\n</think>\n\n`; a template with no thinking renders nothing): one phase, the
    grammar from the first generated token, as before.

FORMATS constrained here: `qwen3_xml` (Qwen3.5/3.6, Qwen3-Coder) and `hermes` (Qwen2.5/Qwen3,
Hermes). LFM2's Pythonic calls and Llama 3.1's bare JSON are refused on this engine with the reason
(`UNCONSTRAINED_FORMAT`); the transformers engine still serves them.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from miserver.api.json_schema_subset import annotation_only, for_llamacpp_converter
from miserver.core.errors import (
    ConstrainedOutputInvalidError,
    FieldNotHonouredError,
    ToolCallArgumentsInvalidError,
)
from miserver.services.tool_calls import (
    END_MARKERS,
    HERMES,
    QWEN3_XML,
    START_MARKERS,
    ToolPlan,
    parse_qwen3_xml,
    resolve_local_ref,
    template_messages,
    xml_value_is_raw,
)
from miserver.services.tool_calls import _call as _tool_call  # one id/arguments shape for both engines
from miserver.services.tool_calls import _parse_hermes

#: The call formats this engine constrains with a grammar.
GGUF_FORMATS: tuple[str, ...] = (QWEN3_XML, HERMES)

UNCONSTRAINED_FORMAT = (
    "on the llama.cpp engine a tool call is generated under a grammar, and miServer has one for "
    "the <tool_call> formats only (Qwen3.5/3.6 <function=...>, Hermes/Qwen2.5/Qwen3 JSON); this "
    "model's template teaches the '{fmt}' format. Serve it with the transformers engine"
)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

#: The raw-text value of a Qwen3.5/3.6 string argument may contain anything but this.
_XML_VALUE_FORBIDDEN = "</parameter"


# =============================================================================
# The prompt
# =============================================================================


def _tojson(
    x: Any, ensure_ascii: bool = False, indent: int | None = None,
    separators: tuple[str, str] | None = None, sort_keys: bool = False,
) -> str:
    """`Jinja2ChatFormatter.tojson` (and Transformers'): plain `json.dumps`, not jinja2's
    HTML-escaping, key-sorting filter."""
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


def render_chat_prompt(
    template: str,
    messages: list[dict[str, Any]],
    *,
    bos: str,
    eos: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    template_kwargs: Mapping[str, Any] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render a GGUF chat template exactly as llama-cpp-python 0.3.36's `Jinja2ChatFormatter` does,
    with the request's tools, history and template variables.

    Raises:
        ValueError: the template raised (`raise_exception`), e.g. an unsupported role order.
    """
    import jinja2
    import jinja2.ext
    import jinja2.parser
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class _IgnoreGenerationTags(jinja2.ext.Extension):
        tags = {"generation"}

        def parse(self, parser: jinja2.parser.Parser) -> Any:
            parser.stream.skip(1)
            return parser.parse_statements(("name:endgeneration",), drop_needle=True)

    env = ImmutableSandboxedEnvironment(
        loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True,
        extensions=[_IgnoreGenerationTags, jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = _tojson

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    kwargs = dict(template_kwargs or {})
    # The formatter's own names win over a same-named template variable, as in its `render` call
    # (where a duplicate would be a TypeError): a client cannot replace `messages` or `tools`.
    for reserved in ("messages", "eos_token", "bos_token", "raise_exception",
                     "add_generation_prompt", "functions", "function_call", "tools",
                     "tool_choice", "strftime_now"):
        kwargs.pop(reserved, None)
    return env.from_string(template).render(
        messages=messages,
        eos_token=eos,
        bos_token=bos,
        raise_exception=raise_exception,
        add_generation_prompt=add_generation_prompt,
        functions=None,
        function_call=None,
        tools=tools,
        tool_choice=tool_choice,
        strftime_now=lambda f: datetime.now().strftime(f),
        **kwargs,
    )


def prompt_opens_think(prompt: str) -> bool:
    """Whether the rendered prompt leaves a `<think>` block open (the generation prompt opened it)."""
    return prompt.rstrip().endswith(THINK_OPEN)


def answer_start(text: str, opened_think: bool) -> int | None:
    """Where the ANSWER begins in generated `text`, or None while it is still reasoning (or might
    still become a `<think>` block). A call marker counts only from here."""
    if opened_think:
        i = text.find(THINK_CLOSE)
        return None if i < 0 else i + len(THINK_CLOSE)
    stripped = text.lstrip()
    lead = len(text) - len(stripped)
    if stripped.startswith(THINK_OPEN):
        i = text.find(THINK_CLOSE, lead)
        return None if i < 0 else i + len(THINK_CLOSE)
    if THINK_OPEN.startswith(stripped):  # "", "<", "<th": undecided
        return None
    return 0


# =============================================================================
# The grammar
# =============================================================================


def _literal(text: str) -> str:
    """A GBNF string literal."""
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _xml_text_rules() -> list[str]:
    """Raw text that never contains `</parameter` (so the value ends at the first `</parameter>`
    and the parser reads exactly what the grammar produced). The automaton of that substring:
    state k = the last k characters read are its first k; reading its next character advances,
    reading `<` restarts at 1, anything else at 0; completing it is the one forbidden move."""
    word = _XML_VALUE_FORBIDDEN
    rules = ['tc-xtext ::= tc-x0']
    for k in range(len(word)):
        alts = []
        if k == 0:
            alts.append('[^<] tc-x0')
            alts.append('"<" tc-x1')
        else:
            nxt = word[k]
            alts.append('"<" tc-x1')
            alts.append(f'[^<{nxt}] tc-x0')
            if k + 1 < len(word):
                alts.append(f'{_literal(nxt)} tc-x{k + 1}')
        rules.append(f"tc-x{k} ::= ( {' | '.join(alts)} )?")
    return rules


#: llama-cpp-python 0.3.36's JSON string character (`PRIMITIVE_RULES["char"]`): it admits a raw
#: control character, which JSON forbids — measured on the tiny model, 2 calls in 30 carried a raw
#: newline inside a string and did not parse. Rewritten to exclude U+0000..U+001F.
CONVERTER_CHAR_RULE = (
    r'char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])'
)
STRICT_CHAR_RULE = (
    r'char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] '
    r'[0-9a-fA-F])'
)


def strict_json_strings(gbnf: str) -> str:
    """`gbnf` with the converter's JSON string character made JSON's (no raw control character).
    Applied to every grammar miServer builds with the converter (tool calls and response_format)."""
    return "\n".join(STRICT_CHAR_RULE if line == CONVERTER_CHAR_RULE else line
                     for line in gbnf.split("\n"))


def _rule_safe(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]+", "-", text)


def _prefix_defs(schema: dict[str, Any], prefix: str) -> dict[str, Any]:
    """A copy of `schema` whose root `$defs` names (and every `#/$defs/...` reference to them) carry
    `prefix`, so several tools' definitions share one converter without colliding."""
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs")
    if not isinstance(defs, dict) or not defs:
        return schema
    renamed = {name: f"{prefix}{name}" for name in defs}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref[len("#/$defs/"):]
                if name in renamed:
                    node["$ref"] = "#/$defs/" + renamed[name]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    schema["$defs"] = {renamed[name]: value for name, value in schema["$defs"].items()}
    return schema


def tool_parameters(tool: Mapping[str, Any]) -> dict[str, Any]:
    """A tool's parameter schema; an absent one, or one of annotations alone, is an object with no
    properties (a call's arguments are always an object)."""
    function = tool.get("function") or {}
    parameters = function.get("parameters")
    if parameters is None or annotation_only(parameters):
        return {"type": "object", "properties": {}}
    return dict(parameters)


def _refuse_tools(why: str) -> FieldNotHonouredError:
    return FieldNotHonouredError(
        f"'tools' is not honoured on /v1/chat/completions with the llamacpp engine: {why}",
        details={"param": "tools", "fields": ["tools"], "endpoint": "chat", "engine": "llamacpp"},
    )


def _xml_raw_rule(schema: Mapping[str, Any], where: str) -> str:
    """The GBNF for a raw-text (string) Qwen3.5/3.6 argument."""
    if "const" in schema or "enum" in schema:
        values = [schema["const"]] if "const" in schema else list(schema["enum"])
        for v in values:
            if not isinstance(v, str):
                raise _refuse_tools(f"{where}: a string enum/const value that is not a string")
            if _XML_VALUE_FORBIDDEN in v:
                raise _refuse_tools(f"{where}: a value containing '</parameter' cannot be "
                                    "written as a raw argument")
        return "( " + " | ".join(_literal(v) for v in values) + " )"
    if "minLength" in schema or "maxLength" in schema:
        # Bounded: counted characters, none of them `<` (stricter than the schema, always valid).
        lo = int(schema.get("minLength", 0))
        hi = schema.get("maxLength")
        return f"[^<]{{{lo},{'' if hi is None else int(hi)}}}"
    return "tc-xtext"


def build_call_grammar(
    fmt: str,
    tools: Sequence[Mapping[str, Any]],
    *,
    allowed: frozenset[str],
    parallel: bool,
    converter: Any,
) -> str:
    """The GBNF for what follows the call marker: one call (or, with `parallel`, one or more) of an
    `allowed` function in `fmt`'s syntax, each argument constrained by its parameter schema.

    `converter` is a fresh `llama_cpp.llama_grammar.SchemaConverter` (its `resolve_refs`, `visit`
    and `format_grammar`; injected so this builds without the wheel in tests).

    Raises:
        FieldNotHonouredError: a tool whose parameters the format cannot express (`param: tools`).
    """
    marker, end = START_MARKERS[fmt], END_MARKERS[fmt]
    assert marker is not None and end is not None
    alternatives: list[str] = []
    own: list[str] = []
    for i, tool in enumerate(tools):
        function = tool.get("function") or {}
        name = function.get("name")
        if name not in allowed:
            continue
        prefix = f"t{i}-"
        # An annotation-only argument schema accepts any JSON value: the converter is given its
        # generic `value` rule for it (`for_llamacpp_converter`; the root stays the object).
        parameters = _prefix_defs(
            for_llamacpp_converter(tool_parameters(tool), rewrite_root=False), prefix
        )
        if fmt == HERMES:
            defs = parameters.pop("$defs", None)
            call_schema: dict[str, Any] = {
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": parameters},
                "required": ["name", "arguments"],
            }
            if defs:
                call_schema["$defs"] = defs
            resolved = converter.resolve_refs(call_schema, "stdin")
            alternatives.append(converter.visit(resolved, f"t{i}-call"))
            continue
        # Qwen3.5/3.6: <function=NAME>\n then each parameter block, in schema order.
        resolved = converter.resolve_refs(parameters, "stdin")
        root = parameters  # unresolved copy: `$ref` lookups by pointer for the raw/JSON decision
        kind = parameters.get("type")
        properties = parameters.get("properties") or {}
        if kind not in (None, "object") or not isinstance(properties, dict) or any(
            k in parameters for k in ("anyOf", "$ref", "enum", "const")
        ):
            raise _refuse_tools(
                f"tools[{i}] ({name}): the Qwen3.5/3.6 call format names each argument, so the "
                "parameters must be an object schema with `properties`"
            )
        required = set(parameters.get("required") or [])
        blocks: list[str] = []
        for pname, sub in properties.items():
            if not isinstance(pname, str) or not pname or any(c in pname for c in "<>\r\n"):
                raise _refuse_tools(f"tools[{i}] ({name}): the parameter name {pname!r} cannot "
                                    "be written in the Qwen3.5/3.6 call format")
            where = f"tools[{i}] ({name}) parameter '{pname}'"
            target = resolve_local_ref(sub, root)
            if xml_value_is_raw(sub, root):
                value = _xml_raw_rule(target, where)
            else:
                value = converter.visit(resolved["properties"][pname], f"t{i}-{_rule_safe(pname)}")
            block = (f'{_literal(f"<parameter={pname}>" + chr(10))} {value} '
                     f'{_literal(chr(10) + "</parameter>" + chr(10))}')
            blocks.append(block if pname in required else f"( {block} )?")
        rule = f"tc-f{i}"
        own.append(f'{rule} ::= {_literal(f"<function={name}>" + chr(10))} '
                   f'{" ".join(blocks)} {_literal("</function>" + chr(10))}')
        alternatives.append(rule)
    if not alternatives:
        raise _refuse_tools("tool_choice names no function in tools")
    one = "tc-call"
    if fmt == HERMES:
        own.append(f"{one} ::= {_literal(chr(10))} ( {' | '.join(alternatives)} ) "
                   f"{_literal(chr(10))}? {_literal(end)}")
    else:
        own.append(f"{one} ::= {_literal(chr(10))} ( {' | '.join(alternatives)} ) {_literal(end)}")
        own.extend(_xml_text_rules())
    more = f" ( {_literal(chr(10) + marker)} {one} )*" if parallel else ""
    own.insert(0, f"root ::= {one}{more}")
    return strict_json_strings("\n".join(own) + "\n" + str(converter.format_grammar()) + "\n")


# =============================================================================
# The request's plan on this engine
# =============================================================================


def reasoning_allowance(total: int, requested: int | None) -> int:
    """The reasoning allowance of a structured-output request: `max_reasoning_tokens` when sent,
    else half of `max_tokens` (rounded down). `total` is the effective `max_tokens`; a requested
    allowance that is not smaller is refused before this is reached (`request_policy`)."""
    return requested if requested is not None else total // 2


@dataclass(frozen=True)
class LlamaToolRun:
    """Everything one GGUF request's tool-aware generation needs, decided before the slot."""

    prompt: str
    prompt_tokens: list[int]
    opened_think: bool
    plan: ToolPlan | None = None
    #: "none" (history only: one free phase), "auto", "forced" (`required` / a named function), or
    #: "document" (a constrained `response_format`: free reasoning, then the document's grammar).
    mode: str = "none"
    grammar_text: str | None = None
    grammar: Any = None
    #: name -> the FULL parameter schema (bounds included), for validating each call.
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The most tokens a prompt-opened reasoning block may take, its closing tag included
    #: ("document", "auto" and "forced"; None: half of `max_tokens`).
    reasoning_allowance: int | None = None
    #: "auto": tokens the free answer leaves for a call it may still open (`call_reserve`).
    call_reserve: int = 0

    @property
    def marker(self) -> str | None:
        return START_MARKERS[self.plan.fmt] if self.plan is not None else None


#: Tokens added to the shortest possible call when an "auto" answer holds a call's worth back: room
#: for argument values beyond the shortest ones (a heuristic; the call still ends `length` if it
#: needs more, and is then never returned as text).
CALL_MARGIN = 64


def shortest_call(fmt: str, tools: Sequence[Mapping[str, Any]], allowed: frozenset[str]) -> str:
    """The shortest call text an allowed function admits, marker included: required arguments
    only, each a one-character value."""
    marker, end = START_MARKERS[fmt], END_MARKERS[fmt]
    best: str | None = None
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if name not in allowed:
            continue
        required = [p for p in (tool_parameters(tool).get("required") or []) if isinstance(p, str)]
        if fmt == HERMES:
            body = json.dumps({"name": name, "arguments": {p: 0 for p in required}})
            text = f"{marker}\n{body}\n{end}"
        else:
            blocks = "".join(f"<parameter={p}>\n0\n</parameter>\n" for p in required)
            text = f"{marker}\n<function={name}>\n{blocks}</function>\n{end}"
        if best is None or len(text) < len(best):
            best = text
    return best or ""


def forced_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
    else:
        name = getattr(getattr(tool_choice, "function", None), "name", None)
    return name if isinstance(name, str) else None


def choice_value(tool_choice: Any) -> Any:
    """`tool_choice` as the plain value a template receives."""
    if hasattr(tool_choice, "model_dump"):
        return tool_choice.model_dump()
    return tool_choice


def plan_mode(plan: ToolPlan | None, tool_choice: Any) -> str:
    if plan is None:
        return "none"
    if tool_choice == "required" or forced_name(tool_choice):
        return "forced"
    return "auto"


# =============================================================================
# Generation
# =============================================================================


@dataclass
class ToolRunResult:
    """One generation's outcome. `text` is what the model wrote outside a returned call (its
    reasoning block included, split by the caller); `tool_calls` are OpenAI calls, or empty."""

    text: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    completion_tokens: int
    prompt_tokens: int
    #: "document" mode only: the reasoning block's text (None when there was none), its tokens
    #: (part of `completion_tokens`), and whether the allowance ended it (miServer closed it).
    reasoning: str | None = None
    reasoning_tokens: int | None = None
    reasoning_cut_short: bool = False


def _longest_marker_prefix(text: str, marker: str) -> int:
    for k in range(min(len(marker) - 1, len(text)), 0, -1):
        if marker.startswith(text[-k:]):
            return k
    return 0


def _count(llama: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(llama.tokenize(text.encode("utf-8"), add_bos=False, special=True))
    except Exception:  # noqa: BLE001 - a count must never fail a request
        return 0


def validate_arguments(name: str, arguments: Any, schema: Mapping[str, Any]) -> None:
    """The second line of defence, and the ONLY enforcement of `minimum`/`maximum`.

    Raises:
        ConstrainedOutputInvalidError: naming the tool, the keyword and its JSON pointer.
    """
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    from jsonschema.exceptions import best_match  # type: ignore[import-untyped]

    error = best_match(Draft202012Validator(dict(schema)).iter_errors(arguments))
    if error is None:
        return
    pointer = "/" + "/".join(str(p) for p in error.absolute_path)
    bound = error.validator_value if isinstance(error.validator_value, (int, float, str)) else None
    # Never the model's value: the message names the schema, which the client sent.
    raise ToolCallArgumentsInvalidError(
        f"the model's call to '{name}' does not satisfy its parameter schema: "
        f"{error.validator}{'' if bound is None else ' ' + json.dumps(bound)} at {pointer or '/'}. "
        "The grammar enforces every keyword but minimum and maximum, which this check enforces "
        "after generation; the request may be retried.",
        details={"tool": name, "keyword": str(error.validator), "pointer": pointer},
    )


class ToolGeneration:
    """Drive the two phases (module docstring) over a llama-cpp-python `Llama`.

    `events()` yields the reply's free text as it is generated (never a call marker, never a call),
    then sets `result`. Run it in one thread; closing the iterator early closes llama's stream.
    """

    def __init__(self, llama: Any, run: LlamaToolRun, params: Mapping[str, Any]) -> None:
        self.llama = llama
        self.run = run
        self.params = dict(params)
        self.result: ToolRunResult | None = None

    def _free(
        self, budget: int, *, until: str | None, prefix: str = "", opened: bool | None = None,
    ) -> Iterator[str]:
        """Phase 1, streamed. Sets `self._free_text` (what to return as text), `self._free_finish`
        ("stop"/"length", None when this cut it) and `self._cut` (the text up to and including what
        ended it — the marker, or `</think>` — or None when nothing did).

        `until`: "marker" (auto: stop at a call marker in the answer), "think" (stop at the end of
        the reasoning block the prompt opened), or None (history only: run to the end). `prefix`:
        text already decided after the prompt (a closed reasoning block), generation continuing
        from it; `opened`: whether a `<think>` block is open where this phase starts (default: the
        prompt's).
        """
        self._free_text, self._free_finish, self._cut = "", None, None
        if budget <= 0:
            self._free_finish = "length"
            return
        run = self.run
        marker = run.marker
        opened_think = run.opened_think if opened is None else opened
        # Never a grammar here: free text is free (a "document" run's grammar is the call phase's).
        free_params = {k: v for k, v in self.params.items() if k != "grammar"}
        tail = (self.llama.tokenize(prefix.encode("utf-8"), add_bos=False, special=True)
                if prefix else [])
        stream = self.llama.create_completion(
            prompt=list(run.prompt_tokens) + list(tail), stream=True,
            **{**free_params, "max_tokens": budget},
        )
        acc, emitted = "", 0
        try:
            for chunk in stream:
                choice = (chunk.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    self._free_finish = choice["finish_reason"]
                acc += choice.get("text") or ""
                start = answer_start(acc, opened_think)
                safe, cut_at = len(acc), None
                if until == "think":
                    if start is not None:
                        cut_at = safe = start  # `</think>` closed the block: the call follows it
                elif until == "marker" and marker is not None:
                    if start is None:
                        # Reasoning streams as it comes; text that might yet open a `<think>` block
                        # is held (it could be the start of a marker too).
                        if not (opened_think or acc.lstrip().startswith(THINK_OPEN)):
                            safe = emitted
                    else:
                        found = acc.find(marker, start)
                        if found >= 0:
                            cut_at = found + len(marker)
                            safe = found
                        else:
                            safe = len(acc) - _longest_marker_prefix(acc[start:], marker)
                safe = max(safe, emitted)
                if safe > emitted:
                    yield acc[emitted:safe]
                    emitted = safe
                if cut_at is not None:
                    self._cut = acc[:cut_at]
                    self._free_finish = None
                    # What is returned as text: everything before the marker (the marker and the
                    # call are the call's), or the whole reasoning block.
                    acc = acc[:safe] if until == "marker" else self._cut
                    if len(acc) > emitted:
                        yield acc[emitted:]
                        emitted = len(acc)
                    break
            else:
                if len(acc) > emitted:
                    yield acc[emitted:]
                    emitted = len(acc)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        self._free_text = acc

    def _call(self, continuation: str, budget: int) -> tuple[str, str, int]:
        """Phase 2: (body text, finish reason, tokens) of the grammar-constrained call."""
        if budget <= 0:
            return "", "length", 0
        tail = (self.llama.tokenize(continuation.encode("utf-8"), add_bos=False, special=True)
                if continuation else [])
        params = {k: v for k, v in self.params.items() if k not in ("stop", "grammar")}
        params["max_tokens"] = budget
        raw = self.llama.create_completion(
            prompt=list(self.run.prompt_tokens) + list(tail), stream=False,
            grammar=self.run.grammar, **params,
        )
        choice = (raw.get("choices") or [{}])[0]
        used = int((raw.get("usage") or {}).get("completion_tokens", 0))
        return choice.get("text") or "", choice.get("finish_reason") or "stop", used

    def _parse(self, call_text: str) -> list[dict[str, Any]]:
        plan = self.run.plan
        assert plan is not None
        if plan.fmt == QWEN3_XML:
            parsed = parse_qwen3_xml(call_text, plan, strict=True)
        else:
            parsed = _parse_hermes(call_text)
        if parsed is None or not parsed[1]:
            raise ConstrainedOutputInvalidError(
                "the grammar-constrained tool call did not parse back as a call "
                f"({len(call_text)} characters)",
                details={"format": plan.fmt},
            )
        calls = parsed[1]
        if any(name not in plan.names for name, _ in calls):
            raise ConstrainedOutputInvalidError(
                "the grammar-constrained tool call names a function the request did not offer",
                details={"format": plan.fmt},
            )
        if not plan.parallel and len(calls) > 1:
            raise ConstrainedOutputInvalidError(
                "the grammar admitted more than one call with parallel_tool_calls false",
                details={"format": plan.fmt},
            )
        for name, arguments in calls:
            validate_arguments(name, arguments, self.run.schemas.get(name, {}))
        return [_tool_call(name, arguments) for name, arguments in calls]

    def _reason(self, budget: int, *, close_an_ended_turn: bool) -> Iterator[str]:
        """The reasoning block the prompt opened, generated freely within the REASONING ALLOWANCE
        (`run.reasoning_allowance`, else half of `budget`; its closing tag counts against it), and
        closed here (`\n</think>`) when the allowance ends it. Yields exactly the text that precedes
        the answer: the block, its closing tag and the template's `\n\n` separator.

        Sets `self._prefix` (that text, the next phase's continuation), `self._reasoning` (the
        reasoning, stripped, or None), `self._reasoning_tokens` (the block's tokens, closing tag
        included), `self._reasoning_cut_short` (the allowance ended it) and `self._turn_ended` (the
        model ended its turn inside the block and `close_an_ended_turn` is False: there is no
        answer to continue to, and `self._prefix` is the raw reasoning).
        """
        run = self.run
        allowance = (run.reasoning_allowance if run.reasoning_allowance is not None
                     else reasoning_allowance(budget, None))
        allowance = min(allowance, max(budget - 1, 0))  # the answer keeps at least one token
        close = "\n" + THINK_CLOSE
        # The closing tag is part of the block, so it comes out of the allowance too.
        yield from self._free(allowance - _count(self.llama, close), until="think")
        text, cut, finish = self._free_text, self._cut, self._free_finish
        self._reasoning_cut_short = False
        self._turn_ended = False
        if cut is None:
            if finish != "length" and not close_an_ended_turn:
                self._turn_ended = True
                self._prefix = text
                self._reasoning = text.strip() or None
                self._reasoning_tokens = _count(self.llama, text)
                return
            # Still reasoning: the allowance ran out ("length") or the turn ended. Closed here, so
            # the answer is generated rather than the reply failing or truncating.
            self._reasoning_cut_short = finish == "length"
            body, cut = text, text + close
            yield close
        else:
            body = cut[: -len(THINK_CLOSE)]
        self._reasoning = body.strip() or None
        self._reasoning_tokens = _count(self.llama, cut)
        # What follows begins AFTER the block and the template's own separator (Qwen3.6 writes
        # `</think>\n\n` before an answer), never inside the reasoning.
        self._prefix = cut + "\n\n"
        yield "\n\n"

    def _document(self, budget: int) -> Iterator[str]:
        """A constrained `response_format` (module docstring, STRUCTURED OUTPUT WITH THINKING):
        the reasoning block (`_reason`), then the document under the grammar, starting after
        `</think>` and its separator, with the rest of `budget`."""
        run = self.run
        reasoning: str | None = None
        reasoning_tokens = 0
        cut_short = False
        continuation = ""
        if run.opened_think:
            yield from self._reason(budget, close_an_ended_turn=True)
            reasoning, reasoning_tokens = self._reasoning, self._reasoning_tokens
            cut_short, continuation = self._reasoning_cut_short, self._prefix
        document, finish, used = self._call(continuation, budget - reasoning_tokens)
        self.result = ToolRunResult(
            document, [], finish, reasoning_tokens + used, len(run.prompt_tokens),
            reasoning=reasoning, reasoning_tokens=reasoning_tokens, reasoning_cut_short=cut_short,
        )

    def events(self) -> Iterator[str]:
        """The reply's text as it is generated (the reasoning block included, never a call marker
        or a call); then `self.result`. TOOLS (2026-11-21, the reasoning allowance on this path,
        live failure WSP run `wsp_84dddc47ecaf`): a prompt-opened reasoning block is bounded as on
        the document path (`_reason`), then
          * forced (`required` / named): the call under the grammar, after the block;
          * auto: the answer, free, from after the block, with `run.call_reserve` held back so a
            call opened late still fits; a marker in it opens the call phase.
        A call the budget cuts short is never returned as text: the reply is `length`, its text
        ending before the marker (the reasoning kept)."""
        run = self.run
        budget = int(self.params.get("max_tokens") or 0)
        prompt_n = len(run.prompt_tokens)
        if run.mode == "document":
            yield from self._document(budget)
            return
        if run.mode == "none":
            # History only, no tools shown: one free phase, as before.
            yield from self._free(budget, until=None)
            text = self._free_text
            self.result = ToolRunResult(text, [], self._free_finish or "stop",
                                        _count(self.llama, text), prompt_n)
            return

        prefix, reasoning_tokens, cut_short = "", 0, False
        if run.opened_think:
            yield from self._reason(budget, close_an_ended_turn=run.mode == "forced")
            reasoning_tokens, cut_short = self._reasoning_tokens, self._reasoning_cut_short
            if self._turn_ended:
                # auto: the model ended its turn inside the reasoning; that is the whole reply.
                self.result = ToolRunResult(
                    self._prefix, [], self._free_finish or "stop", reasoning_tokens, prompt_n,
                    reasoning_tokens=reasoning_tokens)
                return
            prefix = self._prefix
        remaining = budget - reasoning_tokens

        def finish_without_a_call(answer: str, finish: str, used: int) -> None:
            self.result = ToolRunResult(prefix + answer, [], finish, used, prompt_n,
                                        reasoning_tokens=reasoning_tokens,
                                        reasoning_cut_short=cut_short)

        marker = run.marker or ""
        if run.mode == "forced":
            answer, continuation, used = "", prefix + marker, reasoning_tokens
        else:
            # The answer may still open a call: keep a call's worth back (never more than half).
            reserve = min(run.call_reserve, remaining // 2)
            yield from self._free(remaining - reserve, until="marker", prefix=prefix, opened=False)
            answer, cut, finish_free = self._free_text, self._cut, self._free_finish
            if cut is None:
                if finish_free == "length":
                    # Cut while it may have been opening the marker: never return half of one.
                    answer = answer[: len(answer) - _longest_marker_prefix(answer, marker)]
                finish_without_a_call(answer, finish_free or "stop",
                                      reasoning_tokens + _count(self.llama, answer))
                return
            continuation = prefix + cut
            used = reasoning_tokens + _count(self.llama, cut)
        body, finish, used_call = self._call(continuation, budget - used)
        used += used_call
        if finish == "length":
            # The budget ended inside the call: no call, and the opened marker is not content.
            finish_without_a_call(answer, "length", used)
            return
        calls = self._parse(marker + body)
        self.result = ToolRunResult(prefix + answer, calls, "tool_calls", used, prompt_n,
                                    reasoning_tokens=reasoning_tokens,
                                    reasoning_cut_short=cut_short)


def as_chat_chunks(generation: "ToolGeneration") -> Iterator[dict[str, Any]]:
    """A tool generation as llama-cpp-python-shaped chat chunks, for the streaming path: one
    `delta.content` chunk per piece of free text, then one chunk carrying the calls (in `delta`),
    the finish reason and the usage. Closing it closes llama's stream."""
    sent = ""
    for piece in generation.events():
        sent += piece
        yield {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
    result = generation.result
    assert result is not None
    # Text the result returns beyond what streamed (none since 2026-11-21: a call the budget cuts
    # short is not returned as text, and every other piece of the result was yielded as it came).
    if result.text.startswith(sent) and len(result.text) > len(sent):
        yield {"choices": [{"index": 0, "delta": {"content": result.text[len(sent):]},
                            "finish_reason": None}]}
    delta: dict[str, Any] = {}
    if result.tool_calls:
        delta["tool_calls"] = result.tool_calls
    yield {
        "choices": [{"index": 0, "delta": delta, "finish_reason": result.finish_reason}],
        "usage": {"prompt_tokens": result.prompt_tokens,
                  "completion_tokens": result.completion_tokens},
        "tool_run_result": result,
    }


def history_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """The message dicts the template receives (role, content, tool calls, tool call ids)."""
    return template_messages(messages)


def schemas_by_name(tools: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = (tool.get("function") or {}).get("name")
        if isinstance(name, str):
            out[name] = tool_parameters(tool)
    return out


ConverterFactory = Callable[[], Any]
