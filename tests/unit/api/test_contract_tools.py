"""The answer contract's pure transforms, on small synthetic schemas (044 task 1.1, FTID §3.1)."""

from __future__ import annotations

from miserver.api.schemas import contract_tools as tools


def test_always_sent_required_lists_every_unmarked_property_and_drops_the_marker():
    defs = {
        "A": {
            "type": "object",
            "properties": {
                "b": {"type": "integer"},
                "a": {"type": "string", tools.MAY_BE_ABSENT: True},
                "c": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["b"],
        },
        "Empty": {
            "type": "object",
            "properties": {"x": {tools.MAY_BE_ABSENT: True}},
            "required": ["x"],
        },
        "Map": {"type": "object", "additionalProperties": {"type": "integer"}},
    }
    tools.always_sent_required(defs)
    assert defs["A"]["required"] == ["b", "c"]
    assert tools.MAY_BE_ABSENT not in defs["A"]["properties"]["a"]
    assert "required" not in defs["Empty"]
    assert "required" not in defs["Map"]


def test_open_objects_removes_closed_objects_and_keeps_maps():
    node = {
        "a": {"type": "object", "properties": {}, "additionalProperties": False},
        "m": {"type": "object", "additionalProperties": {"type": "string"}},
        "t": {"type": "object", "additionalProperties": True},
    }
    tools.open_objects(node)
    assert "additionalProperties" not in node["a"]
    assert node["m"]["additionalProperties"] == {"type": "string"}
    assert node["t"]["additionalProperties"] is True


def test_strict_form_closes_objects_on_a_copy_and_leaves_maps_and_conditions():
    node = {
        "type": "object",
        "properties": {"x": {"type": "object", "properties": {"y": {}}}},
        "allOf": [{"if": {"properties": {"k": {"const": 1}}}, "then": {"properties": {"x": {}}}}],
        "$defs": {"M": {"type": "object", "additionalProperties": {"type": "integer"}}},
    }
    strict = tools.strict_form(node)
    assert strict["additionalProperties"] is False
    assert strict["properties"]["x"]["additionalProperties"] is False
    assert strict["$defs"]["M"]["additionalProperties"] == {"type": "integer"}
    # A condition's partial view of the object must not close it.
    assert "additionalProperties" not in strict["allOf"][0]["if"]
    assert "additionalProperties" not in strict["allOf"][0]["then"]
    assert "additionalProperties" not in node


def test_reachable_defs_follows_references_and_drops_the_rest():
    defs = {
        "A": {"$ref": "#/$defs/B"},
        "B": {"properties": {"c": {"$ref": "#/$defs/C"}}},
        "C": {},
        "Lost": {"$ref": "#/$defs/A"},
    }
    assert list(tools.reachable_defs([{"x": {"$ref": "#/$defs/A"}}], defs)) == ["A", "B", "C"]


def test_history_form_copies_definitions_without_required():
    defs = {
        "M": {"type": "object", "properties": {"n": {"$ref": "#/$defs/N"}}, "required": ["n"]},
        "N": {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]},
    }
    out = tools.history_form({"$ref": "#/$defs/M"}, defs)
    assert out == {"$ref": "#/$defs/M__history"}
    assert "required" not in defs["M__history"] and "required" not in defs["N__history"]
    assert defs["M__history"]["properties"]["n"] == {"$ref": "#/$defs/N__history"}
    assert defs["M"]["required"] == ["n"] and defs["N"]["required"] == ["v"]
    assert defs["N__history"]["properties"]["v"] == {"type": "integer"}


def test_is_typed_follows_fr_044_4():
    defs = {
        "Obj": {"type": "object", "properties": {"a": {}}},
        "Map": {"type": "object", "additionalProperties": True},
        "Env": {
            "type": "object",
            "properties": {
                "success": {},
                "data": {"anyOf": [{"$ref": "#/$defs/Obj"}, {"type": "null"}]},
                "error": {},
            },
        },
        "EnvAny": {"type": "object", "properties": {"success": {}, "data": {}, "error": {}}},
        "EnvNone": {
            "type": "object",
            "properties": {"success": {}, "data": {"type": "null"}, "error": {}},
        },
    }
    ext = {"https://hitsai.net/schemas/x.json"}
    assert tools.is_typed({"$ref": "#/$defs/Obj"}, defs)
    assert tools.is_typed({"type": "array", "items": {"$ref": "#/$defs/Obj"}}, defs)
    assert tools.is_typed({"anyOf": [{"$ref": "#/$defs/Obj"}, {"type": "null"}]}, defs)
    assert tools.is_typed({"$ref": "#/$defs/Env"}, defs)
    assert tools.is_typed({"$ref": "#/$defs/EnvNone"}, defs)
    assert tools.is_typed({"$ref": "https://hitsai.net/schemas/x.json"}, defs, ext)
    assert not tools.is_typed({"$ref": "https://hitsai.net/schemas/y.json"}, defs, ext)
    assert not tools.is_typed({}, defs)
    assert not tools.is_typed({"$ref": "#/$defs/Map"}, defs)
    assert not tools.is_typed({"$ref": "#/$defs/EnvAny"}, defs)
    assert not tools.is_typed({"type": "array", "items": {}}, defs)
    assert not tools.is_typed({"anyOf": [{"$ref": "#/$defs/Obj"}, {}]}, defs)


def test_content_digest_ignores_revision_and_its_own_field():
    doc = {"a": 1, "revision": 3, "content_sha256": "x"}
    assert tools.content_digest(doc) == tools.content_digest({"a": 1})
    assert tools.content_digest(doc) != tools.content_digest({"a": 2})


def test_strict_form_closes_the_members_of_unions_and_tuples_but_not_of_allof():
    """044 R1-9: objects inside `anyOf`/`oneOf` members and `prefixItems` are closed too. R2-4: an
    `allOf` member is a view of the same object as its siblings, so it is left open (closing it would
    refuse the keys the other members declare); its own nested objects are closed."""
    node = {
        "anyOf": [{"type": "object", "properties": {"a": {}}}],
        "oneOf": [{"type": "object", "properties": {"b": {}}}],
        "allOf": [
            {"type": "object", "properties": {"c": {"type": "object", "properties": {"d": {}}}}}
        ],
        "prefixItems": [{"type": "object", "properties": {"e": {}}}],
    }
    strict = tools.strict_form(node)
    for key in ("anyOf", "oneOf", "prefixItems"):
        assert strict[key][0]["additionalProperties"] is False, key
    assert "additionalProperties" not in strict["allOf"][0]
    assert strict["allOf"][0]["properties"]["c"]["additionalProperties"] is False
