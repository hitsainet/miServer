"""RG-8 and RG-9 stay as they are (miServer 042 FR-042.18, FR-042.19; decisions P9-D8, P9-D9).

* Applied migrations are history that RUNS: Alembic replays them on every fresh database, and the rows
  they wrote are in every live one. P9 cleaned old names everywhere else and left these alone; this
  pins every file under the migrations' `versions/` folder, as it stood when P9 began (`8f1f4b4b`),
  so a later clean-up cannot reach them. A new migration is added to the pin in its own commit.
* The dated records under `0xcc/` are not rewritten; `0xcc/NAMES.md` maps their old names to today's.
  The public mirror strips `0xcc/` (`sync-to-clean.yml`), so there the note is absent and its test
  says so instead of failing.
"""

from __future__ import annotations

import hashlib

import pytest

from tests.support.package_root import package_root

ROOT = package_root().parent
VERSIONS = package_root() / "db" / "migrations" / "versions"
NAMES = ROOT / "0xcc" / "NAMES.md"

#: sha256 of each applied migration at the start of P9 (miServer `8f1f4b4b`, 2026-11-14).
APPLIED: dict[str, str] = {
    "001_create_models_table.py": "6009787431392173d0f42c37a00a12381c14ee7d7bc3a1f2d01a7c19e79b30ac",
    "002_create_sae_tables.py": "2a73d7ddcb0160f0ebb42aa5b615af48cf024f5bab0329cf952cf339d55f85ec",
    "003_create_profiles_table.py": "23610c8ed9508fcbdb27f1f03e2699e669cb2491d96b2ab8ccf9925feafff1fb",
    "004_add_sae_metadata_columns.py": "37f6632245bd32fd9fe373f2f12ac3bdc8ab3ba82b139b69c52bfb43f7f214e0",
    "005_add_fp32_q2_quantization.py": "193cda8d4799f64794717884b4bbf58b7a03b7a648bb6bb61527c8386d531623",
    "006_add_model_locked_column.py": "3b61f0c36f0f84af11a6bb3783687cab1e78986297a1a86bc103e4ab8ba78f09",
    "007_add_cluster_columns_to_profiles.py": "769fc6d97e7abc35ac364eaa22269390ddf6f60357c361b3b5ea422e9036ae71",
    "008_create_sensing_events_table.py": "74a8fa76f590314641dd774788c375b83dde60113d5b73a1575446ff1aac6b1d",
    "009_widen_profile_sae_id.py": "198111ce6efa763d6bccdd9d58ce454da40a8e1fb867741eea1d6f5b3d39d128",
    "010_add_sensing_context_parts.py": "a2c6d51c6f57d6e497a71a364e131c996c04ba9f01c106b489b147de8c26492c",
    "011_add_circuits_table.py": "e95b04bf5657172f82761524c91345310792189349ffa94cd06a170b53cfb791",
    "012_add_circuit_edge_sensing.py": "1a78e91534812a9947b73bdd1115d3a8107ae8a5c05ef4af22ff3ba9eef7d885",
    "013_add_circuit_layer_claims.py": "f7746d41e2537c63c536f686c557220492d51a981da4e0b068b3d735fbf884a9",
    "014_add_gguf_file_selection.py": "607852d38d5f3774c4b52e402a8e29f037c0b6bd7c61d8d18d6432800226d441",
    "015_tag_gguf_model_names.py": "a38fcad1b432cc4ef00ab7cf6ec88b59ec0da2ea9a127e995c80e307eedd2668",
    "016_add_probe_monitors.py": "e3e95f50c48687bce01fbebd125d870a03b0cba9aebe5be342cf545f9a9d7581",
    "017_add_probe_event_window.py": "baf6cdb5c7274d7b6d13907008cbfe99714429d3df8a5f3b8e250c9f598e56fc",
    "018_add_probe_threshold_revision.py": "d0605616b9d53094f29d46d3dc08adea9be80408897f1f79b3a4795e6f71a28f",
    "019_add_batch_api.py": "ba9a04e2ce1756d35ce1e358f293049d19f2f66e5dc77b5d26b86ffd33de78f0",
    "020_add_batch_origin_to_sensing.py": "82d0a89aa66757781aacbcdf26e7d788fba5af2634d8654e0f6b6f293809c8a0",
    "021_add_weights_origin_to_models.py": "c3c0c4652f6401b8a08b11350f9ce9051fb035ce0945d7f062d571d754c70950",
    "022_add_served_model_to_probe_events.py": "a48d0975a6ed627b7d666085f0e9fb930a86103ac8a1b753f2e98b40e493628b",
    "023_dedupe_batch_sensing_events.py": "67cd3f9bbd7575065ece0b736a1c01f86d5372f372dc19eb46147f319b05bcdd",
    "024_attachment_index_per_sae_layer.py": "072c49d0c3fc4cd3a84ee4266c9c8d94467daa1109d4541c11628e75d4b6eba9",
    "025_add_app_settings.py": "71a3691ee75abdc0c32eac2aff41a8c16127c602b387d5c05c8b8bad077e50f8",
    "026_app_settings_last_check_error.py": "84131b1f561601475362a59dd8aa15ead00470da78fec4b2cbeea952a10a4b26",
    "027_unique_model_names.py": "3de43124dab21da13347fea8dc961854037da0a61f6e7ca1c3b90c32f6a9b1b1",
    "028_env_overrides.py": "0e77183fe10ced829fb7644d2055a7cb6b759a349d58037e7e34e2b7b181f735",
    "029_probe_sessions.py": "eb479b74620ee067c7d9f10f8f6535e98e2f627ac67de93f5f6cd302f0da319d",
    "030_probe_cost.py": "b273dc341f1fdc8e93e04fda89337af4856948faa05635f1253f106711b916a2",
    "031_probe_capacity.py": "6cb73b5b3af59c872cfc956ca2765db02f1f16a03fa0ae4819743914b917d027",
    "032_probe_result_contract.py": "55c3f32ff884b470a142ce0cda988a7452533c48d2eae89666f21a98a7784eeb",
    "033_probe_identity_recompute.py": "97a466fe17e73cfb2bfc496a50b1a771bb59328b140ec97d75f2740e12b18fd9",
    "034_behaviour_sets.py": "6da24db33404a6d48045bf632d785c872a9c9f1a96295ee28cbebe958ca9fd2b",
    "035_rule_sets.py": "a807b26db6c1d618019fc1cd1b2e9353130e2d3655008b624459cbc0869733c3",
    "036_rule_code_modules.py": "45adaa60927bd61105ecf7b910e7477847dce62d32700726d23d926cefc41364",
    "037_approval_requests.py": "07ba74d5a37d7e0cee452661d37eb77379a9863a308cd5eaa1e7a49110bbc85a",
    "038_approval_telemetry_write.py": "a352110d3aff6f9d1ff829669426c801a6edb75dce83b0d2b8485a83467dd58b",
    "039_approval_batch_id.py": "f487ced23543ca276649e0bf926071ce8800f400df1d81c68da4864774dde691",
    "040_legacy_name_uses.py": "db5a6eea6c4e29cc8b10e3068161d1635f8075360bdb84c8b768236de42dbfb3",
    "041_behaviour_verdicts.py": "bd34811b22f89f31cdc5065a50941e1eb65dedf0b9100e7adc2cd4f3074cf339",
}


def test_applied_migrations_are_unchanged():
    on_disk = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(VERSIONS.glob("*.py"))}
    assert len(on_disk) >= len(APPLIED) > 30, f"read {len(on_disk)} migrations from {VERSIONS}"
    changed = sorted(n for n, h in APPLIED.items() if on_disk.get(n) != h)
    assert not changed, (f"applied migrations were edited or removed (P9-D8: they stay as they ran): "
                         f"{changed}")


def test_the_names_note_maps_the_old_names():
    if not (ROOT / "0xcc").is_dir():
        pytest.skip("0xcc/ is not in this checkout: the public mirror strips it (sync-to-clean.yml)")
    assert NAMES.is_file(), "0xcc/NAMES.md is missing (P9-D9: the records stay, the note maps them)"
    text = NAMES.read_text(encoding="utf-8")
    for old, new in (("miLLM", "miServer"), ("miStudio", "miWorkbench"), ("MiLLMError", "MiServerError")):
        assert old in text and new in text, (old, new)
    assert "docs/legacy-identifiers.md" in text, "the note points at the frozen names' authority"
