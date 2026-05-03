"""Tests for mempalace.entity_registry."""

import json
from unittest.mock import MagicMock, patch

from mempalace.entity_registry import (
    COMMON_ENGLISH_WORDS,
    PERSON_CONTEXT_PATTERNS,
    EntityRegistry,
)

# Shared mock result for Wikipedia person lookup tests
_MOCK_SAOIRSE_PERSON = {
    "inferred_type": "person",
    "confidence": 0.80,
    "wiki_summary": "Saoirse is an Irish given name.",
    "wiki_title": "Saoirse",
}


# ── COMMON_ENGLISH_WORDS ────────────────────────────────────────────────


def test_common_english_words_has_expected_entries():
    assert "ever" in COMMON_ENGLISH_WORDS
    assert "grace" in COMMON_ENGLISH_WORDS
    assert "will" in COMMON_ENGLISH_WORDS
    assert "may" in COMMON_ENGLISH_WORDS
    assert "monday" in COMMON_ENGLISH_WORDS


def test_common_english_words_is_lowercase():
    for word in COMMON_ENGLISH_WORDS:
        assert word == word.lower(), f"{word} should be lowercase"


# ── PERSON_CONTEXT_PATTERNS ─────────────────────────────────────────────


def test_person_context_patterns_is_nonempty():
    assert len(PERSON_CONTEXT_PATTERNS) > 0


# ── EntityRegistry creation and empty state ─────────────────────────────


def test_load_from_nonexistent_dir(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    assert registry.people == {}
    assert registry.projects == []
    assert registry.mode == "personal"
    assert registry.ambiguous_flags == []


def test_load_from_corrupt_file_falls_back_to_empty_state(tmp_path):
    config_file = tmp_path / "entity_registry.json"
    config_file.write_text("{ not json", encoding="utf-8")

    registry = EntityRegistry.load(config_dir=tmp_path)

    assert registry.people == {}
    assert registry.projects == []
    assert registry.mode == "personal"


def test_save_and_load_roundtrip(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="work",
        people=[{"name": "Alice", "relationship": "colleague", "context": "work"}],
        projects=["MemPalace"],
    )
    # Load again from same dir
    loaded = EntityRegistry.load(config_dir=tmp_path)
    assert loaded.mode == "work"
    assert "Alice" in loaded.people
    assert "MemPalace" in loaded.projects


def test_save_creates_file(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.save()
    assert (tmp_path / "entity_registry.json").exists()


# ── seed ────────────────────────────────────────────────────────────────


def test_seed_registers_people(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
            {"name": "Devon", "relationship": "friend", "context": "personal"},
        ],
        projects=["MemPalace"],
    )
    assert "Riley" in registry.people
    assert "Devon" in registry.people
    assert registry.people["Riley"]["relationship"] == "daughter"
    assert registry.people["Riley"]["source"] == "onboarding"
    assert registry.people["Riley"]["confidence"] == 1.0


def test_seed_registers_projects(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="work", people=[], projects=["Acme", "Widget"])
    assert registry.projects == ["Acme", "Widget"]


def test_seed_sets_mode(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="combo", people=[], projects=[])
    assert registry.mode == "combo"


def test_seed_flags_ambiguous_names(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Grace", "relationship": "friend", "context": "personal"},
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
        ],
        projects=[],
    )
    assert "grace" in registry.ambiguous_flags
    # Riley is not a common English word
    assert "riley" not in registry.ambiguous_flags


def test_seed_with_aliases(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Maxwell", "relationship": "friend", "context": "personal"}],
        projects=[],
        aliases={"Max": "Maxwell"},
    )
    assert "Maxwell" in registry.people
    assert "Max" in registry.people
    assert registry.people["Max"].get("canonical") == "Maxwell"


def test_seed_skips_empty_names(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "", "relationship": "", "context": "personal"}],
        projects=[],
    )
    assert len(registry.people) == 0


# ── lookup ──────────────────────────────────────────────────────────────


def test_lookup_known_person(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("Riley")
    assert result["type"] == "person"
    assert result["confidence"] == 1.0
    assert result["name"] == "Riley"


def test_lookup_known_project(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="work", people=[], projects=["MemPalace"])
    result = registry.lookup("MemPalace")
    assert result["type"] == "project"
    assert result["confidence"] == 1.0


def test_lookup_unknown_word(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])
    result = registry.lookup("Xyzzy")
    assert result["type"] == "unknown"
    assert result["confidence"] == 0.0


def test_lookup_case_insensitive(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("riley")
    assert result["type"] == "person"


def test_lookup_alias(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Maxwell", "relationship": "friend", "context": "personal"}],
        projects=[],
        aliases={"Max": "Maxwell"},
    )
    result = registry.lookup("Max")
    assert result["type"] == "person"


# ── disambiguation ──────────────────────────────────────────────────────


def test_lookup_ambiguous_word_as_person(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Grace", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("Grace", context="I went with Grace today")
    assert result["type"] == "person"


def test_lookup_ambiguous_word_as_concept(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Ever", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("Ever", context="have you ever tried this")
    assert result["type"] == "concept"


def test_lookup_ambiguous_word_tie_falls_back_to_person(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Ever", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("Ever", context="I said ever")
    assert result["type"] == "person"


# ── research — local-only by default ───────────────────────────────────


def test_research_local_only_by_default(tmp_path):
    """research() must NOT call Wikipedia unless allow_network=True."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        side_effect=AssertionError("network call should not happen"),
    ):
        result = registry.research("Saoirse")

    assert result["inferred_type"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["word"] == "Saoirse"
    assert "network lookup disabled" in result.get("note", "")


def test_research_with_allow_network(tmp_path):
    """research(allow_network=True) calls Wikipedia and caches result."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        return_value=dict(_MOCK_SAOIRSE_PERSON),
    ):
        result = registry.research("Saoirse", auto_confirm=True, allow_network=True)
    assert result["inferred_type"] == "person"


def test_research_caches_result(tmp_path):
    """Once cached via allow_network, subsequent calls use cache without network."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        return_value=dict(_MOCK_SAOIRSE_PERSON),
    ):
        result = registry.research("Saoirse", auto_confirm=True, allow_network=True)
    assert result["inferred_type"] == "person"

    # Second call should use cache, not call Wikipedia again
    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        side_effect=AssertionError("should not be called"),
    ):
        cached = registry.research("Saoirse")
    assert cached["inferred_type"] == "person"


def test_wikipedia_lookup_disambiguation_name(tmp_path):
    payload = {
        "type": "disambiguation",
        "description": "A name shared by several famous people",
        "extract": "Saoirse is a well known Irish name.",
        "title": "Saoirse",
    }

    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response

    registry = EntityRegistry.load(config_dir=tmp_path)
    with patch("mempalace.entity_registry.urllib.request.urlopen", return_value=response):
        result = registry.research("Saoirse", auto_confirm=True, allow_network=True)

    assert result["inferred_type"] == "person"
    assert result["confirmed"] is True
    assert result["confidence"] == 0.65


def test_wikipedia_lookup_place_and_concept(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)

    place_payload = {
        "type": "standard",
        "extract": "Saoirse is a city in a river valley.",
        "title": "Saoirse",
    }
    place_response = MagicMock()
    place_response.read.return_value = json.dumps(place_payload).encode()
    place_response.__enter__.return_value = place_response

    concept_payload = {
        "type": "standard",
        "extract": "A conceptual framework for language models.",
        "title": "Xylophone",
    }
    concept_response = MagicMock()
    concept_response.read.return_value = json.dumps(concept_payload).encode()
    concept_response.__enter__.return_value = concept_response

    with patch(
        "mempalace.entity_registry.urllib.request.urlopen",
        side_effect=[place_response, concept_response],
    ):
        place_result = registry.research("Saoirse", auto_confirm=True, allow_network=True)
        concept_result = registry.research("Xylophone", auto_confirm=True, allow_network=True)

    assert place_result["inferred_type"] == "place"
    assert concept_result["inferred_type"] == "concept"


def test_wikipedia_lookup_http_error_code(tmp_path):
    from urllib.error import HTTPError

    def _raise_http_error(*_args, **_kwargs):
        raise HTTPError("https://en.wikipedia.org", 404, "not found", None, None)

    registry = EntityRegistry.load(config_dir=tmp_path)
    with patch("mempalace.entity_registry.urllib.request.urlopen", side_effect=_raise_http_error):
        result = registry.research("MissingWord", auto_confirm=True, allow_network=True)

    assert result["inferred_type"] == "unknown"
    assert result["confidence"] == 0.3
    assert result.get("note") == "not found in Wikipedia"


def test_wikipedia_lookup_non_http_error_returns_unknown(tmp_path):
    from urllib.error import URLError

    registry = EntityRegistry.load(config_dir=tmp_path)
    with patch("mempalace.entity_registry.urllib.request.urlopen", side_effect=URLError("boom")):
        result = registry.research("NetworkDown", auto_confirm=True, allow_network=True)

    assert result["inferred_type"] == "unknown"
    assert result["confidence"] == 0.0


def test_research_local_only_not_cached(tmp_path):
    """Local-only result for uncached word should NOT be persisted to cache."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    registry.research("Xander")  # local-only, no network
    assert "Xander" not in registry._data.get("wiki_cache", {})


def test_confirm_research_adds_to_people(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        return_value=dict(_MOCK_SAOIRSE_PERSON),
    ):
        registry.research("Saoirse", auto_confirm=False, allow_network=True)

    registry.confirm_research("Saoirse", entity_type="person", relationship="friend")
    assert "Saoirse" in registry.people
    assert registry.people["Saoirse"]["source"] == "wiki"


def test_wikipedia_404_returns_unknown(tmp_path):
    """A 404 from Wikipedia should return 'unknown', not assert 'person'."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    mock_result = {
        "inferred_type": "unknown",
        "confidence": 0.3,
        "wiki_summary": None,
        "wiki_title": None,
        "note": "not found in Wikipedia",
    }
    with patch("mempalace.entity_registry._wikipedia_lookup", return_value=mock_result):
        result = registry.research("Zzxqy", auto_confirm=False, allow_network=True)

    assert result["inferred_type"] == "unknown"
    assert result["confidence"] < 0.5


# ── extract_people_from_query ───────────────────────────────────────────


def test_extract_people_from_query(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
            {"name": "Devon", "relationship": "friend", "context": "personal"},
        ],
        projects=[],
    )
    found = registry.extract_people_from_query("What did Riley say about the weather?")
    assert "Riley" in found
    assert "Devon" not in found


def test_extract_people_from_query_respects_disambiguation(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Ever", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    found = registry.extract_people_from_query("Ever said hello to us during planning")
    assert "Ever" in found


# ── extract_unknown_candidates ──────────────────────────────────────────


def test_extract_unknown_candidates(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])
    unknowns = registry.extract_unknown_candidates("Saoirse went to the store")
    assert "Saoirse" in unknowns


def test_extract_unknown_candidates_skips_known(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=[],
    )
    unknowns = registry.extract_unknown_candidates("Riley went to the store")
    assert "Riley" not in unknowns


# ── summary ─────────────────────────────────────────────────────────────


def test_summary(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=["MemPalace"],
    )
    s = registry.summary()
    assert "personal" in s
    assert "Riley" in s
    assert "MemPalace" in s


def test_confirm_research_marks_common_word_as_ambiguous(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)

    registry.confirm_research("May", entity_type="person")

    assert "may" in registry.ambiguous_flags


def test_learn_from_text_discovers_new_person(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)

    with (
        patch("mempalace.entity_detector.extract_candidates", return_value={"Riley": 2}),
        patch("mempalace.entity_detector.score_entity", return_value={"value": 0.9}),
        patch(
            "mempalace.entity_detector.classify_entity",
            return_value={"type": "person", "name": "Riley", "confidence": 0.9},
        ),
    ):
        new_candidates = registry.learn_from_text("Riley did a lot of work today.")

    assert any(candidate["name"] == "Riley" for candidate in new_candidates)
    assert "Riley" in registry.people
    assert registry.people["Riley"]["source"] == "learned"


def test_extract_unknown_candidates_skips_common_words(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    with patch("mempalace.palace._candidate_entity_words", return_value=["Ever", "Saoirse"]):
        unknown = registry.extract_unknown_candidates("Ever and Saoirse are names")

    assert "Ever" not in unknown
    assert "Saoirse" in unknown
