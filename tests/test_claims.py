"""Tests for claim extraction."""

from evaluate.claims import extract_claims, _parse_json_claims


def test_extract_claims_transformer():
    answer = (
        "Multi-head attention allows the model to jointly attend to information "
        "from different representation subspaces. "
        "It projects queries, keys, and values using learned linear projections."
    )

    claims = extract_claims(answer)

    assert isinstance(claims, list)
    assert len(claims) >= 2


def test_extract_claims_empty():
    claims = extract_claims("")
    assert claims == []


def test_extract_claims_no_info_response():
    claims = extract_claims(
        "I don't have enough information in the provided documents to answer this question."
    )
    assert claims == []


def test_parse_json_claims_clean():
    raw = (
        '["Multi-head attention allows the model to jointly attend to '
        'information from different representation subspaces.", '
        '"The outputs of all attention heads are concatenated."]'
    )

    claims = _parse_json_claims(raw)

    assert len(claims) == 2
    assert (
        "Multi-head attention allows the model to jointly attend to information "
        "from different representation subspaces."
        in claims
    )


def test_parse_json_claims_with_markdown():
    raw = """```json
[
    "Retrieval-augmented generation combines retrieval and generation.",
    "Retrieved documents provide grounding."
]
```"""

    claims = _parse_json_claims(raw)

    assert len(claims) == 2
    assert (
        "Retrieval-augmented generation combines retrieval and generation."
        in claims
    )


def test_parse_json_claims_invalid():
    raw = "This is not JSON at all"

    claims = _parse_json_claims(raw)

    assert claims == []