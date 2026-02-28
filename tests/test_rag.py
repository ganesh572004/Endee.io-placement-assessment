from endee_rag_assistant.rag import build_context


def test_build_context_limits_chunks() -> None:
    results = [
        {"payload": {"text": "alpha", "source": "s1"}},
        {"payload": {"text": "beta", "source": "s2"}},
        {"payload": {"text": "gamma", "source": "s3"}},
    ]

    context = build_context(results, max_chunks=2)

    assert "alpha" in context
    assert "beta" in context
    assert "gamma" not in context
    assert "s3" not in context
