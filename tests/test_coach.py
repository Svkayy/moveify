import coach

RESULTS = {"overall_sync": 72.5,
           "limb_performance": {"left_arm": 55.0, "right_arm": 88.0, "torso": 91.0}}


def test_fallback_review_is_deterministic_and_mentions_worst_limb():
    text = coach.build_fallback_review(RESULTS)
    assert "left_arm" in text          # the weakest limb is called out
    assert "72" in text or "73" in text  # overall score referenced


def test_generate_review_uses_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = coach.generate_review(RESULTS)
    assert out["source"] == "fallback"
    assert isinstance(out["review"], str) and out["review"]


def test_generate_review_uses_gemini_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(coach, "_gemini_complete",
                        lambda prompt: "Great work! Focus on your left arm timing.",
                        raising=False)
    out = coach.generate_review(RESULTS)
    assert out["source"] == "gemini"
    assert "left arm" in out["review"].lower()
