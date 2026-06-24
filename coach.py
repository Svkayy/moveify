from __future__ import annotations
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def build_fallback_review(results: dict) -> str:
    overall = results.get("overall_sync", 0.0)
    limbs = results.get("limb_performance", {}) or {}
    worst = min(limbs, key=limbs.get) if limbs else None
    best = max(limbs, key=limbs.get) if limbs else None
    lines = [f"Overall sync: {overall:.0f}%."]
    if worst:
        lines.append(f"Weakest limb: {worst} ({limbs[worst]:.0f}%) — focus your practice there.")
    if best:
        lines.append(f"Strongest limb: {best} ({limbs[best]:.0f}%) — keep it up.")
    return " ".join(lines)


def _gemini_complete(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    resp = model.generate_content(prompt)
    return resp.text


def generate_review(results: dict) -> dict:
    if not os.getenv("GEMINI_API_KEY"):
        return {"review": build_fallback_review(results), "source": "fallback"}
    prompt = (
        "You are a supportive dance coach. Based on these sync metrics, give a "
        "short (3-4 sentence) review with one specific tip. Metrics: "
        f"overall={results.get('overall_sync')}, per_limb={results.get('limb_performance')}."
    )
    try:
        return {"review": _gemini_complete(prompt).strip(), "source": "gemini"}
    except Exception:
        return {"review": build_fallback_review(results), "source": "fallback"}
