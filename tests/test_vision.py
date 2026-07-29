import json
from pathlib import Path
from types import SimpleNamespace

from video_summary import vision
from video_summary.slides import DetectedSlide


class FakeInteractions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "slides": [
                        {
                            "index": 1,
                            "is_presentation_slide": True,
                            "title": "架構圖",
                            "visible_text": ["輸入", "輸出"],
                            "visual_summary": "資料由輸入流向輸出。",
                        }
                    ]
                }
            )
        )


def test_gemini_slide_analysis_uses_inline_image_and_known_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "slide.jpg"
    image.write_bytes(b"jpeg")
    interactions = FakeInteractions()
    fake_client = SimpleNamespace(interactions=interactions)
    monkeypatch.setattr(
        vision.genai,
        "Client",
        lambda **kwargs: fake_client,
    )

    result = vision.analyze_slides(
        [DetectedSlide(index=1, timestamp=12.5, path=image)],
        provider="Gemini",
        api_key="test-key",
        gemini_model="gemini-test",
        openai_model="openai-test",
        reasoning_effort="low",
    )

    assert result[0].timestamp == 12.5
    assert result[0].image_file == "slides/slide.jpg"
    assert result[0].title == "架構圖"
    sent_image = interactions.calls[0]["input"][2]
    assert sent_image["type"] == "image"
    assert sent_image["mime_type"] == "image/jpeg"
