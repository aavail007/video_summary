import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_summary import gemini_provider
from video_summary.models import TranscriptResult, TranscriptSegment


class FakeInteractions:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.responses.pop(0)))


class FakeClient:
    def __init__(self, interactions: FakeInteractions) -> None:
        self.interactions = interactions


def install_fake_client(monkeypatch, responses: list[dict]) -> FakeInteractions:
    interactions = FakeInteractions(responses)
    monkeypatch.setattr(
        gemini_provider.genai,
        "Client",
        lambda **kwargs: FakeClient(interactions),
    )
    return interactions


def test_gemini_transcription_offsets_chunks_and_sends_inline_audio(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first audio")
    second.write_bytes(b"second audio")
    monkeypatch.setattr(gemini_provider, "media_duration", lambda path: 10.0)
    interactions = install_fake_client(
        monkeypatch,
        [
            {
                "detected_language": "zh-TW",
                "segments": [
                    {
                        "start_seconds": 1,
                        "end_seconds": 3,
                        "text": "第一段",
                        "speaker": "講者 1",
                    }
                ],
            },
            {
                "detected_language": "zh-TW",
                "segments": [
                    {
                        "start_seconds": 2,
                        "end_seconds": 4,
                        "text": "第二段",
                        "speaker": "講者 2",
                    }
                ],
            },
        ],
    )

    result = gemini_provider.transcribe_chunks_gemini(
        [first, second],
        api_key="test-key",
        model="gemini-test",
        language_label="繁體中文",
        glossary="Codex",
        source_name="sample",
    )

    assert [segment.start for segment in result.segments] == [1.0, 12.0]
    assert [segment.end for segment in result.segments] == [3.0, 14.0]
    assert result.language == "zh-TW"
    assert len(interactions.calls) == 2
    first_call = interactions.calls[0]
    assert first_call["model"] == "gemini-test"
    assert first_call["response_format"]["mime_type"] == "application/json"
    assert first_call["input"][1] == {
        "type": "audio",
        "data": base64.b64encode(b"first audio").decode("ascii"),
        "mime_type": "audio/mp3",
    }


def test_gemini_summary_uses_structured_output(monkeypatch) -> None:
    interactions = install_fake_client(
        monkeypatch,
        [
            {
                "title": "測試摘要",
                "overview": "內容概覽",
                "key_points": ["重點一"],
                "chapters": [],
                "decisions": [],
                "action_items": [],
                "open_questions": [],
            }
        ],
    )
    transcript = TranscriptResult(
        source_name="sample",
        language="zh-TW",
        model="gemini-test",
        segments=[TranscriptSegment(start=0, end=2, text="測試內容")],
    )

    result = gemini_provider.summarize_transcript_gemini(
        transcript,
        api_key="test-key",
        model="gemini-test",
        style="一般重點摘要",
    )

    assert result.title == "測試摘要"
    call = interactions.calls[0]
    assert call["response_format"]["type"] == "text"
    assert "schema" in call["response_format"]
    assert "測試內容" in call["input"]


def test_gemini_transcription_rejects_oversized_inline_chunk(monkeypatch) -> None:
    chunk = SimpleNamespace(
        name="large.mp3",
        stat=lambda: SimpleNamespace(st_size=18 * 1024 * 1024),
    )
    install_fake_client(monkeypatch, [])

    with pytest.raises(RuntimeError, match="18 MB"):
        gemini_provider.transcribe_chunks_gemini(
            [chunk],
            api_key="test-key",
            model="gemini-test",
            language_label="自動偵測",
            glossary="",
            source_name="sample",
        )
