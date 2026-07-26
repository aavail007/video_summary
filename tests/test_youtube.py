import pytest

from video_summary.youtube import YouTubeDownloadError, _validate_youtube_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://music.youtube.com/watch?v=abc",
    ],
)
def test_youtube_url_validation_accepts_supported_hosts(url: str) -> None:
    _validate_youtube_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/private.mp4",
        "https://example.com/video",
        "https://youtube.com.example.com/watch?v=abc",
    ],
)
def test_youtube_url_validation_rejects_other_sources(url: str) -> None:
    with pytest.raises(YouTubeDownloadError):
        _validate_youtube_url(url)
