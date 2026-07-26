from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


class YouTubeDownloadError(RuntimeError):
    pass


def _validate_youtube_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed_hosts = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
    if parsed.scheme not in {"http", "https"} or host not in allowed_hosts:
        raise YouTubeDownloadError("請輸入有效的 YouTube 或 youtu.be 網址。")


def download_authorized_audio(
    url: str,
    target_dir: Path,
    cookies_file: str = "",
) -> tuple[Path, str]:
    _validate_youtube_url(url)
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise YouTubeDownloadError("尚未安裝 yt-dlp，請先執行 setup.bat。") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "format": "bestaudio/best",
        "outtmpl": str(target_dir / "%(title).120B-%(id)s.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_file:
        cookie_path = Path(cookies_file).expanduser()
        if not cookie_path.is_file():
            raise YouTubeDownloadError("YTDLP_COOKIES_FILE 指向的檔案不存在。")
        options["cookiefile"] = str(cookie_path)

    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            requested = info.get("requested_downloads") or []
            filepath = requested[0].get("filepath") if requested else None
            downloaded = Path(filepath or downloader.prepare_filename(info))
            title = str(info.get("title") or downloaded.stem)
    except DownloadError as exc:
        raise YouTubeDownloadError(
            "YouTube 影音取得失敗。私人影片建議從 YouTube Studio 下載後改用檔案上傳。"
        ) from exc

    if not downloaded.is_file():
        candidates = sorted(target_dir.glob("*"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise YouTubeDownloadError("下載程序完成，但找不到影音檔。")
        downloaded = candidates[-1]
    return downloaded, title
