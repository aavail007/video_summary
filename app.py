from __future__ import annotations

import os
import traceback

import gradio as gr

from video_summary.config import settings
from video_summary.pipeline import (
    SLIDE_SOURCE_AUTO,
    SLIDE_SOURCE_EXISTING,
    SLIDE_SOURCE_NONE,
    SLIDE_SOURCE_OPTIONS,
    available_slide_source_options,
    normalize_slide_source_mode,
    run_pipeline,
)


APP_CSS = """
.gradio-container { max-width: 1120px !important; }
.hero { padding: 1.2rem 0 .4rem; }
.hero h1 { margin-bottom: .25rem; }
.hint { color: var(--body-text-color-subdued); }

#upload-source-shell {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.upload-card {
  border: 1px solid color-mix(in srgb, var(--primary-500) 32%, var(--border-color-primary)) !important;
  border-radius: 0 !important;
  background:
    radial-gradient(circle at 12% 12%, color-mix(in srgb, var(--primary-500) 11%, transparent), transparent 42%),
    linear-gradient(145deg, var(--block-background-fill), var(--background-fill-secondary)) !important;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07) !important;
  overflow: hidden;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.upload-card:hover,
.upload-card:focus-within {
  border-color: var(--primary-500) !important;
  box-shadow: 0 14px 34px color-mix(in srgb, var(--primary-500) 17%, transparent) !important;
  transform: translateY(-1px);
}

.upload-card svg {
  color: var(--primary-500) !important;
}

.dark .upload-card {
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24) !important;
}
"""

SOURCE_UPLOAD = "上傳影片或音檔"
SOURCE_YOUTUBE = "YouTube 網址"
SOURCE_OPTIONS = [SOURCE_UPLOAD, SOURCE_YOUTUBE]


def process(
    source_type,
    provider,
    uploaded_file,
    youtube_url,
    api_key,
    language,
    transcription_model,
    summary_model,
    gemini_model,
    reasoning_effort,
    summary_style,
    glossary,
    slide_source_mode,
    existing_slides_folder,
    existing_slide_images,
    delete_temp,
    progress=gr.Progress(),
):
    status_messages: list[str] = []

    def normalized_text(value, default: str = "") -> str:
        text = value if isinstance(value, str) else ""
        return text.strip() or default

    def file_paths(value) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def update_status(message: str) -> None:
        status_messages.append(message)
        progress((len(status_messages) % 10) / 10, desc=message)

    try:
        youtube_url = normalized_text(youtube_url)
        if source_type == SOURCE_UPLOAD:
            youtube_url = ""
        elif source_type == SOURCE_YOUTUBE:
            uploaded_file = None
        else:
            raise ValueError("不支援的影音來源。")

        transcript, summary, files, status = run_pipeline(
            settings=settings,
            provider=normalized_text(provider),
            uploaded_path=uploaded_file,
            youtube_url=youtube_url,
            api_key_input=normalized_text(api_key),
            transcription_model=normalized_text(
                transcription_model,
                settings.transcription_model,
            ),
            summary_model=normalized_text(summary_model, settings.summary_model),
            gemini_model=normalized_text(gemini_model, settings.gemini_model),
            reasoning_effort=normalized_text(
                reasoning_effort,
                settings.summary_reasoning_effort,
            ),
            language_label=normalized_text(language, "自動偵測"),
            glossary=normalized_text(glossary),
            summary_style=normalized_text(summary_style, "一般重點摘要"),
            slide_source_mode=normalized_text(
                slide_source_mode,
                SLIDE_SOURCE_NONE,
            ),
            existing_slide_paths=(
                file_paths(existing_slides_folder)
                + file_paths(existing_slide_images)
            ),
            delete_temp=delete_temp,
            status_callback=update_status,
        )
        return status, transcript, summary, files
    except Exception as exc:
        traceback.print_exc()
        return f"處理失敗：{exc}", "", "", []


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Video Summary") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>Video Summary</h1>
              <p class="hint">本機影音逐字稿與重點摘要工具。檔案保存在這台電腦；處理時，音訊片段與逐字稿會傳送至你選擇的 Gemini 或 OpenAI API。</p>
            </div>
            """
        )

        with gr.Group():
            gr.Markdown("### 1. 選擇影音來源")
            source_type = gr.Radio(
                SOURCE_OPTIONS,
                value=SOURCE_UPLOAD,
                label="影音來源",
            )
            with gr.Column(
                visible=True,
                elem_id="upload-source-shell",
            ) as upload_source_group:
                uploaded_file = gr.File(
                    label="上傳影片或音檔",
                    type="filepath",
                    file_types=["audio", "video"],
                    elem_id="media-upload",
                    elem_classes=["upload-card"],
                )
            with gr.Column(visible=False) as youtube_source_group:
                youtube_url = gr.Textbox(
                    label="貼上 YouTube 網址",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                gr.Markdown(
                    "⚠️ YouTube 網址只會下載音訊並產生逐字稿與摘要，"
                    "無法從影片擷取投影片截圖。若要搭配簡報內容，可選擇既有投影片，"
                    "或下載完整影片後改用上傳。"
                )
            slide_source_mode = gr.Radio(
                SLIDE_SOURCE_OPTIONS,
                value=SLIDE_SOURCE_AUTO,
                label="投影片來源",
            )
            existing_slides_folder = gr.File(
                label="選擇舊的 slides 資料夾",
                file_count="directory",
                file_types=["image"],
                type="filepath",
                visible=False,
                elem_classes=["upload-card"],
            )
            existing_slide_images = gr.File(
                label="或一次選擇多張既有投影片圖片",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                visible=False,
                elem_classes=["upload-card"],
            )

        with gr.Group():
            gr.Markdown("### 2. 選擇 AI 服務")
            with gr.Row():
                provider = gr.Radio(
                    ["Gemini", "OpenAI"],
                    value="Gemini",
                    label="AI 服務",
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="只保留在本次程式記憶體中",
                )
            with gr.Row():
                language = gr.Dropdown(
                    ["自動偵測", "繁體中文", "英文", "日文"],
                    value="自動偵測",
                    label="主要語言",
                )
                summary_style = gr.Dropdown(
                    ["一般重點摘要", "會議紀錄", "課程筆記"],
                    value="一般重點摘要",
                    label="摘要格式",
                )

        with gr.Accordion("進階設定", open=False):
            glossary = gr.Textbox(
                label="轉錄提示詞（選填）",
                placeholder="每行一個，例如：人名、公司名、產品名、專業術語",
                info="提供容易辨識錯誤的詞彙，幫助模型改善逐字稿拼寫；不需要時留空即可。",
                lines=2,
            )
            gr.Markdown("**Gemini 設定**（Gemini 會使用同一個多模態模型完成轉錄與摘要）")
            gemini_model = gr.Textbox(
                value=settings.gemini_model,
                label="Gemini 模型",
            )
            gr.Markdown("**OpenAI 設定**")
            transcription_model = gr.Dropdown(
                ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe-diarize", "whisper-1"],
                value=settings.transcription_model,
                label="轉錄模型",
            )
            summary_model = gr.Textbox(
                value=settings.summary_model,
                label="摘要模型",
            )
            reasoning_effort = gr.Dropdown(
                ["none", "low", "medium", "high"],
                value=settings.summary_reasoning_effort,
                label="摘要推理強度",
            )
            delete_temp = gr.Checkbox(
                label="完成後刪除來源副本與音訊分段（確認的投影片仍會保留）",
                value=True,
            )

        run_button = gr.Button("開始產生逐字稿與摘要", variant="primary", size="lg")
        status = gr.Textbox(label="狀態", interactive=False)

        with gr.Tabs():
            with gr.Tab("重點摘要"):
                summary_output = gr.Markdown()
            with gr.Tab("逐字稿"):
                transcript_output = gr.Markdown()
            with gr.Tab("匯出檔案"):
                files_output = gr.File(label="下載結果", file_count="multiple")

        def effective_slide_source_inputs(
            selected_source: str,
            upload_path,
            url: str,
        ):
            if selected_source == SOURCE_YOUTUBE:
                return None, (url or "https://youtube.invalid/")
            return upload_path, ""

        def displayed_slide_source_options(
            selected_source: str,
            upload_path,
            url: str,
        ):
            effective_upload, effective_url = effective_slide_source_inputs(
                selected_source,
                upload_path,
                url,
            )
            available = available_slide_source_options(
                effective_upload,
                effective_url,
            )
            if SLIDE_SOURCE_AUTO in available:
                return list(SLIDE_SOURCE_OPTIONS)
            return [
                (
                    f"🔒 {SLIDE_SOURCE_AUTO}（目前來源不可使用）",
                    SLIDE_SOURCE_AUTO,
                ),
                SLIDE_SOURCE_EXISTING,
                SLIDE_SOURCE_NONE,
            ]

        def source_control_updates(
            selected_source: str,
            upload_path,
            url: str,
            current_mode: str,
        ):
            effective_upload, effective_url = effective_slide_source_inputs(
                selected_source,
                upload_path,
                url,
            )
            mode = normalize_slide_source_mode(
                effective_upload,
                effective_url,
                current_mode,
            )
            visible = mode == SLIDE_SOURCE_EXISTING
            return (
                gr.update(
                    choices=displayed_slide_source_options(
                        selected_source,
                        upload_path,
                        url,
                    ),
                    value=mode,
                ),
                gr.update(visible=visible),
                gr.update(visible=visible),
            )

        def validate_slide_source_selection(
            selected_mode: str,
            selected_source: str,
            upload_path,
            url: str,
        ):
            effective_upload, effective_url = effective_slide_source_inputs(
                selected_source,
                upload_path,
                url,
            )
            mode = normalize_slide_source_mode(
                effective_upload,
                effective_url,
                selected_mode,
            )
            visible = mode == SLIDE_SOURCE_EXISTING
            return (
                gr.update(value=mode),
                gr.update(visible=visible),
                gr.update(visible=visible),
            )

        def change_source_type(
            selected_source: str,
            upload_path,
            url: str,
        ):
            default_mode = (
                SLIDE_SOURCE_AUTO
                if selected_source == SOURCE_UPLOAD
                else SLIDE_SOURCE_NONE
            )
            slide_updates = source_control_updates(
                selected_source,
                upload_path,
                url,
                default_mode,
            )
            return (
                gr.update(visible=selected_source == SOURCE_UPLOAD),
                gr.update(visible=selected_source == SOURCE_YOUTUBE),
                *slide_updates,
            )

        def refresh_slide_sources(
            upload_path,
            url: str,
            selected_source: str,
            current_mode: str,
        ):
            return source_control_updates(
                selected_source,
                upload_path,
                url,
                current_mode,
            )

        source_type.input(
            change_source_type,
            inputs=[source_type, uploaded_file, youtube_url],
            outputs=[
                upload_source_group,
                youtube_source_group,
                slide_source_mode,
                existing_slides_folder,
                existing_slide_images,
            ],
        )

        slide_source_mode.input(
            validate_slide_source_selection,
            inputs=[
                slide_source_mode,
                source_type,
                uploaded_file,
                youtube_url,
            ],
            outputs=[
                slide_source_mode,
                existing_slides_folder,
                existing_slide_images,
            ],
        )

        youtube_url.input(
            refresh_slide_sources,
            inputs=[
                uploaded_file,
                youtube_url,
                source_type,
                slide_source_mode,
            ],
            outputs=[
                slide_source_mode,
                existing_slides_folder,
                existing_slide_images,
            ],
        )

        uploaded_file.upload(
            refresh_slide_sources,
            inputs=[
                uploaded_file,
                youtube_url,
                source_type,
                slide_source_mode,
            ],
            outputs=[
                slide_source_mode,
                existing_slides_folder,
                existing_slide_images,
            ],
        )

        uploaded_file.clear(
            refresh_slide_sources,
            inputs=[
                uploaded_file,
                youtube_url,
                source_type,
                slide_source_mode,
            ],
            outputs=[
                slide_source_mode,
                existing_slides_folder,
                existing_slide_images,
            ],
        )

        run_button.click(
            process,
            inputs=[
                source_type,
                provider,
                uploaded_file,
                youtube_url,
                api_key,
                language,
                transcription_model,
                summary_model,
                gemini_model,
                reasoning_effort,
                summary_style,
                glossary,
                slide_source_mode,
                existing_slides_folder,
                existing_slide_images,
                delete_temp,
            ],
            outputs=[status, transcript_output, summary_output, files_output],
        )
    return demo


if __name__ == "__main__":
    application = build_app()
    application.queue(default_concurrency_limit=1)
    application.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=os.getenv("VIDEO_SUMMARY_NO_BROWSER") != "1",
        allowed_paths=[str(settings.data_dir)],
        css=APP_CSS,
    )
