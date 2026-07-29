from __future__ import annotations

import os
import traceback

import gradio as gr

from video_summary.config import settings
from video_summary.pipeline import (
    SLIDE_SOURCE_AUTO,
    SLIDE_SOURCE_EXISTING,
    SLIDE_SOURCE_OPTIONS,
    run_pipeline,
)


APP_CSS = """
.gradio-container { max-width: 1120px !important; }
.hero { padding: 1.2rem 0 .4rem; }
.hero h1 { margin-bottom: .25rem; }
.hint { color: var(--body-text-color-subdued); }
"""


def process(
    provider,
    uploaded_file,
    youtube_url,
    authorized_content,
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
        transcript, summary, files, status = run_pipeline(
            settings=settings,
            provider=provider,
            uploaded_path=uploaded_file,
            youtube_url=youtube_url,
            authorized_content=authorized_content,
            api_key_input=api_key,
            transcription_model=transcription_model,
            summary_model=summary_model.strip(),
            gemini_model=gemini_model.strip(),
            reasoning_effort=reasoning_effort,
            language_label=language,
            glossary=glossary,
            summary_style=summary_style,
            slide_source_mode=slide_source_mode,
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

        with gr.Row():
            with gr.Column(scale=3):
                provider = gr.Radio(
                    ["Gemini", "OpenAI"],
                    value="Gemini",
                    label="AI 服務",
                )
                uploaded_file = gr.File(
                    label="上傳音檔或影片",
                    type="filepath",
                    file_types=["audio", "video"],
                )
                youtube_url = gr.Textbox(
                    label="或貼上 YouTube 網址",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                authorized_content = gr.Checkbox(
                    label="我擁有此 YouTube 內容，或已取得下載與處理授權",
                    value=False,
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
                )
                existing_slide_images = gr.File(
                    label="或一次選擇多張既有投影片圖片",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    visible=False,
                )
            with gr.Column(scale=2):
                api_key = gr.Textbox(
                    label="API Key（依上方選擇 Gemini 或 OpenAI）",
                    type="password",
                    placeholder="只保留在本次程式記憶體中",
                )
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

        glossary = gr.Textbox(
            label="專有名詞（選填）",
            placeholder="每行一個或以逗號分隔，例如：產品名、人名、公司名",
            lines=2,
        )

        with gr.Accordion("進階設定", open=False):
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

        def toggle_existing_slide_inputs(mode: str):
            visible = mode == SLIDE_SOURCE_EXISTING
            return gr.update(visible=visible), gr.update(visible=visible)

        slide_source_mode.change(
            toggle_existing_slide_inputs,
            inputs=[slide_source_mode],
            outputs=[existing_slides_folder, existing_slide_images],
        )

        run_button.click(
            process,
            inputs=[
                provider,
                uploaded_file,
                youtube_url,
                authorized_content,
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
