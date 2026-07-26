# Video Summary

給個人使用的本機影音逐字稿與重點摘要工具。介面只綁定
`127.0.0.1`，來源檔、逐字稿與摘要都存放在本機。

> 處理中的音訊片段與逐字稿會傳送至你在畫面選擇的 Gemini 或 OpenAI API。
> 本機保存不代表完全離線。

## 已完成的 MVP 功能

- 上傳音檔或影片，由內建 FFmpeg 自動抽取音軌。
- 可在同一個畫面切換 Gemini 或 OpenAI，不需修改程式。
- 長檔以 48 kbps 單聲道 MP3 分段，避免超過 API 單次上傳限制。
- Gemini 使用多模態模型完成含講者與時間碼的轉錄，再產生結構化摘要。
- OpenAI 支援 `gpt-4o-transcribe`、mini、講者辨識與 `whisper-1`，摘要使用 Responses API。
- 摘要格式：一般重點、會議紀錄、課程筆記。
- 輸出 Markdown、TXT、JSON、SRT。
- 選配 `yt-dlp`；只應處理自己擁有或已取得授權的內容。
- 完成後可自動刪除來源副本及音訊分段。

## Windows 快速開始

1. 安裝 Python 3.10 以上版本，建議 Python 3.12。
2. 雙擊 `setup.bat` 安裝環境。
3. 雙擊 `run.bat`。
4. 瀏覽器會開啟 <http://127.0.0.1:7860>。
5. 在頁面選擇 Gemini 或 OpenAI，輸入對應的 API Key；也可將金鑰寫入 `.env`。

頁面中輸入的 API Key 只存在本次 Python 程序的記憶體，不會由本程式寫入磁碟。

若專案是在 2026 年 6 月前安裝，請重新執行 `setup.bat`，將
`google-genai` 更新至 2.x；舊版 SDK 已無法呼叫新版 Interactions API。

## API Key 與費用

- Gemini：到 [Google AI Studio](https://aistudio.google.com/app/apikey) 的 API Keys 頁面，對可用專案按「Create API key」。畫面中的 Gemini 教育帳號或 Workspace 帳號不代表一定自動包含 API 額度，實際可用性以該專案的狀態、管理員政策及 Usage 頁面為準。
- OpenAI：ChatGPT Plus 與 OpenAI API 是兩套獨立服務；需到 [OpenAI API Keys](https://platform.openai.com/api-keys) 建立金鑰，並在 API 平台另外設定計費。
- 金鑰可每次貼在本機頁面，或複製 `.env.example` 為 `.env` 後填入 `GEMINI_API_KEY`／`OPENAI_API_KEY`。請勿提交 `.env`。
- 免費層與付費層的限制、資料使用方式及模型供應可能調整，正式處理私人內容前請先確認供應商當下政策。

## Gemini 模式

預設模型為 `gemini-3.6-flash`，同一模型會依序完成音訊轉錄與摘要。每段 MP3
會控制在 Gemini inline request 限制以下，並要求輸出講者、段落時間碼及繁體中文內容。
可在進階設定或 `.env` 的 `GEMINI_MODEL` 更換模型。

## OpenAI 模式與時間碼

- `gpt-4o-transcribe`：一般高品質轉錄。輸出只提供文字，因此時間碼是本機切片級的近似時間。
- `gpt-4o-mini-transcribe`：偏向成本與速度。
- `gpt-4o-transcribe-diarize`：提供講者與段落時間碼，適合會議。
- `whisper-1`：提供段落時間碼，適合輸出 SRT。

OpenAI 預設摘要模型為 `gpt-5.6-terra`，可在畫面的進階設定或 `.env` 修改。

## 本機資料

每次工作會建立：

```text
data/jobs/YYYYMMDD-HHMMSS-xxxxxxxx/
├─ source/
├─ chunks/
└─ output/
   ├─ transcript.md
   ├─ transcript.txt
   ├─ transcript.srt
   ├─ transcript.json
   ├─ summary.md
   └─ summary.json
```

若勾選「完成後刪除來源副本與音訊分段」，`source` 與 `chunks` 會在成功完成後刪除。

## YouTube 注意事項

YouTube 下載功能不是 YouTube 官方影音下載 API，可能因平台更新、權限或地區限制而失敗。
請只處理你擁有或已取得授權的內容。私人影片建議從 YouTube Studio 下載後再上傳本工具。

如需為自己擁有的私人影片提供 Cookie，可在 `.env` 設定
`YTDLP_COOKIES_FILE`，但請不要把 Cookie 檔放進專案或提交到版本控制。

## 開發驗證

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m compileall app.py video_summary
```
