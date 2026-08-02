# Video Summary

給個人使用的本機影音逐字稿與重點摘要工具。介面只綁定
`127.0.0.1`，來源檔、逐字稿與摘要都存放在本機。

> 處理中的音訊片段與逐字稿會傳送至你在畫面選擇的 Gemini 或 OpenAI API。
> 本機保存不代表完全離線。

## 已完成的 MVP 功能

- 上傳音檔或影片，由內建 FFmpeg 自動抽取音軌。
- 可在同一個畫面切換 Gemini 或 OpenAI，不需修改程式。
- 上傳影片時可自動辨識換頁，只保留確認不同的完整投影片並搭配逐字稿理解。
- 偵測用的低解析度畫面只在記憶體中比較，不會每秒產生圖片檔。
- 長檔以 48 kbps 單聲道 MP3 分段，避免超過 API 單次上傳限制。
- Gemini 使用多模態模型完成含講者與時間碼的轉錄，再產生結構化摘要。
- OpenAI 支援 `gpt-4o-transcribe`、mini、講者辨識與 `whisper-1`，摘要使用 Responses API。
- 摘要格式：一般重點、會議紀錄、課程筆記。
- 輸出 Markdown、TXT、JSON。
- 選配 `yt-dlp`；只應處理自己擁有或已取得授權的內容。
- 完成後可自動刪除來源副本及音訊分段。

## Windows 快速開始

1. 安裝 Python 3.10 以上版本，建議 Python 3.12。
2. 雙擊 `setup.bat` 安裝環境。
3. 雙擊 `run.bat`。
4. 瀏覽器會開啟 <http://127.0.0.1:7860>。
5. 先選擇「上傳影片或音檔」或「YouTube 網址」，頁面只會顯示該來源需要的欄位。
6. 在同一區塊選擇 Gemini 或 OpenAI 並輸入對應的 API Key；也可將金鑰寫入 `.env`。

頁面中輸入的 API Key 只存在本次 Python 程序的記憶體，不會由本程式寫入磁碟。
進階設定中的「轉錄提示詞」可填入容易辨識錯誤的人名、公司名、產品名或專業術語，
幫助語音模型改善逐字稿拼寫；不需要時留空即可。

若專案是在 2026 年 6 月前安裝，請重新執行 `setup.bat`，將
`google-genai` 更新至 2.x；舊版 SDK 已無法呼叫新版 Interactions API。

## macOS 快速開始

需要 Python 3.10 以上版本，建議使用 Python 3.11 或 3.12。第一次安裝時，在
「終端機」進入專案目錄並執行：

```bash
bash setup.command
```

腳本會建立 `.venv`、安裝 `requirements.txt`、在尚未存在時複製 `.env.example`
為 `.env`，並將兩個 `.command` 腳本設為可執行。安裝完成後可直接雙擊
`run.command`，或在終端機執行：

```bash
./run.command
```

如果 macOS 阻擋第一次執行，可在終端機使用 `bash run.command`。若內建 FFmpeg
無法使用，可先透過 Homebrew 執行 `brew install ffmpeg`。啟動後瀏覽器會開啟
<http://127.0.0.1:7860>。

## API Key 與費用

- Gemini：到 [Google AI Studio](https://aistudio.google.com/app/apikey) 的 API Keys 頁面，對可用專案按「Create API key」。畫面中的 Gemini 教育帳號或 Workspace 帳號不代表一定自動包含 API 額度，實際可用性以該專案的狀態、管理員政策及 Usage 頁面為準。
- OpenAI：ChatGPT Plus 與 OpenAI API 是兩套獨立服務；需到 [OpenAI API Keys](https://platform.openai.com/api-keys) 建立金鑰，並在 API 平台另外設定計費。
- 金鑰可每次貼在本機頁面，或複製 `.env.example` 為 `.env` 後填入 `GEMINI_API_KEY`／`OPENAI_API_KEY`。請勿提交 `.env`。
- 免費層與付費層的限制、資料使用方式及模型供應可能調整，正式處理私人內容前請先確認供應商當下政策。

Gemini 模式預設會將 API 請求間隔控制為 5 秒；若免費層仍回傳暫時性
`429`，程式會依照 Google 指定的等待秒數自動重試，而不會立即中止整個工作。
可在 `.env` 用 `GEMINI_MIN_REQUEST_INTERVAL_SECONDS` 調整間隔。若碰到的是
每日額度而不是每分鐘限制，仍需等待配額重設或在 Google AI Studio 啟用計費。
若 `gemini-3.6-flash` 回傳 429，程式會先自動切換到
`gemini-3.5-flash-lite` 的模型配額；只有兩個模型都無法使用時才進入等待重試。

## Gemini 模式

預設模型為 `gemini-3.6-flash`，同一模型會依序完成音訊轉錄與摘要。每段 MP3
會控制在 Gemini inline request 限制以下，並要求輸出講者、段落時間碼及繁體中文內容。
可在進階設定或 `.env` 的 `GEMINI_MODEL` 更換模型。

## OpenAI 模式與時間碼

- `gpt-4o-transcribe`：一般高品質轉錄。輸出只提供文字，因此時間碼是本機切片級的近似時間。
- `gpt-4o-mini-transcribe`：偏向成本與速度。
- `gpt-4o-transcribe-diarize`：提供講者與段落時間碼，適合會議。
- `whisper-1`：提供段落時間碼，適合需要較精確時間定位的逐字稿。

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
   ├─ transcript.json
   ├─ summary.md
   ├─ summary.json
   ├─ slides.json
   └─ slides/
      ├─ slide_001_00-00-01.jpg
      └─ ...
```

若勾選「完成後刪除來源副本與音訊分段」，`source` 與 `chunks` 會在成功完成後刪除。
確認不同的投影片位於 `output/slides`，不會被這個選項刪除；它們也會列在匯出檔案中。

## 投影片辨識

介面的「投影片來源」提供三種模式：

- **不分析投影片，只分析音訊**（預設）：不擷取、不匯入也不理解任何投影片，摘要只根據
  逐字稿產生。
- **使用之前已擷取的投影片**：選擇舊的 `slides` 資料夾，或一次選擇多張
  JPG、PNG、WEBP 圖片。程式會沿用 `slide_001_00-12-35.jpg` 檔名中的時間碼，
  不會重新掃描影片或截圖。沒有時間碼的自訂圖片會依排序給予暫時順序時間。
- **自動從影片偵測並擷取**：適用於直接上傳的影片檔。程式以記憶體內的低解析度
  畫面判斷是否換頁、等待畫面穩定並排除重複頁面，然後才從原始影片擷取完整解析度
  JPEG；每秒掃描影格不會寫入硬碟。候選畫面還會分類為真正簡報頁、播放影片、
  講者鏡頭、桌面操作或轉場，只保留真正的 PPT 頁面。

確認的投影片會送至所選的 Gemini 或 OpenAI 視覺模型，辨識標題、重要文字、數字、
表格、圖表及流程，再與同一部影片的逐字稿一起產生摘要。YouTube 網址目前仍只下載
音軌；若需要投影片內容，可搭配「使用之前已擷取的投影片」，或先下載自己有權處理
的影片再上傳影片檔。

介面會先要求選擇「上傳影片或音檔」或「YouTube 網址」，只顯示該來源的輸入欄位。
選擇 YouTube 時會自動切換至「不分析投影片，只分析音訊」，並以鎖定標示阻擋
「自動從影片偵測並擷取」。上傳音檔同樣會鎖定自動擷取模式，只有上傳影片檔時才會
解除鎖定。URL 與音檔仍可改選「使用之前已擷取的投影片」。

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
