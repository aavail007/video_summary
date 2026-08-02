#!/bin/bash

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

pause_before_exit() {
  printf "\n"
  read -r -p "按 Enter 關閉視窗..." _
}

fail() {
  printf "\n[ERROR] %s\n" "$1"
  pause_before_exit
  exit 1
}

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  fail "找不到 Python。請先安裝 Python 3.10 以上版本。"
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  fail "需要 Python 3.10 以上版本，建議安裝 Python 3.11 或 3.12。"
fi

printf "使用 Python：%s\n" "$PYTHON_BIN"
"$PYTHON_BIN" -m venv .venv || fail "無法建立 Python 虛擬環境。"

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  fail "虛擬環境建立完成，但找不到 .venv/bin/python。"
fi

"$VENV_PYTHON" -m pip install --upgrade pip || fail "無法更新 pip。"
"$VENV_PYTHON" -m pip install -r requirements.txt || fail "套件安裝失敗。"

if [ ! -f .env ]; then
  cp .env.example .env || fail "無法建立 .env。"
fi

chmod +x "$PROJECT_DIR/setup.command" "$PROJECT_DIR/run.command" 2>/dev/null || true

printf "\n安裝完成。之後可雙擊 run.command，或執行 ./run.command。\n"
pause_before_exit
