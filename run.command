#!/bin/bash

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  printf "尚未建立 Mac 執行環境，現在開始安裝。\n\n"
  bash "$PROJECT_DIR/setup.command"
fi

if [ ! -x "$VENV_PYTHON" ]; then
  printf "\n[ERROR] 找不到 .venv/bin/python，安裝可能尚未完成。\n"
  read -r -p "按 Enter 關閉視窗..." _
  exit 1
fi

"$VENV_PYTHON" app.py
STATUS=$?

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 130 ]; then
  printf "\n[ERROR] Video Summary 執行失敗，結束代碼：%s\n" "$STATUS"
  read -r -p "按 Enter 關閉視窗..." _
fi

exit "$STATUS"
