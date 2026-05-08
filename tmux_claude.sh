#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SESSION_NAME="claudecode"
WORKDIR="$(pwd)"

# ===== 默认环境变量，可按需修改 =====
export GITHUB_PERSONAL_ACCESS_TOKEN="${GITHUB_PERSONAL_ACCESS_TOKEN:-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}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-sk-f1a1f8afb0364acfae417beeb851c637}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_DEFAULT_SONNET_MODEL:-deepseek-v4-pro[1m]}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_DEFAULT_HAIKU_MODEL:-deepseek-v4-flash}"
export CLAUDE_CODE_SUBAGENT_MODEL="${CLAUDE_CODE_SUBAGENT_MODEL:-deepseek-v4-flash}"
export CLAUDE_CODE_EFFORT_LEVEL="${CLAUDE_CODE_EFFORT_LEVEL:-max}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:8118}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:8118}"

usage() {
  cat <<EOF
用法：

  $0
      启动默认会话：${DEFAULT_SESSION_NAME}

  $0 start [session_name]
      启动指定 tmux 会话，并在当前目录运行 claude

  $0 [session_name]
      兼容旧用法：直接把第一个参数当作会话名启动

  $0 list
      列出当前已有 tmux 会话

  $0 stop [session_name]
      关闭指定 tmux 会话
      如果不指定 session_name，则默认关闭：${DEFAULT_SESSION_NAME}

  $0 stop-all
      关闭所有 tmux 会话

  $0 help
      显示帮助
EOF
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "未检测到 tmux，请先安装："
    echo "sudo apt update && sudo apt install -y tmux"
    exit 1
  fi
}

require_claude() {
  if ! command -v claude >/dev/null 2>&1; then
    echo "未检测到 claude 命令，请确认 Claude Code 已安装并在 PATH 中。"
    exit 1
  fi
}

shell_quote() {
  printf "%q" "$1"
}

session_exists() {
  local name="$1"
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$name"
}

attach_or_switch() {
  local name="$1"

  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$name"
  else
    tmux attach-session -t "$name"
  fi
}

list_sessions() {
  if ! tmux list-sessions >/dev/null 2>&1; then
    echo "当前没有 tmux 会话。"
    return 0
  fi

  echo "当前 tmux 会话："
  tmux list-sessions -F '  - #{session_name} | windows=#{session_windows} | attached=#{session_attached}'
}

stop_session() {
  local name="${1:-$DEFAULT_SESSION_NAME}"

  if session_exists "$name"; then
    tmux kill-session -t "$name"
    echo "已关闭 tmux 会话：$name"
  else
    echo "未找到 tmux 会话：$name"
    exit 1
  fi
}

stop_all_sessions() {
  if ! tmux list-sessions >/dev/null 2>&1; then
    echo "当前没有 tmux 会话。"
    return 0
  fi

  echo "即将关闭以下所有 tmux 会话："
  tmux list-sessions -F '  - #{session_name}'

  tmux kill-server

  echo "已关闭所有 tmux 会话。"
}

start_session() {
  local name="${1:-$DEFAULT_SESSION_NAME}"

  if session_exists "$name"; then
    echo "tmux 会话已存在，正在进入：$name"
    attach_or_switch "$name"
    exit 0
  fi

  require_claude

  tmux new-session -d -s "$name" -c "$WORKDIR"
  tmux rename-window -t "$name:0" "ClaudeCode"

  # ===== 写入环境变量到 tmux 里的 shell =====
  tmux send-keys -t "$name:0" "export GITHUB_PERSONAL_ACCESS_TOKEN=$(shell_quote "$GITHUB_PERSONAL_ACCESS_TOKEN")" C-m
  tmux send-keys -t "$name:0" "export ANTHROPIC_BASE_URL=$(shell_quote "$ANTHROPIC_BASE_URL")" C-m
  tmux send-keys -t "$name:0" "export ANTHROPIC_AUTH_TOKEN=$(shell_quote "$ANTHROPIC_AUTH_TOKEN")" C-m
  tmux send-keys -t "$name:0" "export ANTHROPIC_MODEL=$(shell_quote "$ANTHROPIC_MODEL")" C-m
  tmux send-keys -t "$name:0" "export ANTHROPIC_DEFAULT_OPUS_MODEL=$(shell_quote "$ANTHROPIC_DEFAULT_OPUS_MODEL")" C-m
  tmux send-keys -t "$name:0" "export ANTHROPIC_DEFAULT_SONNET_MODEL=$(shell_quote "$ANTHROPIC_DEFAULT_SONNET_MODEL")" C-m
  tmux send-keys -t "$name:0" "export ANTHROPIC_DEFAULT_HAIKU_MODEL=$(shell_quote "$ANTHROPIC_DEFAULT_HAIKU_MODEL")" C-m
  tmux send-keys -t "$name:0" "export CLAUDE_CODE_SUBAGENT_MODEL=$(shell_quote "$CLAUDE_CODE_SUBAGENT_MODEL")" C-m
  tmux send-keys -t "$name:0" "export CLAUDE_CODE_EFFORT_LEVEL=$(shell_quote "$CLAUDE_CODE_EFFORT_LEVEL")" C-m
  tmux send-keys -t "$name:0" "export HTTP_PROXY=$(shell_quote "$HTTP_PROXY")" C-m
  tmux send-keys -t "$name:0" "export HTTPS_PROXY=$(shell_quote "$HTTPS_PROXY")" C-m

  # ===== 在当前目录打开 Claude Code =====
  tmux send-keys -t "$name:0" "conda activate abiu_mm" C-m
  tmux send-keys -t "$name:0" "cd $(shell_quote "$WORKDIR")" C-m
  tmux send-keys -t "$name:0" "claude" C-m

  echo "已创建 tmux 会话：$name"
  echo "工作目录：$WORKDIR"

  attach_or_switch "$name"
}

require_tmux

COMMAND="${1:-start}"

case "$COMMAND" in
  start|open|run)
    start_session "${2:-$DEFAULT_SESSION_NAME}"
    ;;

  list|ls)
    list_sessions
    ;;

  stop|kill|close)
    stop_session "${2:-$DEFAULT_SESSION_NAME}"
    ;;

  stop-all|kill-all|close-all)
    stop_all_sessions
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    # 兼容旧写法：./claudecode.sh mysession
    start_session "$COMMAND"
    ;;
esac