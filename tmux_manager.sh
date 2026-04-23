#!/usr/bin/env bash
# tmx - 更稳的 tmux 会话管理脚本
set -Eeuo pipefail

# 默认配置：也可以通过环境变量覆盖
TMUX_BIN="${TMUX_BIN:-tmux}"
TMUX_SOCKET="${TMUX_SOCKET:-/home/pythoner/abiu/tmux/tmp/mytmux.sock}"
SESSION_NAME="${SESSION_NAME:-abiu}"

log()  { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
用法:
  $(basename "$0") [选项] [命令] [参数]

选项:
  -s, --session NAME   指定会话名，默认: ${SESSION_NAME}
  -S, --socket  PATH   指定 socket 路径，默认: ${TMUX_SOCKET}
  -h, --help           显示帮助

命令:
  open | o            创建并进入会话（默认命令）
  start               只创建会话，不进入
  attach | a          进入会话，不存在则自动创建
  ls | l              列出当前 socket 下的所有会话
  status | st         查看当前会话状态
  run | r CMD...      向会话发送一条命令
  close | c           关闭当前指定会话
  kill-server | k     关闭当前 socket 下所有会话
  help                显示帮助

例子:
  $(basename "$0")
  $(basename "$0") status
  $(basename "$0") run "nvidia-smi"
  $(basename "$0") -s train_exp1
  $(basename "$0") -S /tmp/mytmux.sock
EOF
}

ensure_env() {
  command -v "$TMUX_BIN" >/dev/null 2>&1 || die "未找到 tmux，请先安装 tmux"
  mkdir -p "$(dirname "$TMUX_SOCKET")"
  chmod 700 "$(dirname "$TMUX_SOCKET")" || true
}

tmux_do() {
  "$TMUX_BIN" -S "$TMUX_SOCKET" "$@"
}

server_exists() {
  tmux_do ls >/dev/null 2>&1
}

session_exists() {
  tmux_do has-session -t "$SESSION_NAME" 2>/dev/null
}

create_session() {
  if session_exists; then
    log "会话已存在: $SESSION_NAME"
  else
    log "创建新会话: $SESSION_NAME"
    tmux_do new-session -d -s "$SESSION_NAME" -n main -c "$PWD"
  fi
}

attach_session() {
  create_session
  if [[ -n "${TMUX:-}" ]]; then
    warn "检测到你当前已经在 tmux 中，下面会进入嵌套会话。退出内层会话可按 Ctrl+b d"
    env TMUX= "$TMUX_BIN" -S "$TMUX_SOCKET" attach-session -t "$SESSION_NAME"
  else
    tmux_do attach-session -t "$SESSION_NAME"
  fi
}

list_sessions() {
  if server_exists; then
    tmux_do list-sessions
  else
    log "当前 socket 下没有运行中的 tmux server"
  fi
}

show_status() {
  echo "TMUX_BIN=$TMUX_BIN"
  echo "TMUX_SOCKET=$TMUX_SOCKET"
  echo "SESSION_NAME=$SESSION_NAME"

  if server_exists; then
    echo "SERVER=running"
    if session_exists; then
      echo "SESSION=exists"
      tmux_do list-windows -t "$SESSION_NAME"
    else
      echo "SESSION=missing"
    fi
  else
    echo "SERVER=stopped"
  fi
}

send_command() {
  [[ $# -gt 0 ]] || die "run 后面必须跟命令，例如: $0 run \"python train.py\""
  create_session
  local cmd="$*"
  tmux_do send-keys -t "$SESSION_NAME" "$cmd" C-m
  log "已发送到 [$SESSION_NAME]: $cmd"
}

close_session() {
  if session_exists; then
    log "关闭会话: $SESSION_NAME"
    tmux_do kill-session -t "$SESSION_NAME"
  else
    die "会话不存在: $SESSION_NAME"
  fi
}

kill_all_sessions() {
  if server_exists; then
    log "关闭当前 socket 下所有 tmux 会话"
    tmux_do kill-server
  else
    log "当前没有可关闭的 tmux server"
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -s|--session)
        [[ $# -ge 2 ]] || die "$1 需要一个参数"
        SESSION_NAME="$2"
        shift 2
        ;;
      -S|--socket)
        [[ $# -ge 2 ]] || die "$1 需要一个参数"
        TMUX_SOCKET="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        break
        ;;
    esac
  done

  CMD="${1:-open}"
  shift || true
  CMD_ARGS=("$@")
}

main() {
  parse_args "$@"
  ensure_env

  case "$CMD" in
    open|o|"")
      attach_session
      ;;
    start)
      create_session
      ;;
    attach|a)
      attach_session
      ;;
    ls|l)
      list_sessions
      ;;
    status|st)
      show_status
      ;;
    run|r)
      send_command "${CMD_ARGS[@]}"
      ;;
    close|c)
      close_session
      ;;
    kill-server|k)
      kill_all_sessions
      ;;
    help)
      usage
      ;;
    *)
      die "未知命令: $CMD。运行 '$0 --help' 查看用法"
      ;;
  esac
}

main "$@"
