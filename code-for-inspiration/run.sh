#!/usr/bin/env bash
# Start the Flask app and a cloudflared quick tunnel, print the public URL.
# Phone opens the https://*.trycloudflare.com link from any network.
set -u

PORT="${PORT:-5050}"
FLASK_LOG="${FLASK_LOG:-/tmp/flask.log}"
CF_LOG="${CF_LOG:-/tmp/cf.log}"

cd "$(dirname "$0")"

log() { echo "$@" >&2; }

start_flask() {
  if curl -sS -o /dev/null -w '' "http://127.0.0.1:${PORT}/" 2>/dev/null; then
    log "[flask] already running on :${PORT}"
    return
  fi
  log "[flask] starting on :${PORT} (logs: ${FLASK_LOG})"
  PORT="$PORT" nohup python app.py > "$FLASK_LOG" 2>&1 &
  for _ in $(seq 1 20); do
    sleep 0.5
    curl -sS -o /dev/null "http://127.0.0.1:${PORT}/" && return
  done
  log "[flask] failed to become ready; see ${FLASK_LOG}"
  tail -20 "$FLASK_LOG" >&2
  exit 1
}

# If a tunnel is already running and has a URL, reuse it.
existing_tunnel_url() {
  pgrep -f "cloudflared tunnel" >/dev/null || return 1
  [ -f "$CF_LOG" ] || return 1
  grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1
}

start_tunnel() {
  local url
  url="$(existing_tunnel_url || true)"
  if [ -n "${url:-}" ]; then
    log "[cloudflared] reusing existing tunnel"
    echo "$url"
    return
  fi

  # trycloudflare's quick-tunnel API occasionally 500s. Retry a few times.
  for attempt in 1 2 3 4; do
    log "[cloudflared] starting quick tunnel (attempt ${attempt}, logs: ${CF_LOG})"
    : > "$CF_LOG"
    nohup cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}" > "$CF_LOG" 2>&1 &
    local pid=$!
    for _ in $(seq 1 30); do
      sleep 1
      url=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1)
      [ -n "${url:-}" ] && { echo "$url"; return; }
      # Fast-fail if the well-known API-500 appears and the process has exited.
      if grep -q "failed to unmarshal quick Tunnel" "$CF_LOG" 2>/dev/null \
         && ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
    done
    kill "$pid" 2>/dev/null || true
    sleep 2
  done
  log "[cloudflared] no URL after retries; see ${CF_LOG}"
  tail -30 "$CF_LOG" >&2
  exit 1
}

start_flask
URL="$(start_tunnel)"
echo
echo "===================================================="
echo "  Open on your phone: ${URL}"
echo "===================================================="
