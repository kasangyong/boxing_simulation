"""
Boxing RL 텔레그램 봇
- 텔레그램 메시지를 받아 Claude Code CLI에 전달
- 결과를 텔레그램으로 전송
- 실행: py -3.11 telegram_bot.py
"""
import requests
import subprocess
import time

BOT_TOKEN = "8616828918:AAGfjlqolCefRNtwiFrxKB0mR7B4JVikByE"
API_URL   = f"https://api.telegram.org/bot{BOT_TOKEN}"
WORK_DIR  = r"c:\Users\SSAFY\Desktop\boxing"

# 첫 메시지 보낸 사람만 허용 (본인 전용 봇)
ALLOWED_IDS: set[int] = set()


def get_updates(offset=None):
    try:
        r = requests.get(f"{API_URL}/getUpdates",
                         params={"timeout": 30, "offset": offset},
                         timeout=35)
        return r.json()
    except Exception:
        return {"result": []}


def send_message(chat_id: int, text: str):
    for i in range(0, max(len(text), 1), 4096):
        requests.post(f"{API_URL}/sendMessage",
                      json={"chat_id": chat_id, "text": text[i:i+4096]},
                      timeout=10)


def send_typing(chat_id: int):
    requests.post(f"{API_URL}/sendChatAction",
                  json={"chat_id": chat_id, "action": "typing"},
                  timeout=5)


def run_claude(message: str) -> str:
    try:
        result = subprocess.run(
            ["claude", "-p", message, "--dangerously-skip-permissions"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=WORK_DIR,
            timeout=300,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if not out and err:
            return f"[오류]\n{err}"
        return out or "(빈 응답)"
    except subprocess.TimeoutExpired:
        return "[타임아웃] 5분 초과."
    except FileNotFoundError:
        return "[오류] claude 명령어를 찾을 수 없습니다."
    except Exception as e:
        return f"[예외] {e}"


def main():
    print("Boxing RL Telegram Bot 시작")
    print(f"봇 주소: t.me/box_simulation_bot")
    print("Ctrl+C 로 종료\n")

    offset = None
    while True:
        try:
            data = get_updates(offset)
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("edited_message") or {}
                if not msg:
                    continue

                chat_id = msg["chat"]["id"]
                text    = msg.get("text", "").strip()
                if not text:
                    continue

                # 첫 사용자 자동 등록
                if not ALLOWED_IDS:
                    ALLOWED_IDS.add(chat_id)
                    print(f"[인증] chat_id={chat_id} 등록")

                if chat_id not in ALLOWED_IDS:
                    send_message(chat_id, "⛔ 권한 없음")
                    continue

                print(f"[수신] {text[:80]}")
                send_typing(chat_id)
                send_message(chat_id, "⏳ Claude 처리 중...")

                response = run_claude(text)
                send_message(chat_id, response)
                print(f"[전송] {len(response)}자 완료\n")

        except KeyboardInterrupt:
            print("\n봇 종료.")
            break
        except Exception as e:
            print(f"[오류] {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
