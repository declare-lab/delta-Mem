from __future__ import annotations

import json


def run_chat_loop(
    session,
    *,
    max_new_tokens: int,
    banner: str,
    allow_snapshot_commands: bool = False,
    dump_state_stats: bool = False,
) -> None:
    print(banner)
    commands = ["/exit", "/reset", "/stats"]
    if allow_snapshot_commands:
        commands.extend(["/save_session <dir>", "/load_session <dir>"])
    print("Commands: " + " ".join(commands))

    while True:
        user_text = input("user> ").strip()
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            session.reset()
            print("assistant> session reset")
            continue
        if user_text == "/stats":
            print(json.dumps({"state_stats": session.state_stats()}, ensure_ascii=False, indent=2))
            continue
        if allow_snapshot_commands and user_text.startswith("/save_session "):
            target = user_text.split(" ", 1)[1].strip()
            session.save_snapshot(target)
            print(f"assistant> saved session to {target}")
            continue
        if allow_snapshot_commands and user_text.startswith("/load_session "):
            target = user_text.split(" ", 1)[1].strip()
            session.load_snapshot_dir(target)
            print(f"assistant> loaded session from {target}")
            continue

        result = session.generate_reply(
            user_text=user_text,
            max_new_tokens=max_new_tokens,
        )
        print(f"assistant> {result.get('assistant_display', result['assistant'])}")
        if dump_state_stats:
            print(
                json.dumps(
                    {"state_stats": result["state_stats"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
