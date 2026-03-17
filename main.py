"""
Skin Assistant — entry point.
  python main.py                    # sync from backend (leave existing data), then run API
  python main.py --no-sync-on-start # run API without syncing on startup
  python main.py sync               # sync from backend; existing CSV/images left as-is
  python main.py sync --overwrite   # sync and overwrite existing CSV with latest from backend
  python main.py chat               # run AI chatbot in console (interactive; same backend as API)
  python main.py chat "message"      # single message: print reply and exit (for testing)
  python main.py train              # train intent model
  python main.py train-products [--image]  # train product text (+ image) models
  python main.py train-skin-condition       # train skin condition classifier from images

  On serve, product data is synced from backend (backend.skinme.store); if CSV/images already exist they are left as-is.
  Console chat uses the same ChatService as the API (ingredients, product recommendations, optional LLM/DB).
  Optional MySQL (skinme_db) is used at runtime when "Check with database" is enabled (set MYSQL_* in .env).
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from console_style import (
        print_header,
        print_section,
        print_success,
        print_info,
        print_warning,
        print_error,
        print_key_value,
        print_line,
    )
except ImportError:
    def print_header(app_name="Skinme AI", tagline=""):
        print("\n  Skinme AI - Skin Assistant\n")
    def print_section(title, char="-"):
        print(f"  {title}\n  {char * 50}")
    def print_success(msg): print("  [OK]", msg)
    def print_info(msg): print("  (i)", msg)
    def print_warning(msg): print("  (!)", msg)
    def print_error(msg): print("  (x)", msg)
    def print_key_value(k, v, indent=2): print(" " * indent + f"{k}: {v}")
    def print_line(char="-", length=50): print("  " + char * length)


def run_api(port: int = 8000, host: str = "0.0.0.0") -> None:
    import uvicorn
    from skin_assistant.api.app import app
    print_section("API Server", "-")
    print_key_value("Host", host)
    print_key_value("Port", str(port))
    print_key_value("Docs", f"http://127.0.0.1:{port}/docs")
    print_line()
    uvicorn.run(app, host=host, port=port)


def run_train() -> int:
    from skin_assistant.models.trainer import train
    print_section("Intent model training", "-")
    metrics = train()
    print_success(f"Training done: {metrics}")
    return 0


def run_sync(
    no_download: bool = False,
    no_cleanup: bool = False,
    overwrite_existing: bool = False,
) -> int:
    from scripts.sync_products import do_sync
    return do_sync(
        no_download=no_download,
        no_cleanup=no_cleanup,
        no_sync=False,
        overwrite_existing=overwrite_existing,
    )


def run_train_products(image: bool = False, **kwargs) -> int:
    from skin_assistant.models.product_trainer import train_product_type_from_text, train_image_classifier
    from skin_assistant.config import get_settings
    print_section("Product models training", "-")
    label = kwargs.get("label_column", "productType")
    r1 = train_product_type_from_text(label_column=label)
    print_key_value("Text model", str(r1))
    if r1.get("error"):
        print_error(r1.get("error", "Training failed"))
        return 1
    if image:
        r2 = train_image_classifier(label_column=label, **{k: v for k, v in kwargs.items() if k in ("epochs", "batch_size")})
        print_key_value("Image model", str(r2))
    print_success("Product training complete")
    return 0


def run_train_skin_condition(**kwargs) -> int:
    from skin_assistant.models.skin_condition_trainer import train_skin_condition_classifier
    print_section("Skin condition classifier", "-")
    result = train_skin_condition_classifier(**kwargs)
    if result.get("error"):
        print_error(result.get("error", "Training failed"))
    else:
        print_key_value("Result", str(result))
        print_success("Skin condition model trained")
    return 1 if result.get("error") else 0


def run_chat_console(
    message: Optional[str] = None,
    use_llm: bool = True,
    use_database: bool = False,
) -> int:
    """Run the AI chatbot in the terminal. Same ChatService as the API. Chats are stored in DB when MySQL is configured."""
    import uuid
    from skin_assistant.services import ChatService
    from skin_assistant.infrastructure import ChatRepository

    chat = ChatService()
    chat_repo = ChatRepository()
    history: list[dict] = []
    session_id = str(uuid.uuid4())

    def save_turn_to_db(user_content: str, assistant_content: str) -> None:
        if chat_repo.is_available():
            try:
                chat_repo.ensure_session(session_id)
                chat_repo.save_message(session_id, "user", user_content)
                chat_repo.save_message(session_id, "assistant", assistant_content)
            except Exception:
                pass

    if message is not None:
        # Single message (test from CLI): print reply and exit
        print_section("Chat (single message)", "-")
        print_info(f"You: {message}")
        reply = chat.get_reply(message, conversation_history=history, use_llm=use_llm, use_database=use_database)
        save_turn_to_db(message, reply)
        print_success("Assistant:")
        print()
        print("  " + reply.replace("\n", "\n  "))
        print_line()
        return 0

    # Interactive console
    print_section("Chat console", "-")
    print_info("Same AI as API. Type your message and press Enter. Commands: quit, exit, or Ctrl+C to stop.")
    if use_database and chat_repo.is_available():
        print_info("Using database for product recommendations.")
    if chat_repo.is_available():
        print_info("Chat is saved to database (skinme_db).")
    print_info("Use LLM (GPT): " + ("yes" if use_llm else "no (retrieval only)"))
    print_line()

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print_success("Bye.")
            return 0
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print_success("Bye.")
            return 0
        reply = chat.get_reply(
            user_input,
            conversation_history=history,
            use_llm=use_llm,
            use_database=use_database,
        )
        save_turn_to_db(user_input, reply)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 20:
            history = history[-20:]
        print("  Assistant:")
        for line in reply.splitlines():
            print("    " + line)
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Skin Assistant")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "sync", "chat", "train", "train-products", "train-skin-condition"],
        help="serve | sync | chat | train | train-products | train-skin-condition",
    )
    parser.add_argument(
        "message",
        nargs="*",
        default=None,
        help="chat: optional single message (e.g. Recommend products for dry skin); omit for interactive console",
    )
    parser.add_argument(
        "--no-sync-on-start",
        action="store_true",
        help="serve: do not sync from backend on startup (default is to sync; existing CSV/images left as-is)",
    )
    parser.add_argument("--no-download", action="store_true", help="sync: skip image download")
    parser.add_argument("--no-cleanup", action="store_true", help="sync: skip deleting unused images")
    parser.add_argument("--overwrite", action="store_true", help="sync: overwrite existing CSV; default is to leave existing data")
    parser.add_argument("--no-llm", action="store_true", dest="chat_no_llm", help="chat: use retrieval only (no OpenAI)")
    parser.add_argument("--use-database", action="store_true", dest="chat_use_db", help="chat: use DB for product recommendations")
    parser.add_argument("--image", action="store_true", help="train-products: also train image model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print_header("Skinme AI", "Skin Assistant - Ingredients & Product Recommendations")
    print_section("Command", "-")
    print_key_value("Mode", "serve (API)" if args.command == "serve" else args.command)
    print_line()

    if args.command == "chat":
        msg = " ".join(args.message).strip() if args.message else None
        return run_chat_console(
            message=msg if msg else None,
            use_llm=not getattr(args, "chat_no_llm", False),
            use_database=getattr(args, "chat_use_db", False),
        )
    if args.command == "serve":
        if not args.no_sync_on_start:
            print_section("Sync from backend", "-")
            print_info("Syncing product data from backend (existing CSV/images left as-is)...")
            if run_sync(
                no_download=args.no_download,
                no_cleanup=args.no_cleanup,
                overwrite_existing=False,
            ) != 0:
                print_warning("Sync had errors; starting API anyway.")
            print_success("Sync finished")
            print_line()
        run_api(port=args.port, host=args.host)
        return 0
    if args.command == "sync":
        print_section("Sync products", "-")
        return run_sync(
            no_download=args.no_download,
            no_cleanup=args.no_cleanup,
            overwrite_existing=args.overwrite,
        )
    if args.command == "train":
        return run_train()
    if args.command == "train-products":
        return run_train_products(
            image=args.image,
            label_column="productType",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    if args.command == "train-skin-condition":
        return run_train_skin_condition(epochs=args.epochs, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
