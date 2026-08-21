import os
import stat

from rich.prompt import Prompt

from main.display import console, print_error, print_success
from main.paths import APP_DATA

ENV_PATH = APP_DATA / ".env"


def load_api_key() -> str:
    """Returns the OpenRouter API key, prompting to set it up if missing.
    Call this before any command that needs to hit OpenRouter."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key

    if ENV_PATH.exists():
        _load_env_file()
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            return key

    return ensure_api_key()


def ensure_api_key() -> str:
    """Interactively prompts for and saves the API key. Safe to call
    repeatedly -- does nothing if a key is already configured."""
    existing = os.getenv("OPENROUTER_API_KEY")
    if not existing and ENV_PATH.exists():
        _load_env_file()
        existing = os.getenv("OPENROUTER_API_KEY")

    if existing:
        return existing

    console.print(
        "\n[bold cyan]QuickSpec needs an OpenRouter API key[/bold cyan] "
        "to generate answers.\n"
        "Get one free at [underline]https://openrouter.ai/keys[/underline]\n"
    )

    key = Prompt.ask("[bold]Paste your API key[/bold]", password=True).strip()

    if not key:
        print_error("No key entered. Exiting.")
        raise SystemExit(1)

    _write_env_file(key)
    os.environ["OPENROUTER_API_KEY"] = key
    print_success(f"Saved to {ENV_PATH}")

    return key


def _write_env_file(key: str) -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)

    # write to a temp file first, then move into place -- avoids leaving a
    # corrupt/partial .env if something interrupts the write
    tmp_path = ENV_PATH.with_suffix(".tmp")
    tmp_path.write_text(f"OPENROUTER_API_KEY={key}\n")

    # restrict permissions to the owner only before the file is live,
    # since it holds a secret
    tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    tmp_path.replace(ENV_PATH)


def _load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
