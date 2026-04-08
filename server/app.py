"""Compatibility entrypoint expected by the OpenEnv validator."""

from app import app as root_app
from app import main as root_main

app = root_app


def main() -> None:
    root_main()


if __name__ == "__main__":
    main()
