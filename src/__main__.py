import sys
from src.cli import CLI

from src.domain.exceptions import RAGError

try:
    import fire
except ImportError as exc:
    missing_lib = exc.name
    print(f"❌ [ERROR] Missing dependency: '{missing_lib}'")
    print("Please install all required dependencies before running the system")
    print("Run 'pip install -e .' or 'make'")
    sys.exit(1)


def main() -> None:
    """Run the command-line interface and handle application errors.
    The CLI is executed through Google Fire. Expected value and application
    errors are reported to the user and terminate the process with a non-zero
    exit status. Unexpected exceptions are also caught to prevent raw
    tracebacks from being exposed to CLI users.

    Raises:
        SystemExit: If a dependency is missing or an application error occurs.
    """
    try:
        fire.Fire(CLI)
    except ValueError as exc:
        print(f"❌ [VALUE ERROR] {exc}")
        sys.exit(1)
    except RAGError as exc:
        print(f"❌ [RAG ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ [UNEXPECTED ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
