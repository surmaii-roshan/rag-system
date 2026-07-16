"""
app.py — CLI entry point for the RAG system.

Commands:
    ingest   Ingest documents from a directory
    query    (Phase 7) Ask a question
    evaluate (Phase 6) Run evaluation suite
    serve    (Phase 7) Launch Gradio web UI
"""

import argparse
import sys

from utils.logger import get_logger

log = get_logger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    from ingest import ingest_directory
    try:
        stats = ingest_directory(args.dir)
        print("\n✅ Ingestion complete!")
        print(f"   Files processed : {stats['processed']}")
        print(f"   Files skipped   : {stats['skipped']} (duplicates)")
        print(f"   Chunks created  : {stats['chunks_created']}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production RAG System (Free Tier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest documents from a directory")
    ingest_p.add_argument("--dir", required=True, help="Path to documents folder")

    # Placeholders — implemented in later phases
    subparsers.add_parser("query",    help="Ask a question (Phase 4)")
    subparsers.add_parser("evaluate", help="Run evaluation suite (Phase 6)")
    subparsers.add_parser("serve",    help="Launch Gradio UI (Phase 7)")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command in ("query", "evaluate", "serve"):
        print(f"'{args.command}' will be implemented in a later phase.")
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()