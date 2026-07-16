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


def cmd_query(args: argparse.Namespace) -> None:
    from generate import answer_question
    from retrieve.cache import clear_cache as _clear

    if args.no_cache:
        _clear()

    print(f"\n🔍 Query: {args.question}\n")
    result = answer_question(args.question)

    print("─" * 60)
    print(result.answer)
    print("─" * 60)

    if result.from_cache:
        print("📦 Served from cache")
    else:
        if result.sources:
            src_str = ", ".join(f"{f} p.{p}" for f, p in result.sources)
            print(f"📄 Sources: {src_str}")
        print(f"🤖 Model: {result.model_used}")
        print(f"🎯 Confidence: {result.confidence} (faithfulness={result.faithfulness:.2f})")
        print(f"⏱  Latency: {result.latency_ms.get('total_ms', '?')}ms")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate.runner import run_evaluation, save_baseline, compare_to_baseline
    print("\n🧪 RAG Evaluation Suite")
    print("=" * 60)
    results = run_evaluation(
        save_results=True,
        question_ids=args.ids if hasattr(args, "ids") else None,
    )
    agg = results["aggregate"]
    print("\n📊 Final Aggregate Metrics:")
    print("─" * 40)
    print(f"  Context Precision : {agg['context_precision']:.4f}")
    print(f"  Context Recall    : {agg['context_recall']:.4f}")
    print(f"  Faithfulness      : {agg['faithfulness']:.4f}")
    print(f"  Answer Relevancy  : {agg['answer_relevancy']:.4f}")
    if agg.get("source_accuracy") is not None:
        print(f"  Source Accuracy   : {agg['source_accuracy']:.4f}")
    print("─" * 40)
    if hasattr(args, "save_baseline") and args.save_baseline:
        save_baseline(results)
        print("✅ Baseline saved!")
    if hasattr(args, "compare") and args.compare:
        compare_to_baseline(results)


def cmd_serve(args: argparse.Namespace) -> None:
    from ui.app_ui import launch
    port = getattr(args, "port", 7860)
    share = getattr(args, "share", False)
    print(f"\n🚀 Starting RAG Web UI at http://localhost:{port}")
    print("   Press Ctrl+C to stop.\n")
    launch(share=share, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production RAG System (Free Tier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest documents from a directory")
    ingest_p.add_argument("--dir", required=True, help="Path to documents folder")

    # query
    query_p = subparsers.add_parser("query", help="Ask a question")
    query_p.add_argument("question", help="Your question")
    query_p.add_argument("--no-cache", action="store_true", help="Bypass semantic cache")
    
    # evaluate (Updated for Phase 6)
    eval_p = subparsers.add_parser("evaluate", help="Run evaluation suite")
    eval_p.add_argument("--save-baseline", action="store_true",
                       help="Save results as the V1 baseline")
    eval_p.add_argument("--compare", action="store_true",
                       help="Compare current run to baseline")
    eval_p.add_argument("--ids", nargs="+", metavar="ID",
                       help="Run specific question IDs only (e.g. q01 q02)")

    # serve (Updated for Phase 7)
    serve_p = subparsers.add_parser("serve", help="Launch Gradio web UI")
    serve_p.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    serve_p.add_argument("--share", action="store_true", help="Create public Gradio share link")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()