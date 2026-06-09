"""CLI entry point for building code + docs indexes.

    python scripts/build.py            # both code + docs
    python scripts/build.py --code     # code only
    python scripts/build.py --docs     # docs only
    python scripts/build.py --resume   # resume from last checkpoint

The dashboard (livedocs/jobs.py) calls livedocs.build directly with a log
callback; this script is the command-line equivalent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from livedocs.build import run_build_code, run_build_docs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build code + docs Qdrant indexes.")
    parser.add_argument("--code",   action="store_true", help="Build code index only")
    parser.add_argument("--docs",   action="store_true", help="Build docs index only")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    run_code = args.code or (not args.code and not args.docs)
    run_docs = args.docs or (not args.code and not args.docs)

    if run_code:
        run_build_code(log=print, resume=args.resume)
    if run_docs:
        run_build_docs(log=print, resume=args.resume)

    print("\n=== Build complete. Start the server: uvicorn livedocs.query.app:app --port 8002 ===")


if __name__ == "__main__":
    main()
