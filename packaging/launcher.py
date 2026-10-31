from paranam.__main__ import main

if __name__ == "__main__":
    import sys

    try:
        code = main()
    except Exception as error:
        if "--self-test" not in sys.argv:
            raise
        import json
        import traceback
        from pathlib import Path

        result = Path(sys.argv[sys.argv.index("--self-test") + 1])
        result.write_text(
            json.dumps(
                {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
            ),
            encoding="utf-8",
        )
        code = 1
    raise SystemExit(code)
