"""Generate the pre-deadline brief as an HTML page."""

from __future__ import annotations

import argparse
from pathlib import Path

from fplopt.brief.assemble import assemble
from fplopt.brief.render import render
from fplopt.data.paths import project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=project_root() / "briefs" / "latest.html")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    brief = assemble(verbose=args.verbose)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(brief), encoding="utf-8")

    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    print(f"target GW{brief.target_gw}, deadline {brief.deadline}")
    print(f"risks {len(brief.risks)}, questions {len(brief.open_questions)}")
    print(f"pending panels: {', '.join(brief.pending) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
