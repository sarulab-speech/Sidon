"""Bookkeeping of the producer tasks (common/shards_qc.py does the checks).

  python tasks.py expected --source SOURCE > expected.json        windows per task t00..t15 and the dev windows
  python tasks.py check --shards OUT/WRITER --task 3 --expected expected.json [--log t03.log] [--max-shortfall 0]
                                                                  one finished task; writes _done/t03.json if it passes
  python tasks.py finalize --shards OUT/WRITER --source SOURCE [--max-shortfall 0]
                                                                  all tasks; read-only shards and N_SHARDS.txt
Without a manifest (the base root, whose conversations the producer cuts itself) leave out --expected and --source:
then only the members, duplicate keys and task logs are checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import core as cm  # noqa: E402
from common import shards_qc as sq  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("expected")
    e.add_argument("--source", type=Path, required=True)
    e.add_argument("--tasks", type=int, default=sq.TASKS)
    c = sub.add_parser("check")
    c.add_argument("--shards", type=Path, required=True, help="<output_root>/<writer_name>")
    c.add_argument("--task", type=int, required=True)
    c.add_argument("--expected", type=Path, default=None)
    c.add_argument("--log", type=Path, default=None)
    c.add_argument("--max-shortfall", type=int, default=0)
    f = sub.add_parser("finalize")
    f.add_argument("--shards", type=Path, required=True)
    f.add_argument("--source", type=Path, default=None)
    f.add_argument("--tasks", type=int, default=sq.TASKS)
    f.add_argument("--max-shortfall", type=int, default=0)
    f.add_argument("--count-file", default="N_SHARDS.txt")
    args = parser.parse_args(argv)
    if args.command == "expected":
        print(json.dumps(sq.expected_per_task(args.source, args.tasks)))
        return 0
    if args.command == "check":
        name = f"t{args.task:02d}"
        expected = cm.load_json(args.expected) if args.expected is not None else None
        record = sq.task_check(args.shards, name, args.task, expected, args.log, args.max_shortfall)
        summary = {k: record[k] for k in ("records", "expected", "valid_records", "shards", "log_errors", "passed")}
        print(f"{name}: {json.dumps(summary)}", flush=True)
        return 0 if record["passed"] else 1
    return sq.finalize(args.shards, args.source, args.tasks, args.count_file, args.max_shortfall)


if __name__ == "__main__":
    sys.exit(main())
