from __future__ import annotations

import argparse
import os
from pathlib import Path


from autosteer import paths

DATA_SUBDIRS = paths.DATA_KINDS
PULL_SCOPES = ("setting", "module", "element")


def _data_dir() -> Path:
    return paths.data_dir()


def _ensure_data_dir(data_dir: Path) -> None:
    for subdir in DATA_SUBDIRS:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)


def run_dev(args: argparse.Namespace) -> None:
    data_dir = paths.set_data_dir(args.data_dir) if args.data_dir else _data_dir()
    _ensure_data_dir(data_dir)
    os.environ.setdefault("AUTOSTEER_WEB_MODE", "local")

    import uvicorn

    uvicorn.run(
        "autosteer.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_data_init(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else _data_dir()
    _ensure_data_dir(data_dir)
    print(f"Initialized autosteer data directory: {data_dir}")


def run_data_path(args: argparse.Namespace) -> None:
    print(_data_dir())


def run_data_pull(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else _data_dir()
    _ensure_data_dir(data_dir)
    target = f" '{args.name}'" if args.name else ""
    raise SystemExit(
        f"The template data{target} comes with the repository, in data/: run autoSTEER from a clone of "
        f"https://github.com/narunraman/autosteer (git clone, then `uv run autosteer dev`), or point "
        f"AUTOSTEER_DATA_DIR at a copy of that folder. (Requested data directory: {data_dir})"
    )


def _print_verify(report: dict) -> None:
    print(f"data directory: {report['data_dir']}")
    print(f"checked {report['checked']} files: {len(report['missing'])} missing, {len(report['changed'])} changed, "
          f"{len(report['extra'])} not in the list")
    for key in ("missing", "changed", "extra"):
        for p in report[key][:20]:
            print(f"  {key}: {p}")
        if len(report[key]) > 20:
            print(f"  ... {len(report[key]) - 20} more {key}")
    print("OK" if report["ok"] else "FAILED")


def run_data_convert(args) -> int:
    from autosteer import dataset_io
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else _data_dir()
    report = dataset_io.convert(data_dir, remove=not args.keep)
    print(f"converted {len(report['converted'])} datasets in {data_dir}; removed {len(report['removed'])} pickle files")
    for name, problems in report["failed"].items():
        print(f"NOT converted {name}: " + "; ".join(problems))
    return 1 if report["failed"] else 0


def run_data_verify(args: argparse.Namespace) -> None:
    from autosteer import data_files

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else _data_dir()
    report = data_files.verify(data_dir)
    _print_verify(report)
    if not report["ok"]:
        raise SystemExit(1)


def run_data_build(args: argparse.Namespace) -> None:
    from autosteer import data_files

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else _data_dir()
    pinned = data_files.load_list()["source"]
    print(f"extracting data/ of {pinned['repo']}@{args.commit or pinned['commit']} from {args.source} into {data_dir}")
    report = data_files.build(data_dir, Path(args.source).expanduser(), args.commit or None)
    _print_verify(report)
    if not report["ok"]:
        raise SystemExit(1)


def run_export(args: argparse.Namespace) -> None:
    from autosteer.release import export

    args.argv = [a for a in os.sys.argv[2:]]
    raise SystemExit(export.run(args))


def run_release(args: argparse.Namespace) -> None:
    args.argv = [a for a in os.sys.argv[2:]]
    raise SystemExit(args.release_func(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autosteer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dev = subparsers.add_parser("dev", help="Run the local authoring web app.")
    dev.add_argument("--host", default="127.0.0.1")
    dev.add_argument("--port", default=8000, type=int)
    dev.add_argument("--reload", action="store_true")
    dev.add_argument("--data-dir", default="")
    dev.set_defaults(func=run_dev)

    data = subparsers.add_parser("data", help="Manage the local autosteer data directory.")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)

    init = data_subparsers.add_parser("init", help="Create the local data directory layout.")
    init.add_argument("--data-dir", default="")
    init.set_defaults(func=run_data_init)

    path = data_subparsers.add_parser("path", help="Print the active data directory.")
    path.set_defaults(func=run_data_path)

    pull = data_subparsers.add_parser(
        "pull",
        help="Download optional seed/template data by setting, module, or element.",
    )
    pull.add_argument("scope", choices=PULL_SCOPES)
    pull.add_argument(
        "name",
        nargs="?",
        help="Optional setting/module/element name once public data packs are available.",
    )
    pull.add_argument("--data-dir", default="")
    pull.set_defaults(func=run_data_pull)

    verify = data_subparsers.add_parser(
        "verify",
        help="Check the data directory against the pinned file list (sizes and sha256).",
    )
    verify.add_argument("--data-dir", default="")
    verify.set_defaults(func=run_data_verify)

    convert = data_subparsers.add_parser(
        "convert",
        help="Rewrite the data directory's pickled datasets as Parquet (checked table by table), then remove the pickles.",
    )
    convert.add_argument("--data-dir", default="")
    convert.add_argument("--keep", action="store_true", help="keep the pickles next to the new files")
    convert.set_defaults(func=run_data_convert)

    build = data_subparsers.add_parser(
        "build",
        help="Recreate the template data from a STEER-generation checkout at the pinned commit, then verify.",
    )
    build.add_argument("--source", required=True, help="path to a STEER-generation git checkout")
    build.add_argument("--commit", default="", help="override the pinned commit (the list must then be rewritten)")
    build.add_argument("--data-dir", default="")
    build.set_defaults(func=run_data_build)

    from autosteer.release.export import add_arguments as add_export_arguments

    exp = subparsers.add_parser(
        "export",
        help="Export STEER and STEER-ME (Parquet per element and a manifest) to a local directory.",
    )
    add_export_arguments(exp)
    exp.set_defaults(func=run_export)

    from autosteer.release.package import add_arguments as add_release_arguments

    rel = subparsers.add_parser(
        "release",
        help="Build the Hugging Face dataset repos of STEER and STEER-ME locally (build), or check a build (check).",
    )
    add_release_arguments(rel)
    rel.set_defaults(func=run_release)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
