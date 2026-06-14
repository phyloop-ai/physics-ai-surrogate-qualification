#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import copy_config_to_pack, ensure_evidence_folders, load_config, study_paths


def main() -> None:
    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)
    copy_config_to_pack(paths)
    print(f"Evidence pack ready: {paths.evidence_pack}")


if __name__ == "__main__":
    main()
