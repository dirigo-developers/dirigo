from pathlib import Path

from dirigo import io


SYSTEM_CONFIG_TEXT = """\
# Dirigo system configuration
# Fill this in for your system.
"""

FRAME_ACQ_TEXT = """\
bidirectional_scanning = false
line_width = "1.0 mm"
frame_height = "1.0 mm"
pixel_size = "1.0 μm"
line_duty_cycle = 0.95
frames_per_acquisition = 10
"""

DIGITIZER_TEXT = """\
# Default digitizer settings
# Fill this in.
"""


def _write_if_missing(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
        return True
    else:
        return False


def main() -> int:
    base_dir = Path(io.config_path()).parent / "dirigo-test"

    system_config_path = base_dir / "system_config.toml"
    frame_acq_path = base_dir / "acquisition" / "frame" / "default.toml"
    digitizer_path = base_dir / "digitizer" / "default.toml"

    base_dir.mkdir(parents=True, exist_ok=True)   

    print()
    print("Dirigo settings folder (bookmark this folder):")
    print(f"  {base_dir}")
    print()
    
    if _write_if_missing(system_config_path, SYSTEM_CONFIG_TEXT):
        print("Initialized blank system_config.toml. Complete before starting Dirigo.")
        print(f"  {system_config_path}")
        print()

    if _write_if_missing(frame_acq_path, FRAME_ACQ_TEXT):
        print("Wrote default frame acquisition specification. Check before starting Dirigo")
        print(f"  {frame_acq_path}")
        print()

    if _write_if_missing(digitizer_path, DIGITIZER_TEXT):
        print("Initialized blank digitizer profile. Complete before starting Dirigo.")
        print(f"  {digitizer_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())