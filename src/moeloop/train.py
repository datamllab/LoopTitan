"""Native TorchTitan launcher with our Python recipe module as its default."""

from __future__ import annotations

import sys


def main() -> None:
    # Importing model/training helpers does not load this training environment.
    from .titan import require_torchtitan_pin
    require_torchtitan_pin()
    if not any(arg == "--module" or arg.startswith("--module=") for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--module", "moeloop.recipes"]
    from torchtitan.train import main as native_main
    native_main()


if __name__ == "__main__":
    main()
