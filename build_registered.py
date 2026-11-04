#!/usr/bin/env python3
"""Run the existing MagicScan EDOF builders with Pearson registration plugged in (no edits to the builders).

  build_registered.py --builder fm    [--register pearson|builtin|off] -- <build_ms.py arguments>
  build_registered.py --builder color [--register pearson|off] [--gray_dir DIR] -- <recolor_MS_v2.4.py arguments>

--register pearson  this repo's registration (msprog/register.py) before EDOF fusion
           builtin  fm only: the builder's own ms_register.py (phase correlation), for comparison
           off      no registration (baseline rebuild)
Everything after `--` goes to the builder unchanged (fm: `--register` is set by this script).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msprog import launch  # noqa: E402


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        own, bargs = argv[:i], argv[i + 1:]
    else:
        own, bargs = argv, []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--builder", choices=["fm", "color"], required=True)
    ap.add_argument("--builder_path", default=None, help="builder script (default: the group's copy on olympus)")
    ap.add_argument("--register", choices=["pearson", "builtin", "off"], default="pearson")
    ap.add_argument("--reg_workers", type=int, default=None,
                    help="threads for registration (default: the builder's --workers for fm, 16 for color)")
    ap.add_argument("--gray_dir", default=None, help="color: also write fused grayscale 470/528 + z-map here")
    ap.add_argument("--allow_protected_dataset", action="store_true",
                    help="fm: allow writing into the existing dataset (default: refuse)")
    a = ap.parse_args(own)
    if a.builder == "color" and a.register == "builtin":
        ap.error("the colour builder has no built-in registration; use pearson or off")
    if not bargs:
        ap.error("pass the builder's own arguments after `--`")
    return launch.run_fm(a, bargs) if a.builder == "fm" else launch.run_color(a, bargs)


if __name__ == "__main__":
    sys.exit(main())
