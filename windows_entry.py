"""Shared entry point for the packaged launcher and its console worker."""

import sys

from auto_ddg_cutline.windows_launcher import main

if __name__ == "__main__":
    sys.exit(main())
