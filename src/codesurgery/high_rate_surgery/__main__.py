"""Run the high-rate surgery command without changing existing CLI entry points."""

from .cli import main

raise SystemExit(main())
