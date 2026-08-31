"""Small command-line entry point for temperature conversion."""

from __future__ import annotations

import sys

from temperature import celsius_to_fahrenheit


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: python cli.py <celsius>")
        return 2
    print(f"{celsius_to_fahrenheit(float(values[0])):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
