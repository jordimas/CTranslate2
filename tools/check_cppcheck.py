"""Run from the repository root. Pass cppcheck preprocessor options on the command line."""

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BLOCKING = {
    "zerodiv", "zerodivcond",
    "nullPointer", "nullPointerRedundantCheck",
    "arrayIndexOutOfBounds", "bufferAccessOutOfBounds",
    "deallocuse", "doubleFree",
}
ROOTS = ("src", "include/ctranslate2", "cli", "python/cpp", "tests", "tools", "examples")


def main():
    command = [
        "cppcheck", "--enable=warning", "--std=c++17", "--xml", "--xml-version=2",
        "--inline-suppr", "-Iinclude", "-Isrc", *sys.argv[1:], *ROOTS,
    ]
    with tempfile.TemporaryFile() as report:
        subprocess.run(command, stderr=report, check=True)
        report.seek(0)
        errors = ET.parse(report).getroot().find("errors")
    if errors is None:
        raise ValueError("Cppcheck report is missing its errors section")

    failed = False
    for error in errors:
        if error.get("id") not in BLOCKING:
            continue
        location = error.find("location")
        path = Path(location.attrib["file"]).resolve()
        if any(path.is_relative_to(Path(root).resolve()) for root in ROOTS):
            print(f"{path.relative_to(Path.cwd())}:{location.get('line')}: "
                  f"{error.get('msg')} [{error.get('id')}]")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
