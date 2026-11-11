"""Local-build feed overrides must not forward URL credentials to Docker."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("https://packages.example.invalid/simple/", True),
    ("https://registry.example.invalid", True),
    ("https://packages.example.invalid:443/simple/", True),
    ("http://packages.example.invalid/simple/", False),
    ("https://user:example-password@packages.example.invalid/simple/", False),
    ("https://packages.example.invalid/simple/?token=example", False),
    ("https://packages.example.invalid/#fragment", False),
    ("https://packages.example.invalid/with space", False),
    ("", False),
)


class PackageFeedTests(unittest.TestCase):
    def check_shell(self, shell: str) -> None:
        executable = shutil.which(shell)
        if not executable:
            self.skipTest(f"{shell} is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            if shell == "pwsh":
                source = (ROOT / "scripts/deploy.ps1").read_text(encoding="utf-8-sig")
                start = source.index("function Test-PublicPackageFeed {")
                end = source.index("\n}\n", start) + 3
                script = Path(directory) / "check.ps1"
                script.write_text(source[start:end] + "\nif(Test-PublicPackageFeed $env:TEST_FEED){exit 0}; exit 1\n", encoding="utf-8")
                command = [executable, "-NoProfile", "-File", str(script)]
            else:
                source = (ROOT / "scripts/deploy.sh").read_text(encoding="utf-8")
                start = source.index("validate_package_feed() {")
                end = source.index("\n}\n", start) + 3
                script = Path(directory) / "check.sh"
                script.write_text(source[start:end] + '\nvalidate_package_feed "$TEST_FEED"\n', encoding="utf-8", newline="\n")
                command = [executable, str(script)]
            for value, expected in CASES:
                with self.subTest(shell=shell, accepted=expected, case=CASES.index((value, expected))):
                    result = subprocess.run(command, env={**os.environ, "TEST_FEED": value}, capture_output=True, timeout=20)
                    self.assertEqual(expected, result.returncode == 0)
                    self.assertEqual(b"", result.stdout + result.stderr)

    def test_powershell(self) -> None:
        self.check_shell("pwsh")

    def test_bash(self) -> None:
        self.check_shell("bash")


if __name__ == "__main__":
    unittest.main()
