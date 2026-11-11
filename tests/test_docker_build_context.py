"""Sensitive local state must not enter the component's COPY-all context."""

from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ".env*", "**/.env*", ".local/", "**/.local/", "local/", "**/local/",
    "local.settings.json", "**/local.settings.json", ".files/", "**/.files/",
    ".logs/", "**/.logs/", ".cache/", "**/.cache/",
}


class DockerContextTests(unittest.TestCase):
    def test_local_state_is_explicitly_excluded(self) -> None:
        rules = {
            line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertTrue(REQUIRED <= rules, f"Missing local-state exclusions: {sorted(REQUIRED - rules)}")
        self.assertFalse(any(line.startswith("!") for line in rules), "Re-inclusion requires explicit context review")

    def test_copy_all_excludes_synthetic_local_inputs(self) -> None:
        docker = shutil.which("docker")
        if docker is None:
            self.skipTest("Docker is not installed")
        try:
            ready = subprocess.run(
                [docker, "version", "--format", "{{.Server.Version}}"],
                capture_output=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("Docker engine readiness timed out")
        if ready.returncode:
            self.skipTest("Docker engine is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            context = base / "context"
            context.mkdir()
            shutil.copyfile(ROOT / ".dockerignore", context / ".dockerignore")
            (context / "Dockerfile").write_text("FROM scratch\nCOPY . /app/\n", encoding="utf-8")
            (context / "application.py").write_text("# public synthetic source\n", encoding="utf-8")
            private = [
                ".env", ".env.production", "nested/.env", "nested/.env.production",
                ".local/state.json", "nested/.local/state.json", "local/state.json",
                "local.settings.json", "nested/local.settings.json", ".files/upload.txt",
                ".logs/deploy.log", ".cache/token",
            ]
            for name in private:
                path = context / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic-local-input-not-a-secret", encoding="utf-8")
            archive = base / "context.tar"
            result = subprocess.run(
                [docker, "build", "--network", "none", "--progress", "plain",
                 "--output", f"type=tar,dest={archive}", str(context)],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            with tarfile.open(archive) as exported:
                files = {item.name.lstrip("./") for item in exported if item.isfile()}
            self.assertIn("app/application.py", files)
            self.assertFalse(files & {"app/" + name for name in private})


if __name__ == "__main__":
    unittest.main()
