"""PRD AC-7 / QA: application modules must not use prohibited crawl/parse stacks."""

import unittest
from pathlib import Path


class TestNoBannedCrawlLibraries(unittest.TestCase):
    def test_core_services_app_exclude_banned_imports(self):
        root = Path(__file__).resolve().parent.parent
        banned = ("scrapy", "beautifulsoup", "bs4")
        paths: list[Path] = []
        for name in ("core", "services"):
            d = root / name
            if d.is_dir():
                paths.extend(d.rglob("*.py"))
        app_py = root / "app.py"
        if app_py.is_file():
            paths.append(app_py)

        offenders: list[str] = []
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            for b in banned:
                if b in text:
                    offenders.append(f"{path.relative_to(root)}: mentions {b}")

        self.assertEqual(
            offenders,
            [],
            "Disallowed high-level crawl/parse libraries referenced in sources:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
