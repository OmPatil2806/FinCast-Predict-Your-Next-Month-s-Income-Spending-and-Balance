"""The README's images and internal links point at things that exist (step 9.6)."""

import re

from src.utils.helpers import get_project_root

ROOT = get_project_root()
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _anchor(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def test_images_exist():
    images = re.findall(r'<img src="([^"]+)"', README) + re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README)
    assert images, "the README should show screenshots"
    for image in images:
        assert (ROOT / image).exists(), image


def test_internal_links_resolve():
    anchors = {_anchor(h) for h in re.findall(r"^#{1,6} (.+)$", README, re.M)}
    for link in re.findall(r"\]\(#([^)]+)\)", README):
        assert link in anchors, link


def test_disclaimer_present():
    assert "not financial advice" in README
