"""Unit tests for core.parser.parse_html — link/text extraction and edge cases."""

import unittest

from core.parser import parse_html


class TestParseHtmlLinks(unittest.TestCase):
    def test_absolute_http_link_preserved(self):
        html = '<html><body><a href="https://example.com/page">x</a></body></html>'
        r = parse_html(html, "https://base.com/")
        self.assertIn("https://example.com/page", r.links)

    def test_relative_url_resolved_against_base(self):
        html = '<a href="/docs/a.html">d</a>'
        r = parse_html(html, "https://site.example/root/here")
        self.assertIn("https://site.example/docs/a.html", r.links)

    def test_fragment_stripped_from_absolute_link(self):
        html = '<a href="https://a.com/x#section">t</a>'
        r = parse_html(html, "https://a.com/")
        self.assertIn("https://a.com/x", r.links)
        self.assertFalse(any("#" in u for u in r.links))

    def test_dedupes_duplicate_hrefs(self):
        html = '<a href="https://dup.com/a">1</a><a href="https://dup.com/a">2</a>'
        r = parse_html(html, "https://x.com/")
        self.assertEqual(r.links.count("https://dup.com/a"), 1)

    def test_non_http_scheme_excluded(self):
        html = '<a href="mailto:a@b.com">m</a><a href="javascript:void(0)">j</a>'
        r = parse_html(html, "https://x.com/")
        self.assertEqual(r.links, [])

    def test_https_scheme_included(self):
        html = '<a href="https://secure.example/z">z</a>'
        r = parse_html(html, "http://insecure.example/")
        self.assertIn("https://secure.example/z", r.links)


class TestParseHtmlTextAndTitle(unittest.TestCase):
    def test_title_extracted(self):
        html = "<html><head><title>  Hello Title  </title></head><body></body></html>"
        r = parse_html(html, "https://t.com/")
        self.assertEqual(r.title.strip(), "Hello Title")

    def test_visible_text_from_body(self):
        html = "<html><body><p>First words</p><div>Second line</div></body></html>"
        r = parse_html(html, "https://t.com/")
        self.assertIn("First", r.text)
        self.assertIn("Second", r.text)

    def test_script_and_style_content_skipped(self):
        html = (
            "<html><body>"
            "<script>alert('bad')</script>"
            "<style>.x{display:none}</style>"
            "<p>Keep this</p>"
            "</body></html>"
        )
        r = parse_html(html, "https://t.com/")
        self.assertNotIn("alert", r.text)
        self.assertNotIn("display", r.text)
        self.assertIn("Keep this", r.text)


class TestParseHtmlEdgeCases(unittest.TestCase):
    def test_empty_html(self):
        r = parse_html("", "https://empty.com/")
        self.assertEqual(r.title, "")
        self.assertEqual(r.text, "")
        self.assertEqual(r.links, [])

    def test_malformed_unclosed_tags_does_not_crash(self):
        html = "<html><body><p>Unclosed<p>Nested"
        r = parse_html(html, "https://m.com/")
        self.assertIsInstance(r.text, str)

    def test_missing_href_on_anchor_ignored(self):
        html = "<a name='x'>no href</a>"
        r = parse_html(html, "https://m.com/")
        self.assertEqual(r.links, [])


if __name__ == "__main__":
    unittest.main()
