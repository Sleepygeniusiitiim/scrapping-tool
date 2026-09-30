"""Contact extraction and ownership: hidden contacts revealed, directory / broker contacts never attached."""
from bs4 import BeautifulSoup

import company_contacts as cc
import fetcher


def test_cloudflare_email_and_link_contacts_are_revealed():
    key = 0x42
    enc = format(key, "02x") + "".join(format(ord(c) ^ key, "02x") for c in "owner@agency.in")
    html = (f'<p>Mail <a href="/cdn-cgi/l/email-protection#{enc}">[email protected]</a> '
            '<a href="tel:+919814012345">Call now</a> <a href="https://wa.me/919876500000">WhatsApp</a></p>')
    text = fetcher.html_to_text(f"<html><body>{html}</body></html>")
    assert "owner@agency.in" in text and "+919814012345" in text and "919876500000" in text


def test_footer_contacts_are_kept_as_the_site_owners():
    html = "<html><body><p>Posts</p><footer>Email: info@agency.in Phone: 0161-2345678</footer></body></html>"
    text = fetcher.html_to_text(html)
    assert "## Website's own contact details" in text and "info@agency.in" in text


def test_directories_match_whole_domains_only():
    assert cc.is_directory("https://www.placementindia.com/x")
    assert cc.is_directory("https://dehradun.idbf.in/")
    assert cc.is_directory("https://contactout.com/company/x")
    assert cc.is_directory("https://maps.google.co.in/")
    assert not cc.is_directory("https://naukripoint.com")      # "naukri" must not match every domain containing it
    assert not cc.is_directory("https://gillinternational.in/")


def test_site_ownership_and_own_emails():
    assert cc.host_owned("Gill International Recruiting Agency", "https://gillinternational.in/")
    assert not cc.host_owned("R.K. International", "https://contactout.com/")
    got = cc.own_emails(["support@contactout.com", "info@gillinternational.in", "gill.intl@gmail.com",
                         "sales@mahajanconveyors.com"], "https://gillinternational.in/")
    assert got == ["info@gillinternational.in", "gill.intl@gmail.com"]


def test_domain_guesses_never_single_generic_word():
    assert "rolex.com" not in cc.guess_domains("Rolex Travel Services Pvt Ltd")
    assert "gillsmartgroup.com" in cc.guess_domains("Gill Smart Group")


def test_linkedin_post_embed_fallback(monkeypatch):
    import asyncio
    import fetcher

    class Resp:
        def __init__(self, code, url, text):
            self.status_code, self.url, self.text = code, url, text

    class Client:
        def __init__(self, code):
            self.code, self.calls = code, []

        async def get(self, url, timeout=None):
            self.calls.append(url)
            html = ("<html><body><div class='post'><a href='/in/ahmed-hr'>Ahmed Khan · HR Manager at Gulf Build LLC</a>"
                    "<p>We are hiring 20 TIG welders from India for Dubai. Send CV to hr@gulfbuild.ae</p></div></body></html>")
            return Resp(self.code, url, html)

    url = "https://www.linkedin.com/posts/ahmed-hr_hiring-welders-activity-7117159826738434048-AbCd"
    ok = Client(200)
    out = asyncio.run(fetcher._linkedin_embed(ok, url, 10))
    assert out.ok and out.via == "linkedin-embed" and "hr@gulfbuild.ae" in out.markdown
    assert ok.calls == ["https://www.linkedin.com/embed/feed/update/urn:li:activity:7117159826738434048"]
    assert asyncio.run(fetcher._linkedin_embed(ok, "https://www.linkedin.com/in/ahmed", 10)) is None
    monkeypatch.setattr(fetcher.random, "uniform", lambda a, b: 0)     # no real wait in the retry
    limited = Client(999)
    out = asyncio.run(fetcher._linkedin_embed(limited, url, 10))
    assert not out.ok and out.blocked and len(limited.calls) == 2      # one slower retry, then reported
