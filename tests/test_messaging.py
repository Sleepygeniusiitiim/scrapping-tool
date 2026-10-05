import json

import httpx

import messaging


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_pinnacle_template_request():
    seen = {}

    def handler(request):
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json={"code": "200", "message": "Success", "data": {"messageid": "wamid.1"}})

    keys = {"pinnacle_api_key": "k1", "pinnacle_waba_number": "+91 98100 00000"}
    out = messaging.send_whatsapp(keys, "919876543210", "TPL42", ["Ravi", "TIG Welder"], client=_client(handler))
    assert out["ok"] and out["id"] == "wamid.1"
    assert seen["url"] == messaging.PINNACLE_URL
    assert seen["headers"]["apikey"] == "k1" and seen["headers"]["wabanumber"] == "+919810000000"
    assert seen["body"] == {"from": "919810000000", "to": "919876543210", "type": "template",
                            "message": {"templateid": "TPL42", "placeholders": ["Ravi", "TIG Welder"]}}
    bad = messaging.send_whatsapp(keys, "919876543210", "TPL42", [], client=_client(
        lambda r: httpx.Response(200, json={"code": "400", "message": "Invalid template"})))
    assert not bad["ok"] and "Invalid template" in bad["error"]
    assert not messaging.send_whatsapp({}, "1", "t", [])["ok"]


def test_brevo_email_request():
    seen = {}

    def handler(request):
        seen["headers"], seen["body"] = request.headers, json.loads(request.content)
        return httpx.Response(201, json={"messageId": "<abc@smtp-relay>"})

    keys = {"brevo_api_key": "xkeysib-1", "brevo_sender_email": "careers@example.com", "brevo_sender_name": "MB Careers"}
    out = messaging.send_email(keys, "ravi@gmail.com", "Ravi Kumar", "Welder jobs", "Hi Ravi,\n\nWe are hiring.",
                               "Reply STOP to opt out.", client=_client(handler))
    assert out["ok"] and out["id"] == "<abc@smtp-relay>"
    assert seen["headers"]["api-key"] == "xkeysib-1"
    b = seen["body"]
    assert b["sender"] == {"email": "careers@example.com", "name": "MB Careers"}
    assert b["to"] == [{"email": "ravi@gmail.com", "name": "Ravi Kumar"}] and b["subject"] == "Welder jobs"
    assert "Reply STOP" in b["textContent"] and "Reply STOP" in b["htmlContent"] and "<p>Hi Ravi," in b["htmlContent"]
    denied = messaging.send_email(keys, "a@b.co", "", "s", "b", "o", client=_client(lambda r: httpx.Response(401, text="bad key")))
    assert not denied["ok"] and denied["stop"]


def test_plan_filters_and_dedupes(monkeypatch):
    monkeypatch.setattr(messaging, "recently_messaged", lambda ch, addrs, days: {"919000000003"} if ch == "whatsapp" else set())
    import privacy
    monkeypatch.setattr(privacy, "_load", lambda: {"phones": {"9000000002"}, "emails": set(), "profiles": set()})
    recs = [{"name": "Ravi", "phone": "9876543210", "email": "ravi@gmail.com", "shows_interest": True},
            {"name": "Ravi again", "phone": "+91 98765 43210", "email": "RAVI@gmail.com", "shows_interest": True},
            {"name": "Opted out", "phone": "9000000002", "shows_interest": True},
            {"name": "Messaged last week", "phone": "9000000003", "shows_interest": True},
            {"name": "Not interested", "phone": "9000000004", "shows_interest": False},
            {"name": "No contact", "shows_interest": True}]
    p = messaging.plan(recs, ["whatsapp", "email"], only_interested=True, skip_days=30)
    assert [x["to"] for x in p["whatsapp"]] == ["919876543210"]
    assert [x["to"] for x in p["email"]] == ["ravi@gmail.com"]
    sk = p["skipped"]
    assert sk["opted_out"] == 1 and sk["recently_messaged"] == 1 and sk["not_interested"] == 1 and sk["duplicate"] == 2


def test_send_batch_logs_and_personalises(monkeypatch):
    logged = []
    monkeypatch.setattr(messaging, "_q", lambda sql, params=None, fetch="": logged.append((sql.split()[0], params)))
    sent_bodies = []
    monkeypatch.setattr(messaging, "send_email", lambda keys, to, name, subject, body, opt_out, client=None:
                        sent_bodies.append((to, subject, body)) or {"ok": True, "id": "m1"})
    r = messaging.send_batch({}, "email", [{"to": "ravi@gmail.com", "name": "Ravi Kumar", "source_url": "https://x/1"}],
                             "Saudi welders", subject="{role} jobs", body="Hi {first_name}", opt_out="STOP",
                             defaults={"role": "TIG Welder"}, gap_s=0)
    assert r["sent"] == 1 and sent_bodies == [("ravi@gmail.com", "TIG Welder jobs", "Hi Ravi")]
    assert logged[0][0] == "INSERT" and logged[0][1][0] == "email" and logged[0][1][5] == "sent"
    assert logged[1][0] == "UPDATE"                     # the candidate is marked as contacted
