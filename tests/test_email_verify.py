"""Zero-send mailbox check against a local fake mail server: valid, invalid, catch-all, disposable."""
import asyncio

import email_verify as ev


def _serve(accept):
    async def handle(r, w):
        w.write(b"220 fake\r\n"); await w.drain()
        while True:
            line = (await r.readline()).decode().strip()
            if not line:
                break
            up = line.upper()
            if up.startswith(("EHLO", "MAIL")):
                w.write(b"250 ok\r\n")
            elif up.startswith("RCPT"):
                w.write(b"250 ok\r\n" if accept(line[8:].strip("<>").lower()) else b"550 5.1.1 no such user\r\n")
            elif up == "QUIT":
                w.write(b"221 bye\r\n"); await w.drain(); break
            await w.drain()
        w.close()
    return handle


def test_mailbox_statuses(monkeypatch):
    async def hosts(domain):
        return {"status": "ok", "hosts": ["127.0.0.1"], "reason": "1 MX"}
    monkeypatch.setattr(ev, "mail_hosts", hosts)

    async def main():
        srv = await asyncio.start_server(_serve(lambda a: a == "priya.nair@acme.test"), "127.0.0.1", 0)
        monkeypatch.setattr(ev, "SMTP_PORT", srv.sockets[0].getsockname()[1])
        ev._cache.clear()
        res = await ev.verify_many(["priya.nair@acme.test", "p.nair@acme.test", "x@mailinator.com", "bad@@x"])
        srv.close()
        srv2 = await asyncio.start_server(_serve(lambda a: True), "127.0.0.1", 0)
        monkeypatch.setattr(ev, "SMTP_PORT", srv2.sockets[0].getsockname()[1])
        ev._cache.clear()
        res2 = await ev.verify_many(["any@catchall.test"])
        srv2.close()
        return res, res2
    res, res2 = asyncio.run(main())
    assert res["priya.nair@acme.test"]["status"] == "valid"
    assert res["p.nair@acme.test"]["status"] == "invalid"
    assert res["x@mailinator.com"]["status"] == "disposable"
    assert res["bad@@x"]["status"] == "invalid"
    assert res2["any@catchall.test"]["status"] == "catch_all"
