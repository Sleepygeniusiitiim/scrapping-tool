"""Fit check and score calibration for business searches."""
import asyncio

from intent_miner import planner, verify
from intent_miner.engine import valid_org_name
from intent_miner.models import QuerySpec

CMD = "Give me contact details of all Recruitment agents (registered under MEA) of North India"


def _spec():
    spec = QuerySpec(summary=CMD, target="organizations", destination=["North India"],
                     requirements=["registered with MEA as a recruiting agent"])
    spec.places = planner.expand_places(["North India"], CMD)
    return spec


def test_region_and_abroad_rules():
    leads = [{"lead_key": "a", "display_name": "Dubai DUTCO Construction Co. LLC", "origin": "Delhi", "why": []},
             {"lead_key": "b", "display_name": "Rolex Travel Services", "origin": "Jamshedpur", "why": []},
             {"lead_key": "c", "display_name": "Qatar Manpower Agency", "origin": "New Delhi", "why": []},
             {"lead_key": "d", "display_name": "Mumbai Agency", "phone": "02240230832", "why": []}]
    kept, rejected = asyncio.run(verify.check(None, _spec(), CMD, leads, {"a": "interview in Delhi",
                                                                         "b": "interview in Delhi"}))
    assert [L["display_name"] for L in kept] == ["Qatar Manpower Agency"]
    assert {L["display_name"] for L, _ in rejected} == {"Dubai DUTCO Construction Co. LLC", "Rolex Travel Services",
                                                        "Mumbai Agency"}


def test_calibration_high_only_with_evidence():
    leads = [dict(display_name="Michael Page Delhi", profession="Recruitment Agency", lead_score=88, why=[]),
             dict(display_name="Delhi Overseas", profession="Employment agency", lead_score=89, why=[]),
             dict(display_name="Taj HR Services", profession="overseas recruitment agency", lead_score=93, why=[],
                  evidence=["Government of India Ministry of External Affairs approved overseas recruitment agency"]),
             dict(display_name="Gill International", profession="recruiting agency", lead_score=89, why=[],
                  gov_match={"status": "matched", "dataset": "MEA active RAs"})]
    kept, dropped = verify.calibrate(leads, _spec(), CMD)
    scores = {L["display_name"]: (L["lead_score"], L["tier"]) for L in kept}
    assert [L["display_name"] for L, _ in dropped] == ["Michael Page Delhi"]
    assert scores["Delhi Overseas"] == (79, "MEDIUM")
    assert scores["Taj HR Services"][1] == "HIGH" and scores["Gill International"][1] == "HIGH"


def test_lead_names():
    for bad in ("# Contact details found on the page", "(anonymous post)", "RA", "Recruitment Agency in India"):
        assert not valid_org_name(bad)
    for good in ("Gill International", "Taj HR Services", "9,pees Placement services"):
        assert valid_org_name(good)


def test_phone_area_codes():
    assert verify.phone_city("02240230832") == "Mumbai"
    assert verify.phone_city("+91 161 2345678") == "Ludhiana"
    assert verify.phone_city("9876543210") is None
