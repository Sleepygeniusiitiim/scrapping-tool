"""Official lists: PDF tables, column recognition, refusal of ID databases, name matching."""
from pathlib import Path

import pytest

import gov_registry as g

PDF = Path(__file__).parent / "fixtures" / "ra_list.pdf"


def test_pdf_register_rows():
    pytest.importorskip("pdfplumber")
    rows = g.read_rows("ra_list.pdf", PDF.read_bytes())
    recs, info = g.rows_to_records("MEA active RAs", "org", rows)
    assert info["columns"]["name"] == "Name of the RA" and info["columns"]["reg_no"] == "Registration No."
    names = [r["name"] for r in recs]
    assert len(recs) == 4 and names[0] == "Magic Billion Overseas Pvt Ltd"   # header repeated on page 2 skipped
    assert recs[0]["reg_no"].startswith("B-1111/DEL")


def test_id_databases_refused():
    with pytest.raises(ValueError):
        g.rows_to_records("x", "person", [["Name", "EPIC No", "Part No"], ["A B", "1", "2"]])


def test_name_similarity():
    assert g.name_similarity("Sharma Motor Driving School", "M/S SHARMA MOTOR DRIVING TRAINING SCHOOL") > 0.85
    assert g.name_similarity("Sharma Motor Driving School", "Verma Motor Driving School") <= 0.4
    assert g.name_similarity("Gurprit Kaur", "Gurpreet Kaur", org=False) >= 0.85
