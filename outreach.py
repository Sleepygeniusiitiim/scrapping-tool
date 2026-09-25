"""
Assisted outreach: the AI drafts a message, a recruiter sends it, the reply is read back.

Nothing here sends messages. LinkedIn and Facebook forbid automated messaging,
so the recruiter sends each draft themselves (the page copies it and opens the
candidate's post). When the candidate answers, the recruiter pastes the reply
and `read_reply` pulls out the phone / email the candidate chose to share.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from pipeline import _email_on_page, _phone_on_page
from schema import clean_email, clean_phone

MAX_MESSAGE_CHARS = 700


class OutreachBrief(BaseModel):
    recruiter_name: str = Field(..., min_length=1, max_length=80)
    agency_name: str = Field(..., min_length=1, max_length=120)
    role: str = Field(..., min_length=2, max_length=160)
    job_location: str = Field("", max_length=160)
    extra: str = Field("", max_length=600, description="Salary, benefits, visa, anything else to mention")


class _Draft(BaseModel):
    message: str = ""


class ReplyReading(BaseModel):
    interested: Optional[bool] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: str = ""


DRAFT_SYSTEM = """You write short, honest first-contact messages from a recruitment consultant to a
candidate who publicly showed interest in overseas work (for example by commenting on a hiring post).

Rules:
- Plain text, friendly and professional, 60–110 words, no hashtags, no emojis, no links.
- Greet the person by first name if known.
- Say who is writing (recruiter name, agency) and that we are hiring for the given role (and location).
- Refer to where they showed interest ONLY as described in the facts (e.g. "your comment on a
  press brake operator vacancy post on LinkedIn"). Never invent details about them or the post.
- Ask them, if interested, to reply with their phone / WhatsApp number and email so the team can
  get back to them. Make clear sharing is optional.
- End with a simple opt-out line such as "If you're not interested, no need to reply."
"""

REPLY_SYSTEM = """You read a candidate's reply to a recruiter's message.
- interested: true if they want to proceed, false if they decline, null if unclear.
- email / phone: copy EXACTLY as written in the reply, only if the candidate shared them in this
  reply. Never invent or reformat beyond copying. Null if absent.
- summary: one short sentence on what they said (e.g. availability, experience, questions).
"""


def _facts(candidate: dict, brief: OutreachBrief) -> str:
    where = candidate.get("platform") or "a public page"
    lines = [
        f"Recruiter: {brief.recruiter_name}, {brief.agency_name}",
        f"Role we are hiring for: {brief.role}" + (f" in {brief.job_location}" if brief.job_location else ""),
        f"Extra details to mention if relevant: {brief.extra}" if brief.extra else "",
        f"Candidate name: {candidate.get('name') or 'unknown'}",
        f"Candidate's stated role / skills: {candidate.get('current_role') or '-'}; "
        f"{', '.join(candidate.get('skills') or []) or '-'}",
        f"Where they showed interest: on {where} — their words: \"{(candidate.get('evidence_snippet') or '')[:300]}\"",
    ]
    return "\n".join(l for l in lines if l)


async def draft_message(llm, candidate: dict, brief: OutreachBrief) -> str:
    out = await llm.generate_structured(
        _facts(candidate, brief) + "\n\nWrite the message.", _Draft,
        system_instruction=DRAFT_SYSTEM, temperature=0.5, max_retries=3,
    )
    return " ".join((out.message or "").split())[:MAX_MESSAGE_CHARS] or ""


async def read_reply(llm, reply_text: str) -> ReplyReading:
    reading = await llm.generate_structured(
        f"----- REPLY -----\n{reply_text}\n----- END -----", ReplyReading,
        system_instruction=REPLY_SYSTEM, temperature=0.0, max_retries=3,
    )
    # Keep only contacts that really are in the reply.
    reading.email = clean_email(reading.email) if _email_on_page(reading.email, reply_text) else None
    reading.phone = clean_phone(reading.phone) if _phone_on_page(reading.phone, reply_text) else None
    return reading
