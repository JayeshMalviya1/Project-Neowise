#!/usr/bin/env python3
"""Stage 4 Domain-Agnostic RAFT (Retrieval-Augmented Fine-Tuning) + Conversational Fallback
Trainer for Northh Dhruva 500M (Project BHARAT-LLM).

Trains Dhruva-500M as a general-purpose, domain-agnostic RAG + structured JSON model
spanning 8 everyday industries (HR, Software/API, Finance/Invoices, Legal/Contracts,
Healthcare/Labs, E-commerce/Product Specs, Education/Academics, Operations/Logistics)
in both English and Hindi/Hinglish.

Balanced 2,500-pair mix (epochs=2, lr=1e-5, batch=2, grad_accum=8 -> ~313 steps):
  1. Procedural Domain-Agnostic RAFT Q&A (1,500 pairs): Dynamically randomized entities,
     numbers, dates, percentages, filenames, and page numbers (1..60) with 1 Oracle +
     2-3 cross-domain Distractors and exact inline `[Doc: {doc} | Page {page}]` citations.
  2. Abstention Offloaded to BAAI/bge-m3 Gate (`n_abstain=0` by default): Prevents
     trigger-happy false refusals on grounded chunks; handled by the 3-tier similarity
     gate in `src/rag/pipeline.py`.
  3. Natural Bilingual Greetings & General Chat (250 pairs): 100% standalone English, Hindi
     & Hinglish greetings/identity turns.
  4. Strong Multi-Domain JSON Replay (750 pairs): Dynamic `type: "table"` (including
     `Invoice, Customer, Amount, Status` invoice tables) and `type: "specs_card"` schemas.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.byte_bpe import ByteBPETokenizer


DEFAULT_STAGE3_CKPT = "/content/drive/MyDrive/Project_dhurva/instruct_500m/dhruva_500m_instruct_json.pt"


# =============================================================================
# 1. Procedural Domain-Agnostic RAFT Generators (8 Everyday Industries)
# =============================================================================

COMPANIES = [
    "ApexNova Technologies", "VayuCloud Systems", "IndusPay Solutions", "Kiran HealthTech",
    "Meridian Logistics", "Solstice Retail", "Zenith FinCorp", "OrbitGrid Labs",
    "Pragati EdTech", "Neelkanth Manufacturing", "VertexSoft India", "Aarohan Enterprises",
]

PEOPLE = [
    "Aarav Sharma", "Priya Nair", "Rohan Kulkarni", "Ananya Iyer", "Vikram Malhotra",
    "Neha Deshmukh", "Siddharth Verma", "Kavita Joshi", "Arjun Reddy", "Meera Krishnan",
    "Kabir Mehta", "Divya Menon", "Aditya Rao", "Pooja Agarwal", "Nikhil Bansal",
    "Sneha Kapoor", "Karan Thakur", "Ritu Choudhary", "Varun Gupta", "Shreya Chatterjee",
]

CITIES = [
    "Bengaluru", "Mumbai", "Pune", "Hyderabad", "Gurugram", "Chennai",
    "Noida", "Ahmedabad", "Kolkata", "Jaipur", "Kochi", "Indore",
]


def _make_doc(rng: random.Random, prefixes: List[str]) -> Tuple[str, int]:
    prefix = rng.choice(prefixes)
    style = rng.randint(0, 3)
    if style == 0:
        name = f"{prefix}_{rng.randint(2022, 2027)}.pdf"
    elif style == 1:
        name = f"{prefix}_v{rng.randint(1, 6)}.pdf"
    elif style == 2:
        name = f"{prefix}_{rng.randint(1000, 9999)}.pdf"
    else:
        name = f"{prefix}_Q{rng.randint(1, 4)}_{rng.randint(2023, 2026)}.pdf"
    return name, rng.randint(1, 60)


def _gen_hr_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["HR_Handbook", "Employee_Policy", "PeopleOps_Guide", "Leave_Policy"])
    comp = rng.choice(COMPANIES)
    sub = rng.randint(0, 3)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        days = rng.randint(14, 30)
        carry = rng.randint(5, 12)
        text = (
            f"At {comp}, full-time employees receive {days} days of paid annual leave per calendar year, "
            f"and up to {carry} unused days may be carried forward into the next year."
        )
        if hindi:
            q = f"{comp} में कर्मचारियों को प्रति वर्ष कितने दिन की पेड एनुअल लीव मिलती है और कितने दिन कैरी फॉरवर्ड हो सकते हैं?"
            a = f"{comp} में कर्मचारियों को प्रति वर्ष {days} दिन की पेड एनुअल लीव मिलती है और अधिकतम {carry} दिन अगले वर्ष में कैरी फॉरवर्ड किए जा सकते हैं {cit}।"
        else:
            q = f"How many annual leave days and carry-forward days does {comp} allow?"
            a = f"{comp} provides {days} paid annual leave days per year, with up to {carry} unused days allowed for carry-forward {cit}."
    elif sub == 1:
        wfh_days = rng.randint(2, 4)
        stipend = rng.randrange(1500, 6000, 500)
        text = (
            f"{comp}'s Hybrid Work Policy allows eligible staff to work remotely up to {wfh_days} days "
            f"per week and provides a monthly home-internet reimbursement limit of Rs {stipend}."
        )
        if hindi:
            q = f"{comp} की हाइब्रिड वर्क पॉलिसी में सप्ताह में कितने दिन रिमोट वर्क और कितना इंटरनेट रिइम्बर्समेंट मिलता है?"
            a = f"{comp} में कर्मचारी सप्ताह में {wfh_days} दिन रिमोट वर्क कर सकते हैं और प्रति माह Rs {stipend} तक इंटरनेट रिइम्बर्समेंट प्राप्त कर सकते हैं {cit}।"
        else:
            q = f"What are the remote work days and internet reimbursement limit at {comp}?"
            a = f"{comp} allows up to {wfh_days} remote work days per week and reimburses up to Rs {stipend} per month for home internet {cit}."
    elif sub == 2:
        notice = rng.choice([30, 45, 60, 90])
        probation_notice = rng.choice([7, 15])
        text = (
            f"Confirmed employees at {comp} must serve a {notice}-day notice period upon resignation, "
            f"whereas employees on probation are subject to a {probation_notice}-day notice period."
        )
        if hindi:
            q = f"{comp} में कन्फर्म कर्मचारियों और प्रोबेशन कर्मचारियों के लिए नोटिस पीरियड कितना है?"
            a = f"{comp} में कन्फर्म कर्मचारियों के लिए {notice} दिन और प्रोबेशन पर कर्मचारियों के लिए {probation_notice} दिन का नोटिस पीरियड है {cit}।"
        else:
            q = f"What is the resignation notice period for confirmed and probationary staff at {comp}?"
            a = f"Confirmed employees at {comp} have a {notice}-day notice period, while employees on probation have a {probation_notice}-day notice period {cit}."
    else:
        meal = rng.randrange(500, 2000, 100)
        travel = rng.randrange(5000, 20000, 1000)
        text = (
            f"Business travel expenses at {comp} are reimbursed up to Rs {meal} per day for meals and "
            f"Rs {travel} per domestic trip for hotel accommodation with valid GST receipts."
        )
        if hindi:
            q = f"{comp} में बिजनेस ट्रैवल के लिए दैनिक भोजन और होटल रिइम्बर्समेंट की सीमा क्या है?"
            a = f"{comp} में वैध GST रसीद के साथ भोजन के लिए प्रति दिन Rs {meal} और घरेलू यात्रा पर होटल के लिए Rs {travel} तक रिइम्बर्समेंट मिलता है {cit}।"
        else:
            q = f"What are the meal and hotel reimbursement caps for business travel at {comp}?"
            a = f"{comp} reimburses up to Rs {meal} per day for meals and Rs {travel} per domestic trip for hotels with valid GST receipts {cit}."

    return {"domain": "hr", "doc": doc, "page": page, "text": text, "q": q, "a": a}


def _gen_software_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["API_Guide", "SDK_Reference", "Dev_Docs", "Gateway_Spec"])
    svc = rng.choice(["VayuAPI", "IndusAuth", "PulseStream", "NovaDB", "GridSearch API", "PayCore API"])
    sub = rng.randint(0, 2)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        rpm = rng.choice([60, 120, 300, 600, 1200, 2500])
        header = rng.choice(["X-API-Key", "Authorization: Bearer <token>", "X-Client-Secret"])
        text = (
            f"The {svc} production endpoint enforces a rate limit of {rpm} requests per minute per tenant. "
            f"All requests must include the `{header}` authentication header over TLS 1.3."
        )
        if hindi:
            q = f"{svc} की रेट लिमिट और आवश्यक ऑथेंटिकेशन हेडर क्या है?"
            a = f"{svc} की रेट लिमिट {rpm} रिक्वेस्ट प्रति मिनट है और हर रिक्वेस्ट में `{header}` हेडर होना आवश्यक है {cit}।"
        else:
            q = f"What is the rate limit and required authentication header for {svc}?"
            a = f"{svc} enforces a rate limit of {rpm} requests per minute and requires the `{header}` authentication header {cit}."
    elif sub == 1:
        timeout = rng.choice([5, 10, 15, 30, 45, 60])
        retries = rng.randint(2, 5)
        text = (
            f"By default, the {svc} client SDK sets an HTTP request timeout of {timeout} seconds "
            f"and performs up to {retries} automatic retries with exponential backoff on HTTP 503 errors."
        )
        if hindi:
            q = f"{svc} SDK में डिफ़ॉल्ट टाइमआउट और ऑटोमैटिक रीट्राई की संख्या क्या है?"
            a = f"{svc} SDK में डिफ़ॉल्ट HTTP टाइमआउट {timeout} सेकंड है और HTTP 503 एरर पर अधिकतम {retries} बार ऑटोमैटिक रीट्राई किया जाता है {cit}।"
        else:
            q = f"What are the default HTTP timeout and retry count in the {svc} SDK?"
            a = f"The {svc} SDK uses a default timeout of {timeout} seconds and performs up to {retries} retries with exponential backoff on HTTP 503 errors {cit}."
    else:
        old_v = f"v{rng.randint(1, 3)}"
        new_v = f"v{int(old_v[1:]) + 1}"
        date_str = f"{rng.choice(['March', 'June', 'September', 'December'])} {rng.randint(1, 28)}, {rng.randint(2026, 2028)}"
        text = (
            f"{svc} {old_v} endpoints are officially deprecated and will reach end-of-life on {date_str}. "
            f"Integrations must migrate to {svc} {new_v} before the sunset date."
        )
        if hindi:
            q = f"{svc} {old_v} कब बंद (end-of-life) होगा और किस वर्ज़न पर माइग्रेट करना है?"
            a = f"{svc} {old_v} की एंड-ऑफ-लाइफ तिथि {date_str} है और इससे पहले {svc} {new_v} पर माइग्रेट करना आवश्यक है {cit}।"
        else:
            q = f"When does {svc} {old_v} reach end-of-life and which version should developers migrate to?"
            a = f"{svc} {old_v} reaches end-of-life on {date_str}, and developers must migrate to {svc} {new_v} before then {cit}."

    return {"domain": "software", "doc": doc, "page": page, "text": text, "q": q, "a": a}


def _gen_finance_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["Invoice", "Quarterly_Report", "Finance_SOP", "Billing_Terms"])
    comp = rng.choice(COMPANIES)
    sub = rng.randint(0, 2)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        net_days = rng.choice([15, 30, 45, 60])
        late_fee = rng.choice([1.2, 1.5, 2.0, 2.5])
        gst = rng.choice([5, 12, 18, 28])
        text = (
            f"Invoices issued by {comp} carry Net {net_days} payment terms with {gst}% GST applied. "
            f"Overdue balances accrue a late payment fee of {late_fee}% per month."
        )
        if hindi:
            q = f"{comp} के इनवॉइस की पेमेंट टर्म, GST दर और लेट फीस कितनी है?"
            a = f"{comp} के इनवॉइस पर Net {net_days} पेमेंट टर्म, {gst}% GST और देरी से भुगतान पर {late_fee}% प्रति माह लेट फीस लागू होती है {cit}।"
        else:
            q = f"What are the payment terms, GST rate, and late fee on invoices from {comp}?"
            a = f"{comp} invoices use Net {net_days} payment terms, apply {gst}% GST, and charge a {late_fee}% monthly late fee on overdue balances {cit}."
    elif sub == 1:
        qtr = f"Q{rng.randint(1, 4)} FY{rng.randint(25, 27)}"
        rev = rng.randint(12, 480)
        yoy = rng.randint(6, 38)
        text = (
            f"In {qtr}, {comp} reported total consolidated revenue of Rs {rev} crore, "
            f"representing a {yoy}% year-over-year growth compared to the prior period."
        )
        if hindi:
            q = f"{qtr} में {comp} का कुल रेवेन्यू और सालाना ग्रोथ (YoY) कितना रहा?"
            a = f"{qtr} में {comp} का कुल रेवेन्यू Rs {rev} करोड़ रहा, जो पिछले वर्ष की तुलना में {yoy}% की वृद्धि दर्शाता है {cit}।"
        else:
            q = f"What was {comp}'s total revenue and YoY growth in {qtr}?"
            a = f"In {qtr}, {comp} reported Rs {rev} crore in total revenue, up {yoy}% year-over-year {cit}."
    else:
        inv_no = rng.randint(10000, 99999)
        amt = rng.randrange(25000, 950000, 5000)
        due = f"{rng.choice(['Jan', 'Feb', 'May', 'Aug', 'Nov'])} {rng.randint(10, 28)}, 2026"
        text = (
            f"Invoice #{inv_no} billed to {comp} totals Rs {amt:,} inclusive of taxes and is due for "
            f"settlement via NEFT/RTGS on or before {due}."
        )
        if hindi:
            q = f"इनवॉइस #{inv_no} की कुल राशि और अंतिम भुगतान तिथि क्या है?"
            a = f"{comp} के नाम जारी इनवॉइस #{inv_no} की कुल राशि Rs {amt:,} है और इसके भुगतान की अंतिम तिथि {due} है {cit}।"
        else:
            q = f"What is the total amount and due date for Invoice #{inv_no}?"
            a = f"Invoice #{inv_no} for {comp} totals Rs {amt:,} and is due on or before {due} {cit}."

    return {"domain": "finance", "doc": doc, "page": page, "text": text, "q": q, "a": a}


def _gen_legal_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["SLA_Contract", "MSA_Agreement", "NDA_Policy", "Vendor_Terms"])
    comp = rng.choice(COMPANIES)
    sub = rng.randint(0, 2)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        uptime = rng.choice(["99.5%", "99.9%", "99.95%", "99.99%"])
        credit = rng.choice([5, 10, 15, 20])
        text = (
            f"Under the Master Service Agreement with {comp}, the provider guarantees {uptime} monthly "
            f"service uptime; any breach entitles the customer to a {credit}% monthly service credit."
        )
        if hindi:
            q = f"{comp} के SLA समझौते में मासिक अपटाइम गारंटी और सर्विस क्रेडिट कितना है?"
            a = f"{comp} के समझौते में {uptime} मासिक अपटाइम की गारंटी है और इसके उल्लंघन पर {credit}% मासिक सर्विस क्रेडिट मिलता है {cit}।"
        else:
            q = f"What uptime guarantee and service credit apply under {comp}'s SLA?"
            a = f"{comp}'s agreement guarantees {uptime} monthly uptime, with a {credit}% monthly service credit for any breach {cit}."
    elif sub == 1:
        term_days = rng.choice([30, 60, 90])
        city = rng.choice(CITIES)
        text = (
            f"Either party may terminate the {comp} vendor contract for convenience by delivering "
            f"{term_days} days' prior written notice. Disputes are governed by arbitration seated in {city}."
        )
        if hindi:
            q = f"{comp} कॉन्ट्रैक्ट समाप्त करने के लिए कितने दिन का लिखित नोटिस आवश्यक है और मध्यस्थता (arbitration) कहां होगी?"
            a = f"{comp} कॉन्ट्रैक्ट समाप्त करने के लिए {term_days} दिन का पूर्व लिखित नोटिस देना होगा और विवादों की मध्यस्थता {city} में होगी {cit}।"
        else:
            q = f"How much written notice is required to terminate the {comp} contract and where is arbitration seated?"
            a = f"Terminating the {comp} contract requires {term_days} days' prior written notice, with arbitration seated in {city} {cit}."
    else:
        nda_yrs = rng.randint(2, 7)
        text = (
            f"All proprietary technical and pricing disclosures exchanged with {comp} remain strictly "
            f"confidential for a duration of {nda_yrs} years following the expiration of the NDA."
        )
        if hindi:
            q = f"{comp} के NDA के तहत गोपनीयता (confidentiality) की अवधि कितने वर्षों तक रहती है?"
            a = f"{comp} के साथ साझा की गई गोपनीय जानकारी NDA समाप्त होने के {nda_yrs} वर्षों तक गोपनीय रहती है {cit}।"
        else:
            q = f"How long do confidentiality obligations last under the {comp} NDA?"
            a = f"Confidentiality obligations under the {comp} NDA remain in force for {nda_yrs} years after expiration {cit}."

    return {"domain": "legal", "doc": doc, "page": page, "text": text, "q": q, "a": a}


def _gen_healthcare_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["Lab_Report", "Clinical_Guide", "Patient_Summary", "Rx_Protocol"])
    sub = rng.randint(0, 2)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        low = rng.choice([70, 75])
        high = rng.choice([99, 100])
        hba1c = rng.choice(["5.7%", "5.6%"])
        text = (
            f"Standard adult fasting blood glucose reference range is {low} to {high} mg/dL, "
            f"and a normal HbA1c level is below {hba1c}."
        )
        if hindi:
            q = "लैब रिपोर्ट के अनुसार वयस्कों के लिए सामान्य फास्टिंग ब्लड ग्लूकोज और HbA1c की सीमा क्या है?"
            a = f"सामान्य फास्टिंग ब्लड ग्लूकोज की सीमा {low} से {high} mg/dL है और सामान्य HbA1c स्तर {hba1c} से कम होता है {cit}।"
        else:
            q = "What are the reference ranges for adult fasting blood glucose and normal HbA1c?"
            a = f"The adult fasting blood glucose reference range is {low} to {high} mg/dL, and normal HbA1c is below {hba1c} {cit}."
    elif sub == 1:
        dose = rng.choice([250, 500, 650])
        hrs = rng.choice([6, 8, 12])
        days = rng.randint(5, 14)
        text = (
            f"The prescribed oral dosage schedule is {dose} mg every {hrs} hours after meals for a "
            f"course of {days} days, followed by a clinical review."
        )
        if hindi:
            q = "दवा की निर्धारित खुराक (dosage) और कोर्स की अवधि क्या है?"
            a = f"निर्धारित खुराक भोजन के बाद हर {hrs} घंटे में {dose} mg है, जिसे {days} दिनों तक लेना है {cit}।"
        else:
            q = "What is the prescribed dosage schedule and duration?"
            a = f"The prescribed schedule is {dose} mg every {hrs} hours after meals for {days} days {cit}."
    else:
        weeks = rng.choice([2, 4, 6, 8, 12])
        test = rng.choice(["Lipid Profile", "Thyroid Panel (TSH)", "Complete Blood Count", "Vitamin D3"])
        text = (
            f"Patients starting therapy should schedule a follow-up {test} lab test after {weeks} weeks "
            f"to evaluate treatment response."
        )
        if hindi:
            q = f"उपचार शुरू करने के कितने सप्ताह बाद {test} की फॉलो-अप जांच करवानी चाहिए?"
            a = f"उपचार शुरू करने के {weeks} सप्ताह बाद {test} की फॉलो-अप जांच करवानी चाहिए {cit}।"
        else:
            q = f"When should the follow-up {test} be scheduled?"
            a = f"A follow-up {test} should be scheduled after {weeks} weeks to evaluate treatment response {cit}."

    return {"domain": "healthcare", "doc": doc, "page": page, "text": text, "q": q, "a": a}


def _gen_ecommerce_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["Product_Manual", "Warranty_Card", "Store_Policy", "Spec_Sheet"])
    prod = rng.choice(["AeroBook Pro 14", "NovaTab X2", "VayuBuds Max", "ZenWatch S5", "VoltCharge PowerBank"])
    sub = rng.randint(0, 2)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        mah = rng.choice([4500, 5000, 6000, 10000, 20000])
        watt = rng.choice([33, 45, 65, 100])
        warranty = rng.choice([12, 18, 24])
        text = (
            f"The {prod} packs a {mah} mAh battery supporting {watt}W fast charging and includes a "
            f"{warranty}-month standard manufacturer warranty."
        )
        if hindi:
            q = f"{prod} की बैटरी क्षमता, फास्ट चार्जिंग और वारंटी अवधि कितनी है?"
            a = f"{prod} में {mah} mAh की बैटरी, {watt}W फास्ट चार्जिंग और {warranty} महीने की वारंटी मिलती है {cit}।"
        else:
            q = f"What are the battery capacity, charging wattage, and warranty of the {prod}?"
            a = f"The {prod} has a {mah} mAh battery, supports {watt}W fast charging, and comes with a {warranty}-month warranty {cit}."
    elif sub == 1:
        ret_days = rng.choice([7, 10, 14, 30])
        ship_days = rng.choice([2, 3, 5])
        text = (
            f"Orders for the {prod} ship within {ship_days} business days and are eligible for a "
            f"{ret_days}-day hassle-free return or replacement in original packaging."
        )
        if hindi:
            q = f"{prod} की शिपिंग अवधि और रिटर्न पॉलिसी कितने दिनों की है?"
            a = f"{prod} के ऑर्डर {ship_days} कार्यदिवसों में शिप होते हैं और मूल पैकेजिंग में {ret_days} दिनों की रिटर्न/रिप्लेसमेंट सुविधा मिलती है {cit}।"
        else:
            q = f"What are the shipping time and return window for the {prod}?"
            a = f"The {prod} ships within {ship_days} business days and has a {ret_days}-day return or replacement window {cit}."
    else:
        weight = rng.choice(["1.25 kg", "1.42 kg", "480 g", "210 g"])
        ip = rng.choice(["IP54", "IP67", "IP68"])
        text = (
            f"The {prod} weighs {weight} and is certified {ip} for dust and water resistance."
        )
        if hindi:
            q = f"{prod} का वजन और डस्ट/वाटर रेजिस्टेंस रेटिंग क्या है?"
            a = f"{prod} का वजन {weight} है और इसे {ip} डस्ट व वाटर रेजिस्टेंस रेटिंग मिली है {cit}।"
        else:
            q = f"What is the weight and IP rating of the {prod}?"
            a = f"The {prod} weighs {weight} and is rated {ip} for dust and water resistance {cit}."

    return {"domain": "ecommerce", "doc": doc, "page": page, "text": text, "q": q, "a": a}


def _gen_education_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["Academic_Bulletin", "Exam_Rules", "Course_Syllabus", "Student_Handbook"])
    univ = rng.choice(["Nalanda Tech University", "Deccan Institute of Science", "Vindhya Engineering College", "Kaveri National University"])
    sub = rng.randint(0, 2)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        att = rng.choice([75, 80, 85])
        credits = rng.choice([120, 140, 160, 180])
        text = (
            f"At {univ}, students must maintain a minimum attendance of {att}% in each course to appear "
            f"for end-semester exams and must complete {credits} credits to graduate."
        )
        if hindi:
            q = f"{univ} में परीक्षा में बैठने के लिए न्यूनतम उपस्थिति और स्नातक के लिए कुल क्रेडिट कितने आवश्यक हैं?"
            a = f"{univ} में परीक्षा के लिए न्यूनतम {att}% उपस्थिति और स्नातक के लिए {credits} क्रेडिट पूरे करना अनिवार्य है {cit}।"
        else:
            q = f"What are the minimum attendance and graduation credit requirements at {univ}?"
            a = f"{univ} requires at least {att}% course attendance to take end-semester exams and {credits} total credits to graduate {cit}."
    elif sub == 1:
        mid = rng.choice([20, 25, 30])
        final = rng.choice([50, 60])
        lab = 100 - mid - final
        text = (
            f"Course grading at {univ} allocates {mid}% weightage to mid-semester tests, {lab}% to "
            f"assignments and labs, and {final}% to the final examination."
        )
        if hindi:
            q = f"{univ} में मिड-सेमेस्टर, लैब/असाइनमेंट और फाइनल परीक्षा का वेटेज कितना है?"
            a = f"{univ} में मिड-सेमेस्टर का वेटेज {mid}%, लैब व असाइनमेंट का {lab}% और फाइनल परीक्षा का {final}% है {cit}।"
        else:
            q = f"What is the exam and assignment weightage breakdown at {univ}?"
            a = f"Grading at {univ} is weighted {mid}% for mid-semester tests, {lab}% for labs/assignments, and {final}% for the final exam {cit}."
    else:
        cgpa = rng.choice(["6.0", "6.5", "7.0", "7.5"])
        text = (
            f"Merit scholarships at {univ} require maintaining a minimum CGPA of {cgpa} with zero active "
            f"backlogs in the preceding academic year."
        )
        if hindi:
            q = f"{univ} में मेरिट स्कॉलरशिप के लिए न्यूनतम CGPA कितना होना चाहिए?"
            a = f"{univ} में मेरिट स्कॉलरशिप के लिए बिना किसी बैकलॉग के न्यूनतम {cgpa} CGPA बनाए रखना आवश्यक है {cit}।"
        else:
            q = f"What minimum CGPA is required to keep a merit scholarship at {univ}?"
            a = f"Students at {univ} must maintain a minimum CGPA of {cgpa} with zero active backlogs to retain merit scholarships {cit}."

    return {"domain": "education", "doc": doc, "page": page, "text": text, "q": q, "a": a}


def _gen_operations_fact(rng: random.Random, hindi: bool) -> Dict[str, object]:
    doc, page = _make_doc(rng, ["Warehouse_SOP", "Logistics_Manual", "Safety_Protocol", "ColdChain_Spec"])
    hub = rng.choice(["Bhiwandi Hub", "Hosur Fulfillment Center", "Manesar Depot", "Sri City Warehouse"])
    sub = rng.randint(0, 2)
    cit = f"[Doc: {doc} | Page {page}]"

    if sub == 0:
        cutoff = rng.choice(["2:00 PM", "3:30 PM", "4:00 PM", "5:00 PM"])
        text = (
            f"Same-day dispatch orders at {hub} must be confirmed before the {cutoff} IST cutoff; "
            f"orders placed after {cutoff} are scheduled for the next morning's line-haul."
        )
        if hindi:
            q = f"{hub} में सेम-डे डिस्पैच के लिए कट-ऑफ समय क्या है?"
            a = f"{hub} में सेम-डे डिस्पैच के लिए ऑर्डर {cutoff} IST से पहले कन्फर्म होना आवश्यक है {cit}।"
        else:
            q = f"What is the same-day dispatch cutoff time at {hub}?"
            a = f"Same-day dispatch orders at {hub} must be confirmed before {cutoff} IST {cit}."
    elif sub == 1:
        t_min = rng.choice([2, 4])
        t_max = rng.choice([8, 10])
        alert_min = rng.choice([10, 15, 20])
        text = (
            f"Cold-chain storage bays at {hub} must be maintained strictly between {t_min}°C and {t_max}°C. "
            f"Any temperature excursion lasting longer than {alert_min} minutes triggers an automated QA alarm."
        )
        if hindi:
            q = f"{hub} में कोल्ड-चेन स्टोरेज का तापमान कितना रहना चाहिए और अलार्म कब बजता है?"
            a = f"{hub} में कोल्ड-चेन स्टोरेज को {t_min}°C से {t_max}°C के बीच रखा जाना चाहिए, और {alert_min} मिनट से अधिक के विचलन पर QA अलार्म सक्रिय होता है {cit}।"
        else:
            q = f"What is the cold-chain temperature range and alarm threshold at {hub}?"
            a = f"Cold-chain bays at {hub} must stay between {t_min}°C and {t_max}°C, and excursions exceeding {alert_min} minutes trigger an automated QA alarm {cit}."
    else:
        speed = rng.choice([8, 10, 12])
        load = rng.choice([1500, 2000, 2500])
        text = (
            f"Forklifts operating inside {hub} aisles must not exceed a maximum speed of {speed} km/h "
            f"and a safe lifting capacity of {load} kg."
        )
        if hindi:
            q = f"{hub} के अंदर फोर्कलिफ्ट की अधिकतम गति और भार क्षमता क्या है?"
            a = f"{hub} के अंदर फोर्कलिफ्ट की अधिकतम गति {speed} km/h और सुरक्षित भार क्षमता {load} kg है {cit}।"
        else:
            q = f"What are the forklift speed limit and safe load capacity inside {hub}?"
            a = f"Inside {hub}, forklifts are limited to {speed} km/h and a maximum safe lifting capacity of {load} kg {cit}."

    return {"domain": "operations", "doc": doc, "page": page, "text": text, "q": q, "a": a}


DOMAIN_GENERATORS: List[Callable[[random.Random, bool], Dict[str, object]]] = [
    _gen_hr_fact,
    _gen_software_fact,
    _gen_finance_fact,
    _gen_legal_fact,
    _gen_healthcare_fact,
    _gen_ecommerce_fact,
    _gen_education_fact,
    _gen_operations_fact,
]


# =============================================================================
# 2. Clean Domain-Agnostic Abstention Questions (ZERO `[Doc: ...]` Tags!)
# =============================================================================

ABSTENTION_BANK: List[Tuple[str, str]] = [
    ("What is the company's stock ticker symbol on the NSE?", "The provided documents do not contain information about the company's NSE stock ticker symbol."),
    ("Who is the Chief Information Security Officer and what is their direct email?", "I am sorry, the provided documents do not list the Chief Information Security Officer or their email address."),
    ("What is the exact customs duty percentage for importing lithium cells from Japan?", "The provided documents do not contain customs duty rates for importing lithium cells."),
    ("What is the maximum payload weight allowed on a Boeing 777 freighter?", "The provided documents do not include aviation payload specifications for a Boeing 777 freighter."),
    ("What is the tuition fee for the PhD program in Astrophysics?", "The provided documents do not mention tuition fees for a PhD program in Astrophysics."),
    ("What is the recommended pediatric dosage for Amoxicillin in infants?", "The provided documents do not contain pediatric dosage instructions for Amoxicillin."),
    ("What is the GraphQL pagination cursor TTL in seconds?", "The provided documents do not specify a GraphQL pagination cursor TTL."),
    ("What is the exact overtime pay multiplier for weekend shifts in Germany?", "The provided documents do not cover weekend overtime pay multipliers in Germany."),
    ("What is the penalty clause for late delivery of maritime shipping containers?", "The provided documents do not contain penalty clauses for maritime container delays."),
    ("What is the replacement cost of an OLED display panel out of warranty?", "The provided documents do not mention out-of-warranty OLED display replacement costs."),
    ("इस कंपनी के मुख्य कार्यकारी अधिकारी (CEO) का व्यक्तिगत फोन नंबर क्या है?", "क्षमा करें, दिए गए दस्तावेजों में CEO का व्यक्तिगत फोन नंबर उपलब्ध नहीं है."),
    ("कंपनी के आगामी वित्तीय वर्ष 2029 के लिए अनुमानित लाभांश (dividend) कितना है?", "दिए गए दस्तावेजों में वित्तीय वर्ष 2029 के अनुमानित लाभांश के बारे में कोई जानकारी नहीं है."),
    ("क्या इस उत्पाद के लिए अंतरराष्ट्रीय एयरपोर्ट लाउंज एक्सेस शामिल है?", "उपलब्ध दस्तावेजों में अंतरराष्ट्रीय एयरपोर्ट लाउंज एक्सेस का कोई उल्लेख नहीं है."),
    ("बायोमेडिकल वेस्ट के निपटान के लिए 2028 का नया शुल्क कितना है?", "दिए गए संदर्भ दस्तावेजों में बायोमेडिकल वेस्ट निपटान शुल्क की जानकारी उपलब्ध नहीं है."),
    ("इस विश्वविद्यालय के हॉस्टल में मासिक मेस शुल्क कितना है?", "दिए गए दस्तावेजों में हॉस्टल के मासिक मेस शुल्क का विवरण नहीं है."),
    ("इस गोदाम के लिए ड्रोन डिलीवरी की अधिकतम उड़ान सीमा कितनी है?", "उपलब्ध दस्तावेजों में ड्रोन डिलीवरी की उड़ान सीमा से संबंधित कोई जानकारी नहीं है."),
]


# =============================================================================
# 3. Natural Bilingual Greetings & General Chat Pool
# =============================================================================

GREETING_CONVERSATIONS: List[Tuple[str, str]] = [
    ("Hello", "Hello! I am Dhruva-500M, your bilingual English and Hindi assistant. How can I help you today?"),
    ("Hi", "Hi! How can I help you today?"),
    ("Hi there!", "Hi there! Ask me anything in English, Hindi, or Hinglish."),
    ("Hey", "Hey! What can I help you with today?"),
    ("Hey Dhruva", "Hello! I am ready to help. What would you like to work on?"),
    ("Hello, how are you?", "I am doing great, thank you! How can I assist you today?"),
    ("Good morning", "Good morning! How can I help you with your questions or documents today?"),
    ("Good evening!", "Good evening! What can I help you with?"),
    ("Who are you?", "I am Northh Dhruva-500M, a bilingual English and Hindi AI assistant built for document Q&A, general chat, and structured JSON extraction."),
    ("What can you do?", "I can answer questions in English and Hindi, extract cited answers from your documents, and convert unstructured text into structured JSON tables or specification cards."),
    ("Thanks!", "You're welcome! Let me know if you need anything else."),
    ("Thank you!", "You're welcome! Happy to help anytime."),
    ("Namaste", "Namaste! मैं ध्रुव-500M हूँ। बताइए, आज मैं आपकी किस प्रकार सहायता कर सकता हूँ?"),
    ("Namaste, aap kaun hain?", "Namaste! Main Northh Dhruva-500M hoon, ek bilingual English aur Hindi AI assistant. Aaj main aapki kya madad kar sakta hoon?"),
    ("Kaise ho?", "Main bilkul theek hoon, shukriya! Aap kaise hain? Batayiye kya madad karoon?"),
    ("Aap kaise hain?", "Main bilkul theek hoon, dhanyawad! Aaj main aapki kis prakar sahayata kar sakta hoon?"),
    ("Kya haal hai?", "Sab badhiya hai! Batayiye, aaj kis topic par baat karni hai?"),
    ("Dhruva kya kar sakta hai?", "Main English aur Hindi mein sawalon ke jawab de sakta hoon, documents se accurate citations nikal sakta hoon, aur structured JSON tables ya specs cards bana sakta hoon."),
    ("Shukriya!", "Aapka swagat hai! Koi aur sawal ho to zaroor puchein."),
    ("नमस्ते", "नमस्ते! मैं ध्रुव-500M हूँ, आपका द्विभाषी एआई सहायक। आज मैं आपकी क्या मदद कर सकता हूँ?"),
    ("नमस्ते, आप कैसे हैं?", "नमस्ते! मैं बिल्कुल ठीक हूँ, धन्यवाद। बताइए, आज मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("आप कौन हैं?", "मैं नॉर्थ ध्रुव-500M हूँ, एक द्विभाषी (हिंदी और अंग्रेजी) एआई मॉडल जिसे दस्तावेज़ विश्लेषण, बातचीत और संरचित JSON डेटा निष्कर्षण के लिए बनाया गया है।"),
    ("धन्यवाद!", "आपका स्वागत है! यदि आपको कोई और जानकारी चाहिए तो बेझिझक पूछें।"),
]


# =============================================================================
# 4. Domain-Agnostic JSON Replay Generator (`table` & `specs_card`)
# =============================================================================

def _gen_dynamic_table_pair(rng: random.Random) -> Tuple[str, str]:
    kind = rng.randint(0, 3)
    num_rows = rng.randint(2, 4)

    if kind == 0:
        # Employee HR Leave Table
        depts = ["Engineering", "Finance", "Product", "Operations", "HR", "Design", "Legal"]
        statuses = ["Active", "On Leave", "Remote", "Probation"]
        names = rng.sample(PEOPLE, k=num_rows)
        lines, rows = [], []
        for idx, name in enumerate(names, 1):
            dept = rng.choice(depts)
            bal = f"{rng.randint(4, 24)} days"
            st = rng.choice(statuses)
            lines.append(f"{idx}. Employee: {name}, Department: {dept}, Leave Balance: {bal}, Status: {st}")
            rows.append([name, dept, bal, st])
        ctx = f"HR Roster - {rng.choice(COMPANIES)}:\n" + "\n".join(lines)
        inst = "Extract the employee leave roster into a JSON table with columns: Employee, Department, Leave Balance, Status."
        obj = {"type": "table", "columns": ["Employee", "Department", "Leave Balance", "Status"], "rows": rows}
    elif kind == 1:
        # Invoice Records Table (`Invoice, Customer, Amount, Status`)
        statuses = ["Paid", "Pending", "Overdue", "Approved", "In Review"]
        customers = rng.sample(COMPANIES, k=num_rows)
        lines, rows = [], []
        for idx, cust in enumerate(customers, 1):
            inv_id = f"INV-{rng.randint(1000, 9999)}"
            amt = f"Rs {rng.randrange(5000, 250000, 500):,}"
            st = rng.choice(statuses)
            lines.append(f"{idx}. Invoice: {inv_id}, Customer: {cust}, Amount: {amt}, Status: {st}")
            rows.append([inv_id, cust, amt, st])
        ctx = f"Quarterly Billing Ledger ({rng.choice(['Q1', 'Q2', 'Q3', 'Q4'])} 2026):\n" + "\n".join(lines)
        inst = "Extract the invoice records into a JSON table with columns: Invoice, Customer, Amount, Status."
        obj = {"type": "table", "columns": ["Invoice", "Customer", "Amount", "Status"], "rows": rows}
    elif kind == 2:
        # Student Grades Table
        courses = ["Machine Learning", "Database Systems", "Operating Systems", "Linear Algebra", "Data Structures", "Cloud Computing"]
        grades = ["A+", "A", "B+", "B", "A-"]
        names = rng.sample(PEOPLE, k=num_rows)
        lines, rows = [], []
        for idx, name in enumerate(names, 1):
            course = rng.choice(courses)
            score = f"{rng.randint(68, 99)}%"
            grade = rng.choice(grades)
            lines.append(f"{idx}. Student: {name}, Course: {course}, Score: {score}, Grade: {grade}")
            rows.append([name, course, score, grade])
        ctx = "Semester Grade Sheet:\n" + "\n".join(lines)
        inst = "Extract the student grades into a JSON table with columns: Student, Course, Score, Grade."
        obj = {"type": "table", "columns": ["Student", "Course", "Score", "Grade"], "rows": rows}
    else:
        # E-commerce Order Logs Table
        statuses = ["Delivered", "Shipped", "Processing", "Returned", "Cancelled"]
        names = rng.sample(PEOPLE, k=num_rows)
        lines, rows = [], []
        for idx, name in enumerate(names, 1):
            oid = f"ORD-{rng.randint(10000, 99999)}"
            amt = f"Rs {rng.randrange(499, 25000, 100)}"
            st = rng.choice(statuses)
            lines.append(f"{idx}. Order ID: {oid}, Customer: {name}, Amount: {amt}, Status: {st}")
            rows.append([oid, name, amt, st])
        ctx = "Daily E-Commerce Order Log:\n" + "\n".join(lines)
        inst = "Extract the order log into a JSON table with columns: Order ID, Customer, Amount, Status."
        obj = {"type": "table", "columns": ["Order ID", "Customer", "Amount", "Status"], "rows": rows}

    prompt = f"Context:\n{ctx}\n\nInstruction:\n{inst}"
    return prompt, json.dumps(obj, indent=2, ensure_ascii=False)


def _gen_dynamic_specs_pair(rng: random.Random) -> Tuple[str, str]:
    kind = rng.randint(0, 3)

    if kind == 0:
        # Laptop Specs Card
        model = rng.choice(["AeroBook Pro 14", "ZenLite X16", "NovaPad Ultra", "KiranBook Air 13"])
        cpu = rng.choice(["Intel Core Ultra 7", "AMD Ryzen 7 8840U", "Apple M3 Pro", "Intel Core i5-1340P"])
        ram = rng.choice(["16 GB LPDDR5", "32 GB DDR5", "24 GB Unified"])
        storage = rng.choice(["512 GB NVMe SSD", "1 TB Gen4 SSD", "2 TB NVMe SSD"])
        price = rng.randrange(55000, 185000, 5000)
        ctx = f"Laptop Listing: {model} powered by {cpu}, {ram} RAM, {storage}, priced at Rs {price:,}."
        inst = "Extract the laptop specifications into a specs_card JSON."
        obj = {
            "type": "specs_card",
            "data": {"Model": model, "Processor": cpu, "RAM": ram, "Storage": storage, "Price": f"Rs {price:,}"},
        }
    elif kind == 1:
        # Cloud Server Specs Card
        inst_name = rng.choice(["c7g.2xlarge", "m6i.4xlarge", "g5.xlarge", "n2-standard-8"])
        vcpu = rng.choice([4, 8, 16, 32])
        mem = rng.choice(["16 GB", "32 GB", "64 GB", "128 GB"])
        region = rng.choice(["ap-south-1 (Mumbai)", "ap-south-2 (Hyderabad)", "us-east-1", "eu-central-1"])
        cost = rng.randrange(3000, 45000, 500)
        ctx = f"Cloud Compute Instance {inst_name}: {vcpu} vCPUs, {mem} memory, deployed in {region}, billed at Rs {cost:,}/month."
        inst = "Extract the cloud server specifications into a specs_card JSON."
        obj = {
            "type": "specs_card",
            "data": {"Instance": inst_name, "vCPU": vcpu, "Memory": mem, "Region": region, "Monthly_Cost": f"Rs {cost:,}"},
        }
    elif kind == 2:
        # SaaS Plan Specs Card
        plan = rng.choice(["Starter Plan", "Growth Plan", "Business Pro", "Enterprise Tier"])
        users = rng.choice(["Up to 10", "Up to 50", "Up to 250", "Unlimited"])
        api_lim = rng.choice(["50,000 req/month", "500,000 req/month", "5M req/month", "Unlimited"])
        support = rng.choice(["Email Support", "24x7 Chat Support", "Dedicated Account Manager"])
        price = rng.randrange(999, 25000, 500)
        ctx = f"SaaS Subscription - {plan}: {users} users, {api_lim} API quota, {support}, at Rs {price:,} per month."
        inst = "Extract the SaaS plan specifications into a specs_card JSON."
        obj = {
            "type": "specs_card",
            "data": {"Plan": plan, "Users": users, "API_Limit": api_lim, "Support": support, "Price_Per_Month": f"Rs {price:,}"},
        }
    else:
        # Home Appliance Specs Card
        prod = rng.choice(["Split Inverter AC", "Front-Load Washing Machine", "Frost-Free Refrigerator", "RO+UV Water Purifier"])
        brand = rng.choice(["Voltas", "LG", "Samsung", "Whirlpool", "Blue Star", "Bosch"])
        cap = rng.choice(["1.5 Ton", "7.5 kg", "340 Litres", "10 Litres"])
        rating = rng.choice(["3-Star BEE", "4-Star BEE", "5-Star BEE"])
        warranty = rng.choice(["1 Year Product / 10 Years Compressor", "2 Years Comprehensive", "5 Years Motor Warranty"])
        ctx = f"Appliance Spec: {brand} {prod}, Capacity {cap}, Energy Rating {rating}, Warranty: {warranty}."
        inst = "Extract the appliance specifications into a specs_card JSON."
        obj = {
            "type": "specs_card",
            "data": {"Product": prod, "Brand": brand, "Capacity": cap, "Energy_Rating": rating, "Warranty": warranty},
        }

    prompt = f"Context:\n{ctx}\n\nInstruction:\n{inst}"
    return prompt, json.dumps(obj, indent=2, ensure_ascii=False)


# =============================================================================
# 5. Dataset Builder & ChatML Tokenization
# =============================================================================

def _format_chunks(chunks: List[Tuple[str, int, str]]) -> str:
    return "\n\n".join(
        f"[Chunk {idx} | Doc: {doc} | Page {page}]\n{text}"
        for idx, (doc, page, text) in enumerate(chunks, start=1)
    )


def build_stage4_dataset(
    n_raft: int = 1500,
    n_abstain: int = 0,
    n_greet: int = 250,
    n_json: int = 750,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Builds the balanced 2,500-pair domain-agnostic Stage 4 dataset (abstention offloaded to RAG gate)."""
    rng = random.Random(seed)
    dataset: List[Tuple[str, str]] = []

    # 1. Domain-Agnostic Procedural RAFT Q&A (1,400 pairs)
    for i in range(n_raft):
        oracle_idx = i % len(DOMAIN_GENERATORS)
        use_hindi = (i % 2 == 1)
        oracle = DOMAIN_GENERATORS[oracle_idx](rng, use_hindi)

        # Pick 2 or 3 distractor chunks from DIFFERENT random domains
        num_distractors = rng.choice([2, 3])
        other_indices = [idx for idx in range(len(DOMAIN_GENERATORS)) if idx != oracle_idx]
        picked_dist_indices = rng.sample(other_indices, k=num_distractors)

        chunk_entries: List[Tuple[str, int, str]] = [
            (str(oracle["doc"]), int(oracle["page"]), str(oracle["text"]))
        ]
        for d_idx in picked_dist_indices:
            dist = DOMAIN_GENERATORS[d_idx](rng, False)
            chunk_entries.append((str(dist["doc"]), int(dist["page"]), str(dist["text"])))
        rng.shuffle(chunk_entries)

        prompt = f"Retrieved Context:\n{_format_chunks(chunk_entries)}\n\nQuestion:\n{oracle['q']}"
        dataset.append((prompt, str(oracle["a"])))

    # 2. Clean Domain-Agnostic Abstention (180 pairs, ZERO `[Doc: ...]` tags)
    for i in range(n_abstain):
        question, abstention_reply = ABSTENTION_BANK[i % len(ABSTENTION_BANK)]
        assert "[Doc:" not in abstention_reply, "Abstention replies must never contain [Doc: ...] citations"
        picked_dist_indices = rng.sample(range(len(DOMAIN_GENERATORS)), k=3)
        chunk_entries = []
        for d_idx in picked_dist_indices:
            dist = DOMAIN_GENERATORS[d_idx](rng, False)
            chunk_entries.append((str(dist["doc"]), int(dist["page"]), str(dist["text"])))

        prompt = f"Retrieved Context:\n{_format_chunks(chunk_entries)}\n\nQuestion:\n{question}"
        dataset.append((prompt, abstention_reply))

    # 3. Natural Bilingual Greetings & General Chat (300 pairs; 100% standalone)
    for i in range(n_greet):
        user_msg, reply = GREETING_CONVERSATIONS[i % len(GREETING_CONVERSATIONS)]
        dataset.append((user_msg, reply))

    # 4. Domain-Agnostic JSON Replay (500 pairs)
    for i in range(n_json):
        if i % 2 == 0:
            dataset.append(_gen_dynamic_table_pair(rng))
        else:
            dataset.append(_gen_dynamic_specs_pair(rng))

    rng.shuffle(dataset)
    return dataset


class RAFTChatMLDataset(Dataset):
    """Tokenizes (prompt, answer) pairs into exact Stage 4 ChatML with `-100` prompt masking."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: ByteBPETokenizer,
        max_seq_len: int = 1024,
    ) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id

        for user_prompt, assistant_answer in pairs:
            prefix = f"### User:\n{user_prompt}\n\n### Assistant:\n"
            prompt_ids = [bos] + tokenizer.encode(prefix)
            answer_ids = tokenizer.encode(assistant_answer) + [eos]

            full_ids = (prompt_ids + answer_ids)[:max_seq_len]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = ([-100] * prompt_len) + full_ids[prompt_len:]

            if any(lbl != -100 for lbl in labels):
                self.samples.append((full_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids, lbls = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lbls, dtype=torch.long)


def _collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    max_len = max(x.size(0) for x, _ in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        L = x.size(0)
        input_ids[i, :L] = x
        labels[i, :L] = y
    return input_ids, labels


def resolve_ckpt(ckpt_arg: str) -> Path:
    candidates = [
        Path(ckpt_arg),
        PROJECT_ROOT / ckpt_arg,
        PROJECT_ROOT / "checkpoints" / "dhruva_500m_instruct_json.pt",
        PROJECT_ROOT.parent / "instruct_500m" / "dhruva_500m_instruct_json.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Stage 3 checkpoint not found. Checked: {[str(c) for c in candidates]}")


def resolve_tokenizer(tok_arg: str) -> Path:
    for c in (Path(tok_arg), PROJECT_ROOT / tok_arg, PROJECT_ROOT / "data" / "tokenizer_32k"):
        if c.is_dir() and (c / "vocab.json").is_file():
            return c.resolve()
    raise FileNotFoundError(f"Tokenizer directory not found: {tok_arg}")


def train_raft(args: argparse.Namespace) -> None:
    print("=" * 76)
    print("  🚀 NORTHH DHRUVA 500M — STAGE 4 DOMAIN-AGNOSTIC RAFT TRAINER")
    print("=" * 76)

    tok_dir = resolve_tokenizer(args.tokenizer_dir)
    tokenizer = ByteBPETokenizer()
    tokenizer.load(str(tok_dir))

    pairs = build_stage4_dataset(
        n_raft=args.n_raft,
        n_abstain=args.n_abstain,
        n_greet=args.n_greet,
        n_json=args.n_json,
        seed=args.seed,
    )
    dataset = RAFTChatMLDataset(pairs, tokenizer, max_seq_len=args.max_seq_len)
    total_optim_steps = max(1, math.ceil((len(dataset) / args.batch_size) * args.epochs / args.grad_accum))

    print(f"  Tokenizer:     {tok_dir} (vocab={tokenizer.vocab_size:,})")
    print(
        f"  Dataset Pairs: {len(dataset):,} "
        f"(RAFT={args.n_raft}, Abstain={args.n_abstain}, Greet={args.n_greet}, JSON={args.n_json})"
    )
    print(
        f"  Schedule:      epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, "
        f"grad_accum={args.grad_accum} (~{total_optim_steps} optimizer steps)"
    )

    if args.dry_run:
        sample_x, sample_y = dataset[0]
        masked_cnt = int((sample_y == -100).sum().item())
        active_cnt = int((sample_y != -100).sum().item())
        raft_ans = [a for p, a in pairs if "Retrieved Context:" in p and "[Doc:" in a]
        unique_citations = set(a[a.rfind("[Doc:") :] for a in raft_ans)
        json_ans = [a for p, a in pairs if p.startswith("Context:\n") and a.startswith("{")]
        valid_json = sum(1 for a in json_ans if isinstance(json.loads(a), dict))
        print(f"  [Dry Run] Sample 0 length={sample_x.size(0)}, masked(-100)={masked_cnt}, active_loss_tokens={active_cnt}")
        print(f"  [Dry Run] Unique dynamic citations in RAFT set: {len(unique_citations):,} / {len(raft_ans):,}")
        print(f"  [Dry Run] Valid multi-domain JSON replay pairs: {valid_json:,} / {len(json_ans):,}")
        print("  [Dry Run] Domain-agnostic RAFT dataset & ChatML masking verified cleanly. Exiting.")
        return

    ckpt_path = resolve_ckpt(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"  Stage 3 Ckpt:  {ckpt_path}")
    print(f"  Device:        {device} (bf16={use_bf16})")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    config: DhruvaConfig = ckpt.get("config")
    if config is None:
        config = DhruvaConfig(
            vocab_size=32768,
            dim=1280,
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            intermediate_size=3584,
            max_seq_len=2048,
        )
    config.use_checkpoint = args.grad_checkpoint

    model = DhruvaForCausalLM(config, init_weights=False)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    cleaned_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(cleaned_sd, strict=False)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.tok_embeddings.weight
    del raw_sd, cleaned_sd, ckpt
    gc.collect()

    model.to(device=device, dtype=torch.bfloat16 if use_bf16 else torch.float32)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_pad(b, pad_id=tokenizer.pad_token_id),
    )

    total_micro_steps = len(loader) * args.epochs
    total_optim_steps = max(1, math.ceil(total_micro_steps / args.grad_accum))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(0.05 * total_optim_steps))
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total_optim_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Optimizer:     AdamW (lr={args.lr}, cosine decay, grad_clip={args.grad_clip})")
    print("-" * 76)

    optim_step = 0
    running_loss = 0.0
    t_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        for micro_idx, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / args.grad_accum

            loss.backward()
            running_loss += loss.item()

            if micro_idx % args.grad_accum == 0 or micro_idx == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                if optim_step % args.log_interval == 0 or optim_step == total_optim_steps:
                    cur_lr = scheduler.get_last_lr()[0]
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"  [Epoch {epoch}/{args.epochs} | Step {optim_step:4d}/{total_optim_steps}] "
                        f"loss={running_loss:.4f} | lr={cur_lr:.2e} | elapsed={elapsed:.1f}s"
                    )
                running_loss = 0.0

    # Save Stage 4 RAFT Checkpoint in BF16
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_sd = {k: v.detach().to(device="cpu", dtype=torch.bfloat16) for k, v in model.state_dict().items()}
    torch.save(
        {
            "config": config,
            "model_state_dict": save_sd,
            "stage": "stage4_raft_domain_agnostic",
            "raft_steps": optim_step,
            "dataset_size": len(dataset),
        },
        str(out_path),
    )
    print("-" * 76)
    print(f"  ✅ Saved Stage 4 RAFT checkpoint to: {out_path.resolve()}")
    print("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 Domain-Agnostic RAFT Trainer for Dhruva 500M")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_STAGE3_CKPT, help="Path to Stage 3 checkpoint")
    parser.add_argument("--output", type=str, default="dhruva_500m_raft_final.pt", help="Output Stage 4 checkpoint path")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tokenizer_32k", help="32K ByteBPETokenizer directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs (default: 2)")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per step")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.0e-5, help="Peak AdamW learning rate (default: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping threshold")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max training sequence length")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable activation gradient checkpointing")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval in optimizer steps")
    parser.add_argument("--n_raft", type=int, default=1500, help="Procedural RAFT pairs across 8 domains (default: 1500)")
    parser.add_argument("--n_abstain", type=int, default=0, help="Abstention pairs (default: 0; handled by BGE-M3 gate)")
    parser.add_argument("--n_greet", type=int, default=250, help="Standalone bilingual greeting/Hinglish pairs (default: 250)")
    parser.add_argument("--n_json", type=int, default=750, help="Multi-domain JSON replay pairs (default: 750)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Build and validate dataset without training")
    return parser.parse_args()


if __name__ == "__main__":
    train_raft(parse_args())
