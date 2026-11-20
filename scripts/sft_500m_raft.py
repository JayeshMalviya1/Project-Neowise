#!/usr/bin/env python3
"""Stage 4 RAFT (Retrieval-Augmented Fine-Tuning) + Conversational Fallback Trainer
for Northh Dhruva 500M (Project BHARAT-LLM).

Fine-tunes from the Stage 3 checkpoint (`dhruva_500m_instruct_json.pt`) and saves
`dhruva_500m_raft_final.pt` using our exact ChatML template:
    ### User:\n{prompt}\n\n### Assistant:\n
with user prompt tokens masked to `-100`.

Balanced 3,100-pair training mix:
  1. Grounded Multi-Chunk RAFT Q&A (~1,600 pairs) with 1 Oracle + 3 Distractor chunks
     and inline source citations `[Doc: {doc} | Page {page}]`.
  2. Strict Abstention on Missing Facts (~450 pairs) with 3 Distractor chunks (0 Oracle).
  3. Greeting & General Conversation Bypass (~600 pairs; 50% wrapped in noisy Retrieved Context,
     50% standalone chat turns across English & Hindi).
  4. Structured JSON Replay Buffer (~450 pairs) preserving Stage 3 `table` & `specs_card` JSON.
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
from typing import Dict, List, Optional, Tuple

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
# 1. Domain Knowledge Bank (English & Hindi Policy / Engineering Facts)
# =============================================================================

ORACLE_FACTS: List[Dict[str, str]] = [
    # --- FRA 2006 IFR / CFR Rules (English & Hindi) ---
    {
        "doc": "FRA_2006_Act.pdf",
        "page": "4",
        "text": (
            "Under Section 4(6) of the Forest Rights Act (FRA) 2006, the maximum area of forest land "
            "that can be recognized for Individual Forest Rights (IFR) is restricted to the area under "
            "actual occupation and shall in no case exceed 4 hectares per nuclear family."
        ),
        "q_en": "What is the maximum land ceiling for Individual Forest Rights (IFR) under FRA 2006?",
        "a_en": (
            "Under Section 4(6) of the Forest Rights Act 2006, Individual Forest Rights (IFR) are limited "
            "to the land under actual occupation and cannot exceed 4 hectares per nuclear family "
            "[Doc: FRA_2006_Act.pdf | Page 4]."
        ),
        "q_hi": "वन अधिकार अधिनियम (FRA) 2006 के तहत व्यक्तिगत वन अधिकार (IFR) के लिए अधिकतम भूमि सीमा क्या है?",
        "a_hi": (
            "वन अधिकार अधिनियम (FRA) 2006 की धारा 4(6) के अनुसार व्यक्तिगत वन अधिकार (IFR) वास्तविक कब्जे "
            "वाली भूमि तक सीमित हैं और प्रति एकल परिवार अधिकतम 4 हेक्टेयर से अधिक नहीं हो सकते "
            "[Doc: FRA_2006_Act.pdf | Page 4]।"
        ),
    },
    {
        "doc": "FRA_2006_Act.pdf",
        "page": "2",
        "text": (
            "Section 2(o) of the Forest Rights Act 2006 defines Other Traditional Forest Dwellers (OTFD) "
            "as any member or community who has for at least three generations (75 years) prior to the "
            "13th day of December 2005 primarily resided in and depended on the forest or forest land for "
            "bona fide livelihood needs."
        ),
        "q_en": "What is the eligibility criteria and cutoff date for Other Traditional Forest Dwellers (OTFD) under FRA 2006?",
        "a_en": (
            "Other Traditional Forest Dwellers (OTFD) must have primarily resided in and depended on "
            "forest land for bona fide livelihood needs for at least three generations (75 years) prior "
            "to December 13, 2005 [Doc: FRA_2006_Act.pdf | Page 2]."
        ),
        "q_hi": "FRA 2006 के अंतर्गत अन्य पारंपरिक वन निवासी (OTFD) के लिए पात्रता और कट-ऑफ तिथि क्या है?",
        "a_hi": (
            "FRA 2006 की धारा 2(o) के अनुसार अन्य पारंपरिक वन निवासियों (OTFD) को 13 दिसंबर 2005 से पहले "
            "कम से कम तीन पीढ़ियों (75 वर्षों) तक वन भूमि पर निवास और आजीविका के लिए निर्भर होना आवश्यक है "
            "[Doc: FRA_2006_Act.pdf | Page 2]।"
        ),
    },
    {
        "doc": "FRA_2006_Act.pdf",
        "page": "3",
        "text": (
            "Section 3(1)(i) of the Forest Rights Act 2006 vests Community Forest Resource (CFR) rights "
            "in the Gram Sabha to protect, regenerate, conserve, or manage any community forest resource "
            "which they have been traditionally protecting and conserving for sustainable use."
        ),
        "q_en": "Which section of FRA 2006 grants Community Forest Resource (CFR) management rights to the Gram Sabha?",
        "a_en": (
            "Section 3(1)(i) of FRA 2006 grants the Gram Sabha Community Forest Resource (CFR) rights "
            "to protect, regenerate, conserve, and manage traditional community forest resources "
            "[Doc: FRA_2006_Act.pdf | Page 3]."
        ),
        "q_hi": "FRA 2006 की कौन सी धारा ग्राम सभा को सामुदायिक वन संसाधन (CFR) प्रबंधन का अधिकार देती है?",
        "a_hi": (
            "FRA 2006 की धारा 3(1)(i) ग्राम सभा को पारंपरिक सामुदायिक वन संसाधनों (CFR) के संरक्षण, "
            "पुनरुत्पादन और प्रबंधन का अधिकार प्रदान करती है [Doc: FRA_2006_Act.pdf | Page 3]।"
        ),
    },
    # --- Gram Sabha Quorum & FRC Rules ---
    {
        "doc": "FRA_Rules_2012.pdf",
        "page": "7",
        "text": (
            "Under Rule 4(2) of the Scheduled Tribes and Other Traditional Forest Dwellers (Recognition "
            "of Forest Rights) Amendment Rules 2012, the quorum of the Gram Sabha meeting shall be not "
            "less than one-half (50%) of all adult members of such Gram Sabha, provided that at least "
            "one-third of the members present are women."
        ),
        "q_en": "What is the required Gram Sabha quorum and women's representation under FRA Amendment Rules 2012?",
        "a_en": (
            "Under Rule 4(2) of the FRA Amendment Rules 2012, the Gram Sabha quorum must be at least "
            "one-half (50%) of all adult members, and at least one-third of the members present must be "
            "women [Doc: FRA_Rules_2012.pdf | Page 7]."
        ),
        "q_hi": "FRA संशोधन नियम 2012 के अनुसार ग्राम सभा का कोरम और महिलाओं की भागीदारी कितनी होनी चाहिए?",
        "a_hi": (
            "FRA संशोधन नियम 2012 के नियम 4(2) के तहत ग्राम सभा का कोरम सभी वयस्क सदस्यों का कम से कम "
            "आधा (50%) होना चाहिए, और उपस्थित सदस्यों में कम से कम एक-तिहाई महिलाएं होनी आवश्यक हैं "
            "[Doc: FRA_Rules_2012.pdf | Page 7]।"
        ),
    },
    {
        "doc": "FRA_Rules_2012.pdf",
        "page": "8",
        "text": (
            "Rule 3(1) mandates that the Gram Sabha shall elect a Forest Rights Committee (FRC) of not "
            "less than 10 and not more than 15 members from amongst its members, where at least two-thirds "
            "of the members shall be Scheduled Tribes and at least one-third shall be women."
        ),
        "q_en": "What is the composition rule for the Forest Rights Committee (FRC) elected by the Gram Sabha?",
        "a_en": (
            "According to Rule 3(1), the Forest Rights Committee (FRC) must have between 10 and 15 members, "
            "with at least two-thirds Scheduled Tribes and at least one-third women "
            "[Doc: FRA_Rules_2012.pdf | Page 8]."
        ),
        "q_hi": "ग्राम सभा द्वारा चुनी गई वन अधिकार समिति (FRC) की संरचना के क्या नियम हैं?",
        "a_hi": (
            "नियम 3(1) के अनुसार वन अधिकार समिति (FRC) में 10 से 15 सदस्य होने चाहिए, जिनमें से कम से कम "
            "दो-तिहाई अनुसूचित जनजाति (ST) के सदस्य और कम से कम एक-तिहाई महिलाएं होनी चाहिए "
            "[Doc: FRA_Rules_2012.pdf | Page 8]।"
        ),
    },
    # --- IS 456:2000 Concrete & Curing ---
    {
        "doc": "IS_456_2000.pdf",
        "page": "27",
        "text": (
            "Clause 13.5.1 of IS 456:2000 specifies that exposed surfaces of concrete shall be kept "
            "continuously in a damp or wet condition for a minimum period of 7 days from the date of "
            "placing concrete when Ordinary Portland Cement (OPC) is used, and not less than 10 days "
            "when mineral admixtures or blended cements are used, extended to 14 days in hot and dry weather."
        ),
        "q_en": "What is the minimum concrete curing period specified by IS 456:2000 for OPC and blended cements?",
        "a_en": (
            "IS 456:2000 Clause 13.5.1 requires a minimum curing period of 7 days for Ordinary Portland "
            "Cement (OPC), at least 10 days for blended cements or mineral admixtures, and 14 days in "
            "hot and dry weather [Doc: IS_456_2000.pdf | Page 27]."
        ),
        "q_hi": "IS 456:2000 के अनुसार OPC और ब्लेंडेड सीमेंट के लिए कंक्रीट की न्यूनतम क्योरिंग अवधि क्या है?",
        "a_hi": (
            "IS 456:2000 के खंड 13.5.1 के अनुसार OPC के लिए न्यूनतम 7 दिन, ब्लेंडेड सीमेंट या मिनरल "
            "एडमिक्सचर के लिए कम से कम 10 दिन, और गर्म व शुष्क मौसम में 14 दिन क्योरिंग आवश्यक है "
            "[Doc: IS_456_2000.pdf | Page 27]।"
        ),
    },
    {
        "doc": "IS_456_2000.pdf",
        "page": "19",
        "text": (
            "Table 5 of IS 456:2000 mandates that the minimum grade of reinforced concrete (RCC) for "
            "mild exposure conditions shall be M20 with a minimum cement content of 300 kg/m3 and a "
            "maximum free water-cement ratio of 0.55."
        ),
        "q_en": "What is the minimum RCC grade and maximum water-cement ratio for mild exposure under IS 456:2000?",
        "a_en": (
            "Under Table 5 of IS 456:2000, mild exposure requires a minimum RCC grade of M20, minimum "
            "cement content of 300 kg/m3, and a maximum free water-cement ratio of 0.55 "
            "[Doc: IS_456_2000.pdf | Page 19]."
        ),
        "q_hi": "IS 456:2000 के अनुसार सामान्य (mild) वातावरण में RCC का न्यूनतम ग्रेड और अधिकतम जल-सीमेंट अनुपात क्या है?",
        "a_hi": (
            "IS 456:2000 की तालिका 5 के अनुसार सामान्य (mild) वातावरण के लिए न्यूनतम RCC ग्रेड M20, "
            "न्यूनतम सीमेंट मात्रा 300 किग्रा/घन मीटर, और अधिकतम जल-सीमेंट अनुपात 0.55 है "
            "[Doc: IS_456_2000.pdf | Page 19]।"
        ),
    },
    # --- Seismic Zone III Concrete & Steel Specs ---
    {
        "doc": "IS_13920_2016.pdf",
        "page": "6",
        "text": (
            "For buildings located in Seismic Zone III and above, IS 13920:2016 Clause 5.2 specifies "
            "a minimum concrete grade of M20 (M25 recommended for buildings taller than 15m) and "
            "ductile steel reinforcement of grade Fe415D or Fe500D with minimum elongation of 14.5% "
            "and UTS/YS ratio of at least 1.10."
        ),
        "q_en": "What concrete grade and steel reinforcement specifications apply in Seismic Zone III per IS 13920:2016?",
        "a_en": (
            "IS 13920:2016 Clause 5.2 specifies a minimum concrete grade of M20 (M25 for buildings over "
            "15m) and ductile Fe415D or Fe500D steel with at least 14.5% elongation and UTS/YS ratio "
            ">= 1.10 in Seismic Zone III [Doc: IS_13920_2016.pdf | Page 6]."
        ),
        "q_hi": "IS 13920:2016 के अनुसार भूकंप क्षेत्र III (Seismic Zone III) में कंक्रीट और स्टील के क्या मानक हैं?",
        "a_hi": (
            "IS 13920:2016 के खंड 5.2 के अनुसार भूकंप क्षेत्र III में न्यूनतम M20 कंक्रीट (15 मीटर से ऊंचे "
            "भवनों के लिए M25) और कम से कम 14.5% बढ़ाव वाले डक्टाइल Fe415D या Fe500D स्टील का उपयोग "
            "किया जाना चाहिए [Doc: IS_13920_2016.pdf | Page 6]।"
        ),
    },
    {
        "doc": "IS_13920_2016.pdf",
        "page": "11",
        "text": (
            "In Seismic Zone III ductile beams under IS 13920:2016 Clause 6.3.5, closed stirrups with "
            "135-degree hooks and 10d extension must be spaced at not more than d/4 or 8 times the "
            "diameter of the smallest longitudinal bar within a distance of 2d from the column face."
        ),
        "q_en": "What is the stirrup spacing and hook requirement for ductile beams in Seismic Zone III?",
        "a_en": (
            "Per IS 13920:2016 Clause 6.3.5, ductile beams require closed stirrups with 135-degree hooks "
            "(10d extension) spaced at no more than d/4 or 8 times the smallest longitudinal bar diameter "
            "within 2d of the column face [Doc: IS_13920_2016.pdf | Page 11]."
        ),
        "q_hi": "भूकंप क्षेत्र III में डक्टाइल बीम के लिए स्टिरप (stirrup) की दूरी और हुक के क्या नियम हैं?",
        "a_hi": (
            "IS 13920:2016 के खंड 6.3.5 के अनुसार कॉलम फेस से 2d दूरी के भीतर 135-डिग्री हुक (10d एक्सटेंशन) "
            "वाले बंद स्टिरप की दूरी d/4 या सबसे छोटे लोंगिट्यूडिनल बार के व्यास के 8 गुना से अधिक नहीं "
            "होनी चाहिए [Doc: IS_13920_2016.pdf | Page 11]।"
        ),
    },
    # --- GFR 2017 Bid Security & Performance Security ---
    {
        "doc": "GFR_2017.pdf",
        "page": "56",
        "text": (
            "Rule 170 of the General Financial Rules (GFR) 2017 states that Bid Security (Earnest Money "
            "Deposit / EMD) should normally range between 2% to 5% of the estimated value of the goods "
            "to be procured. Micro and Small Enterprises (MSEs) and Startups recognized by DPIIT are "
            "exempted from furnishing Bid Security."
        ),
        "q_en": "What is the Bid Security (EMD) percentage under GFR 2017 Rule 170 and who is exempted?",
        "a_en": (
            "Under GFR 2017 Rule 170, Bid Security (EMD) ranges from 2% to 5% of the estimated procurement "
            "value, and Micro and Small Enterprises (MSEs) as well as DPIIT-recognized Startups are "
            "exempted [Doc: GFR_2017.pdf | Page 56]."
        ),
        "q_hi": "GFR 2017 नियम 170 के तहत बिड सिक्योरिटी (EMD) कितनी होती है और किसे छूट प्राप्त है?",
        "a_hi": (
            "GFR 2017 नियम 170 के अनुसार बिड सिक्योरिटी (EMD) अनुमानित खरीद मूल्य का 2% से 5% तक होती है, "
            "और सूक्ष्म एवं लघु उद्यमों (MSEs) तथा DPIIT मान्यता प्राप्त स्टार्टअप्स को इससे छूट दी गई है "
            "[Doc: GFR_2017.pdf | Page 56]।"
        ),
    },
    {
        "doc": "GFR_2017.pdf",
        "page": "57",
        "text": (
            "Rule 171 of GFR 2017 mandates that Performance Security shall be obtained from the successful "
            "bidder at 3% to 10% of the value of the contract, valid for a period of 60 days beyond the "
            "date of completion of all contractual obligations including warranty."
        ),
        "q_en": "What is the Performance Security requirement and validity period under GFR 2017 Rule 171?",
        "a_en": (
            "GFR 2017 Rule 171 requires Performance Security of 3% to 10% of the contract value from the "
            "successful bidder, remaining valid for 60 days beyond completion of all contractual and "
            "warranty obligations [Doc: GFR_2017.pdf | Page 57]."
        ),
        "q_hi": "GFR 2017 नियम 171 के अंतर्गत परफॉर्मेंस सिक्योरिटी की दर और वैधता अवधि क्या है?",
        "a_hi": (
            "GFR 2017 नियम 171 के अनुसार सफल बोलीदाता से अनुबंध मूल्य का 3% से 10% परफॉर्मेंस सिक्योरिटी "
            "ली जाती है, जो वारंटी सहित सभी अनुबंध दायित्व पूरे होने के 60 दिन बाद तक वैध रहनी चाहिए "
            "[Doc: GFR_2017.pdf | Page 57]।"
        ),
    },
    {
        "doc": "GFR_2017.pdf",
        "page": "52",
        "text": (
            "Rule 154 of GFR 2017 permits direct purchase of goods up to Rs. 50,000 on each occasion "
            "without inviting quotations or bids, subject to the competent authority recording a "
            "certificate of reasonableness of price and quality."
        ),
        "q_en": "What is the monetary limit for direct purchase without quotations under GFR 2017 Rule 154?",
        "a_en": (
            "Under GFR 2017 Rule 154, direct purchase without inviting quotations is permitted up to "
            "Rs. 50,000 per occasion with a reasonableness certificate from the competent authority "
            "[Doc: GFR_2017.pdf | Page 52]."
        ),
        "q_hi": "GFR 2017 नियम 154 के तहत बिना कोटेशन सीधी खरीद की अधिकतम सीमा क्या है?",
        "a_hi": (
            "GFR 2017 नियम 154 के अनुसार सक्षम प्राधिकारी के प्रमाणपत्र के साथ प्रति अवसर 50,000 रुपये तक "
            "की सीधी खरीद बिना कोटेशन के की जा सकती है [Doc: GFR_2017.pdf | Page 52]।"
        ),
    },
]

UNANSWERABLE_QUESTIONS: List[Tuple[str, str]] = [
    (
        "What is the exact penalty amount for late filing of GST Form GSTR-9C in 2026?",
        "I am sorry, the provided documents do not contain information about GST Form GSTR-9C late filing penalties.",
    ),
    (
        "What is the maximum permissible wind speed design load for bridges in Assam under IRC 6?",
        "The provided documents do not contain information regarding IRC 6 bridge wind load specifications for Assam.",
    ),
    (
        "Who is the current District Collector of Mandla district and what is their phone number?",
        "I cannot answer this because the provided documents do not contain contact details or names of current district officials.",
    ),
    (
        "What is the solar rooftop subsidy percentage under PM Surya Ghar Muft Bijli Yojana?",
        "The provided documents do not contain details about the PM Surya Ghar Muft Bijli Yojana subsidy rates.",
    ),
    (
        "What is the minimum runway length required for an Airbus A320neo at Leh airport?",
        "The retrieved context does not contain aviation or airport runway specifications for Leh airport.",
    ),
    (
        "2026 में आयकर अधिनियम की धारा 80C के तहत अधिकतम कर छूट की सीमा क्या है?",
        "क्षमा करें, दिए गए दस्तावेजों में आयकर अधिनियम की धारा 80C की कर छूट सीमा के बारे में कोई जानकारी उपलब्ध नहीं है।",
    ),
    (
        "प्रधानमंत्री आवास योजना (ग्रामीण) के अंतर्गत पहाड़ी क्षेत्रों में प्रति घर कितनी राशि दी जाती है?",
        "दिए गए संदर्भ दस्तावेजों में प्रधानमंत्री आवास योजना (ग्रामीण) की वित्तीय सहायता राशि का उल्लेख नहीं है।",
    ),
    (
        "भोपाल मेट्रो रेल परियोजना के फेज-2 की कुल अनुमानित लागत कितनी है?",
        "उपलब्ध दस्तावेजों में भोपाल मेट्रो रेल परियोजना के फेज-2 की लागत से संबंधित कोई जानकारी नहीं है।",
    ),
    (
        "रेलवे ब्रिज के लिए IS 800 के तहत स्टील गर्डर की न्यूनतम मोटाई क्या है?",
        "दिए गए दस्तावेजों में IS 800 या रेलवे ब्रिज स्टील गर्डर की मोटाई से जुड़ी जानकारी उपलब्ध नहीं है।",
    ),
    (
        "What is the interest rate for Kisan Credit Card loans above Rs 3 lakh?",
        "The provided documents do not mention Kisan Credit Card loan interest rates.",
    ),
]

GREETING_CONVERSATIONS: List[Tuple[str, str]] = [
    ("Hello", "Hello! I am Dhruva-500M, your bilingual English and Hindi assistant. How can I help you today?"),
    ("Hi there!", "Hi! How can I assist you today? Feel free to ask questions in English or Hindi."),
    ("Hey", "Hello! How can I help you today?"),
    ("Good morning!", "Good morning! How can I help you with your questions or documents today?"),
    ("Namaste", "Namaste! मैं ध्रुव-500M हूँ। बताइए, आज मैं आपकी किस प्रकार सहायता कर सकता हूँ?"),
    ("नमस्ते", "नमस्ते! मैं ध्रुव-500M हूँ, आपका द्विभाषी एआई सहायक। आज मैं आपकी क्या मदद कर सकता हूँ?"),
    ("नमस्ते, आप कैसे हैं?", "नमस्ते! मैं बिल्कुल ठीक हूँ, धन्यवाद। बताइए, आज मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("Who are you?", "I am Northh Dhruva-500M, a 470.6M-parameter bilingual (English & Hindi) foundation model built for grounded Q&A, chat, and structured JSON extraction."),
    ("आप कौन हैं?", "मैं नॉर्थ ध्रुव-500M (Northh Dhruva-500M) हूँ, एक द्विभाषी (हिंदी और अंग्रेजी) एआई मॉडल जिसे सटीक उत्तर, दस्तावेज़ विश्लेषण और संरचित डेटा निष्कर्षण के लिए बनाया गया है।"),
    ("Thank you!", "You're welcome! Let me know if you need anything else."),
    ("धन्यवाद!", "आपका स्वागत है! यदि आपको कोई और जानकारी चाहिए तो बेझिझक पूछें।"),
    ("How are you doing today?", "I'm doing great and ready to help! What would you like to work on today?"),
]

JSON_REPLAY_TEMPLATES: List[Dict[str, str]] = [
    {
        "context": (
            "Forest Rights Act (FRA) Gram Sabha Claim Register - {district} District:\n"
            "1. Claimant: {n1}, Village: {v1}, Area Claimed: {a1} Ha, Status: {s1}\n"
            "2. Claimant: {n2}, Village: {v2}, Area Claimed: {a2} Ha, Status: {s2}\n"
            "3. Claimant: {n3}, Village: {v3}, Area Claimed: {a3} Ha, Status: {s3}"
        ),
        "instruction": "Extract the FRA land claims into a structured JSON table.",
        "kind": "table",
    },
    {
        "context": (
            "Contractor Quotation: {pkg} at Rs {price} per sq.ft. "
            "Specifications: {cement}, {steel}, {wiring}, and {paint}."
        ),
        "instruction": "Extract the construction quote into a specs_card JSON.",
        "kind": "specs_card",
    },
]


def _format_chunks(chunks: List[Dict[str, str]]) -> str:
    parts = []
    for idx, ch in enumerate(chunks, start=1):
        parts.append(f"[Chunk {idx} | Doc: {ch['doc']} | Page {ch['page']}]\n{ch['text']}")
    return "\n\n".join(parts)


def build_stage4_dataset(
    n_raft: int = 1600,
    n_abstain: int = 450,
    n_greet: int = 600,
    n_json: int = 450,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Builds the balanced 3,100-pair Stage 4 RAFT + Conversational Fallback + JSON dataset."""
    rng = random.Random(seed)
    dataset: List[Tuple[str, str]] = []

    # 1. Grounded Multi-Chunk RAFT Q&A (~1,600 pairs)
    for i in range(n_raft):
        oracle = ORACLE_FACTS[i % len(ORACLE_FACTS)]
        distractor_pool = [f for f in ORACLE_FACTS if f is not oracle]
        distractors = rng.sample(distractor_pool, k=3)
        chunks = [oracle] + distractors
        rng.shuffle(chunks)

        use_hindi = (i % 2 == 1)
        question = oracle["q_hi"] if use_hindi else oracle["q_en"]
        answer = oracle["a_hi"] if use_hindi else oracle["a_en"]

        prompt = f"Retrieved Context:\n{_format_chunks(chunks)}\n\nQuestion:\n{question}"
        dataset.append((prompt, answer))

    # 2. Strict Abstention on Missing Facts (~450 pairs)
    for i in range(n_abstain):
        question, abstention_reply = UNANSWERABLE_QUESTIONS[i % len(UNANSWERABLE_QUESTIONS)]
        distractors = rng.sample(ORACLE_FACTS, k=3)
        prompt = f"Retrieved Context:\n{_format_chunks(distractors)}\n\nQuestion:\n{question}"
        dataset.append((prompt, abstention_reply))

    # 3. Greeting & General Conversation Bypass (~600 pairs; 50% noisy context, 50% standalone)
    for i in range(n_greet):
        user_msg, reply = GREETING_CONVERSATIONS[i % len(GREETING_CONVERSATIONS)]
        if i % 2 == 0:
            # 50% wrapped inside noisy Retrieved Context chunks
            distractors = rng.sample(ORACLE_FACTS, k=3)
            prompt = f"Retrieved Context:\n{_format_chunks(distractors)}\n\nQuestion:\n{user_msg}"
        else:
            # 50% standalone chat turns
            prompt = user_msg
        dataset.append((prompt, reply))

    # 4. Structured JSON Replay Buffer (~450 pairs)
    districts = ["Mandla", "Dindori", "Balaghat", "Bastar", "Gadchiroli", "Chhindwara"]
    names = ["Ramesh Gond", "Sukarti Baiga", "Mangal Singh", "Kamla Devi", "Birsa Uikey", "Savitri Markam"]
    villages = ["Bichhiya", "Mawai", "Ghughri", "Karanjia", "Baihar", "Narayanpur"]
    statuses = ["Approved", "Pending", "Verified"]
    packages = ["Standard Residential Package", "Premium Civic Package", "Turnkey Gram Bhawan Package"]
    cements = ["UltraTech OPC 43 Grade Cement", "ACC PPC Blended Cement", "Shree Cement OPC 53 Grade"]
    steels = ["Tata Tiscon Fe500D TMT Steel", "JSW Neosteel Fe500D TMT", "SAIL SeQR Fe500D Steel"]
    wirings = ["Finolex Electrical Wiring", "Polycab FR-LSH Wiring", "Havells Flame Retardant Cables"]
    paints = ["Asian Paints Royale finish", "Berger WeatherCoat Exterior", "Nerolac Impressions finish"]

    for i in range(n_json):
        if i % 2 == 0:
            dist = rng.choice(districts)
            picked_names = rng.sample(names, k=3)
            picked_villages = rng.sample(villages, k=3)
            areas = [f"{rng.uniform(0.8, 3.8):.2f}" for _ in range(3)]
            sts = [rng.choice(statuses) for _ in range(3)]
            ctx = JSON_REPLAY_TEMPLATES[0]["context"].format(
                district=dist,
                n1=picked_names[0], v1=picked_villages[0], a1=areas[0], s1=sts[0],
                n2=picked_names[1], v2=picked_villages[1], a2=areas[1], s2=sts[1],
                n3=picked_names[2], v3=picked_villages[2], a3=areas[2], s3=sts[2],
            )
            inst = JSON_REPLAY_TEMPLATES[0]["instruction"]
            prompt = f"Context:\n{ctx}\n\nInstruction:\n{inst}"
            target_obj = {
                "type": "table",
                "columns": ["Claimant", "Village", "Area Claimed", "Status"],
                "rows": [
                    [picked_names[0], picked_villages[0], f"{areas[0]} Ha", sts[0]],
                    [picked_names[1], picked_villages[1], f"{areas[1]} Ha", sts[1]],
                    [picked_names[2], picked_villages[2], f"{areas[2]} Ha", sts[2]],
                ],
            }
            answer = json.dumps(target_obj, indent=2, ensure_ascii=False)
        else:
            pkg = rng.choice(packages)
            price = rng.choice([1450, 1650, 1850, 2100])
            cem = rng.choice(cements)
            stl = rng.choice(steels)
            wir = rng.choice(wirings)
            pnt = rng.choice(paints)
            ctx = JSON_REPLAY_TEMPLATES[1]["context"].format(
                pkg=pkg, price=price, cement=cem, steel=stl, wiring=wir, paint=pnt
            )
            inst = JSON_REPLAY_TEMPLATES[1]["instruction"]
            prompt = f"Context:\n{ctx}\n\nInstruction:\n{inst}"
            target_obj = {
                "type": "specs_card",
                "data": {
                    "Package_Name": pkg,
                    "Price_Per_SqFt": price,
                    "Cement_Grade": cem,
                    "Steel_Brand": stl,
                    "Electrical_Wiring": wir,
                    "Paint_Finish": pnt,
                },
            }
            answer = json.dumps(target_obj, indent=2, ensure_ascii=False)
        dataset.append((prompt, answer))

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
    print("  🚀 NORTHH DHRUVA 500M — STAGE 4 RAFT + CONVERSATIONAL FALLBACK TRAINER")
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
    print(f"  Tokenizer:     {tok_dir} (vocab={tokenizer.vocab_size:,})")
    print(f"  Dataset Pairs: {len(dataset):,} (RAFT={args.n_raft}, Abstain={args.n_abstain}, Greet={args.n_greet}, JSON={args.n_json})")

    if args.dry_run:
        sample_x, sample_y = dataset[0]
        masked_cnt = int((sample_y == -100).sum().item())
        active_cnt = int((sample_y != -100).sum().item())
        print(f"  [Dry Run] Sample 0 length={sample_x.size(0)}, masked(-100)={masked_cnt}, active_loss_tokens={active_cnt}")
        print("  [Dry Run] Dataset & ChatML masking verified cleanly. Exiting.")
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

    print(f"  Epochs:        {args.epochs} | Batch: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"  Optimizer:     AdamW (lr={args.lr}, cosine decay, grad_clip={args.grad_clip})")
    print(f"  Total Steps:   {total_optim_steps} optimizer steps")
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
            "stage": "stage4_raft_conversational",
            "raft_steps": optim_step,
            "dataset_size": len(dataset),
        },
        str(out_path),
    )
    print("-" * 76)
    print(f"  ✅ Saved Stage 4 RAFT checkpoint to: {out_path.resolve()}")
    print("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 RAFT + Conversational Fallback Trainer for Dhruva 500M")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_STAGE3_CKPT, help="Path to Stage 3 checkpoint")
    parser.add_argument("--output", type=str, default="dhruva_500m_raft_final.pt", help="Output Stage 4 checkpoint path")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tokenizer_32k", help="32K ByteBPETokenizer directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per step")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Peak AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping threshold")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max training sequence length")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable activation gradient checkpointing")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval in optimizer steps")
    parser.add_argument("--n_raft", type=int, default=1600, help="Number of grounded RAFT pairs")
    parser.add_argument("--n_abstain", type=int, default=450, help="Number of strict abstention pairs")
    parser.add_argument("--n_greet", type=int, default=600, help="Number of greeting/bypass pairs")
    parser.add_argument("--n_json", type=int, default=450, help="Number of JSON replay pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Build and validate dataset without training")
    return parser.parse_args()


if __name__ == "__main__":
    train_raft(parse_args())
