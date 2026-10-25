"""Generate the demo knowledge base (PDF, CSV, Bangla text, markdown, test set).

The generated files live in ``data/samples`` and are also used by the pytest
suite and the RAGAS evaluation, so the whole project is reproducible from a
clean checkout:

    python scripts/generate_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
EVAL_DIR = BASE_DIR / "data" / "evaluation"

FOOTER = "Acme Bangladesh Ltd  |  Confidential  |  Internal use only"

HANDBOOK_PAGES: list[tuple[str, list[str]]] = [
    (
        "1. Company Overview",
        [
            "Acme Bangladesh Ltd is a Dhaka-based enterprise software company founded in 2016.",
            "The company headquarters is located in Gulshan, Dhaka, and it operates a second office in Chattogram.",
            "Acme Bangladesh employs 240 people across engineering, sales, support and corporate functions.",
            "Acme Bangladesh owns two subsidiaries: Acme Cloud Services, which was launched in 2019 in Chattogram, "
            "and Acme Labs, which focuses on applied machine learning research.",
            "The company is ISO 27001 certified and works with partners such as Grameen Digital and BRAC IT.",
        ],
    ),
    (
        "2. Leave and Absence Policy",
        [
            "Every permanent employee is entitled to 20 days of annual leave per calendar year.",
            "Employees also receive 10 days of sick leave and 14 weeks of maternity leave; paternity leave is 2 weeks.",
            "A maximum of 5 unused annual leave days may be carried over to the following year.",
            "Leave requests must be submitted through the People Operations portal at least 14 days in advance.",
            "Emergency leave requires approval from the reporting manager within 48 hours.",
        ],
    ),
    (
        "3. Working Hours and Remote Work",
        [
            "Core working hours are 10:00 to 16:00 Bangladesh Standard Time, Monday to Thursday.",
            "Employees may work remotely up to 3 days per week, subject to team coverage requirements.",
            "New joiners are not eligible for remote work during their first 90 days of employment.",
            "A home internet allowance of BDT 3,000 per month is paid to employees on a remote contract.",
            "Fridays and Saturdays are the weekly rest days; Sunday is a normal working day.",
        ],
    ),
    (
        "4. Information Security",
        [
            "Passwords must contain at least 12 characters and must be rotated every 180 days.",
            "Multi-factor authentication is mandatory for all company accounts and VPN access.",
            "Security incidents must be reported to security@acme.com.bd within 24 hours of discovery.",
            "Company laptops are encrypted and managed by the IT department; installing unapproved software is prohibited.",
            "Personal USB storage devices must never be connected to company devices.",
        ],
    ),
    (
        "5. Expenses and Travel",
        [
            "Expense claims must be submitted within 30 days of the expense date with itemised receipts.",
            "Expenses above BDT 10,000 require prior approval from the reporting manager.",
            "The daily travel per diem is BDT 2,500 for Dhaka and BDT 2,000 for Chattogram.",
            "Client entertainment expenses require approval from a department head.",
            "Reimbursements are processed with the monthly payroll run.",
        ],
    ),
    (
        "6. Learning and Performance",
        [
            "Each employee receives an annual learning budget of BDT 50,000 for courses and certifications.",
            "Performance reviews are conducted twice a year, in June and December.",
            "Promotion decisions are announced in July and January.",
            "Professional certification fees are reimbursed after successful completion.",
            "Managers must complete a leadership training programme within 12 months of appointment.",
        ],
    ),
    (
        "7. Data and Confidentiality",
        [
            "Client data is classified as Confidential and may only be stored in approved systems.",
            "Customer records must be retained for 7 years and then securely destroyed.",
            "Employees must never share client data with third parties without written approval.",
            "Access to production databases requires approval from the Data Governance Committee.",
            "Data protection questions should be sent to the Data Protection Officer.",
        ],
    ),
]

BANGLA_POLICY = """আসমে বাংলাদেশ লিমিটেড — মানবসম্পদ নীতিমালা (সারসংক্ষেপ)

আসমে বাংলাদেশ লিমিটেড ২০১৬ সালে ঢাকায় প্রতিষ্ঠিত হয়। কোম্পানির প্রধান কার্যালয় গুলশান, ঢাকায় অবস্থিত এবং দ্বিতীয় কার্যালয় চট্টগ্রামে অবস্থিত। কোম্পানিতে মোট ২৪০ জন কর্মী কাজ করেন।

ছুটি সংক্রান্ত নীতিমালা:
প্রতি ক্যালেন্ডার বছরে প্রত্যেক স্থায়ী কর্মী ২০ দিন বার্ষিক ছুটি পাবেন। অসুস্থতার জন্য ১০ দিন এবং মাতৃত্বকালীন ছুটি ১৪ সপ্তাহ পাওয়া যায়। পিতৃত্বকালীন ছুটি ২ সপ্তাহ। বার্ষিক ছুটির সর্বোচ্চ ৫ দিন পরের বছরে বহন করা যায়।

কাজের সময় ও রিমোট কাজ:
মূল কাজের সময় সকাল ১০টা থেকে বিকাল ৪টা পর্যন্ত। কর্মীরা সপ্তাহে সর্বোচ্চ ৩ দিন রিমোট কাজ করতে পারেন। নতুন কর্মীদের প্রথম ৯০ দিন রিমোট কাজের সুযোগ নেই। রিমোট কাজের জন্য মাসে ৩,০০০ টাকা ইন্টারনেট ভাতা দেওয়া হয়।

শিক্ষা ও মূল্যায়ন:
প্রতি কর্মীর জন্য বছরে ৫০,০০০ টাকার শিক্ষা বাজেট রয়েছে। বার্ষিক মূল্যায়ন ডিসেম্বর মাসে অনুষ্ঠিত হয় এবং পদোন্নতির ঘোষণা জানুয়ারি মাসে দেওয়া হয়।

তথ্য নিরাপত্তা:
পাসওয়ার্ডে অন্তত ১২টি অক্ষর থাকতে হবে এবং প্রতি ১৮০ দিনে পাসওয়ার্ড পরিবর্তন করতে হবে। নিরাপত্তা ঘটনা নিশ্চিত হওয়ার ২৪ ঘণ্টার মধ্যে security@acme.com.bd ঠিকানায় জানাতে হবে। ক্লায়েন্টের তথ্য ৭ বছর সংরক্ষণ করতে হবে।

ভ্রমণ ভাতা:
ঢাকার জন্য দৈনিক ভাতা ২,৫০০ টাকা এবং চট্টগ্রামের জন্য ২,০০০ টাকা। ১০,০০০ টাকার বেশি খরচে রিপোর্টিং ম্যানেজারের পূর্বানুমোদন প্রয়োজন।
"""

EMPLOYEES_CSV = """employee_id,name,department,role,location,manager,email,join_date,salary_band
E001,Nusrat Jahan,People Operations,HR Director,Dhaka,Rafiq Ahmed,nusrat.jahan@acme.com.bd,2017-03-01,B5
E002,Rafiq Ahmed,Executive,Chief Executive Officer,Dhaka,Board,rafiq.ahmed@acme.com.bd,2016-01-05,E1
E003,Tanvir Hasan,Engineering,Head of Platform,Dhaka,Nusrat Jahan,tanvir.hasan@acme.com.bd,2018-07-16,D3
E004,Sadia Islam,Engineering,Senior Data Engineer,Chattogram,Tanvir Hasan,sadia.islam@acme.com.bd,2020-02-10,C4
E005,Mahmudul Karim,Security,Information Security Manager,Dhaka,Tanvir Hasan,mahmudul.karim@acme.com.bd,2019-11-03,D2
E006,Fariha Chowdhury,Finance,Finance Manager,Dhaka,Rafiq Ahmed,fariha.chowdhury@acme.com.bd,2018-01-22,C5
E007,Imran Hossain,Sales,Regional Sales Lead,Chattogram,Rafiq Ahmed,imran.hossain@acme.com.bd,2021-05-17,C3
E008,Shirin Akter,People Operations,People Operations Partner,Dhaka,Nusrat Jahan,shirin.akter@acme.com.bd,2022-08-01,B3
E009,Arif Mahmud,Support,Customer Support Lead,Chattogram,Imran Hossain,arif.mahmud@acme.com.bd,2020-09-14,B4
E010,Mehjabin Rahman,Acme Labs,Research Scientist,Dhaka,Tanvir Hasan,mehjabin.rahman@acme.com.bd,2023-01-09,C4
E011,Kamal Uddin,IT,IT Operations Engineer,Dhaka,Mahmudul Karim,kamal.uddin@acme.com.bd,2021-10-25,B3
E012,Sumaiya Noor,Finance,Payroll Accountant,Dhaka,Fariha Chowdhury,sumaiya.noor@acme.com.bd,2022-04-11,B2
"""

PRODUCTS_CSV = """product,category,price_bdt,support_window,sla_hours,description
Atlas Analytics Suite,Analytics,450000,24x5,8,Self-service analytics platform with Bengali dashboards and governed data models
Atlas Data Fabric,Data Platform,780000,24x7,4,Managed ingestion and catalog layer for multi-source enterprise data
Nimbus OCR,Document AI,220000,24x5,8,Bangla and English OCR service for scanned documents and forms
Nimbus Chat,Conversational AI,310000,24x7,4,Retrieval augmented assistant for internal knowledge bases
Sentinel Compliance,Governance,190000,24x5,12,Policy and audit tracking with evidence collection workflows
Mridul Vision,Computer Vision,260000,24x5,8,Quality inspection models for manufacturing lines
"""

OVERVIEW_MD = """# Acme Bangladesh Ltd — Company Overview

Acme Bangladesh Ltd is an enterprise software company headquartered in Gulshan, Dhaka.
It was founded in 2016 by Rafiq Ahmed and Nusrat Jahan.

## Business units
- **Acme Cloud Services** — managed cloud and data platform business, launched in 2019, based in Chattogram.
- **Acme Labs** — applied machine learning research unit in Dhaka, created in 2021.

## Partnerships and certifications
Acme Bangladesh partners with Grameen Digital for rural connectivity and with BRAC IT for
financial services integrations. The company achieved ISO 27001 certification in 2022.

## Customers
The company serves 45 enterprise customers in banking, telecom, retail and manufacturing.
The largest customer segment is banking, which contributes 38 percent of revenue.

## Support model
Enterprise support follows the SLA attached to each product. Severity-1 incidents are
acknowledged within 4 hours for products with 24x7 support coverage.
"""

TEST_SET = [
    {
        "question": "How many days of annual leave do employees get?",
        "ground_truth": "Every permanent employee is entitled to 20 days of annual leave per calendar year.",
        "source": "employee_handbook.pdf",
        "language": "en",
    },
    {
        "question": "What are the core working hours?",
        "ground_truth": "Core working hours are 10:00 to 16:00 Bangladesh Standard Time, Monday to Thursday.",
        "source": "employee_handbook.pdf",
        "language": "en",
    },
    {
        "question": "Who must approve expenses above BDT 10,000?",
        "ground_truth": "Expenses above BDT 10,000 require prior approval from the reporting manager.",
        "source": "employee_handbook.pdf",
        "language": "en",
    },
    {
        "question": "Where is Acme Cloud Services located and when was it launched?",
        "ground_truth": "Acme Cloud Services was launched in 2019 and is based in Chattogram.",
        "source": "company_overview.md",
        "language": "en",
    },
    {
        "question": "What is the daily travel per diem for Chattogram?",
        "ground_truth": "The daily travel per diem is BDT 2,000 for Chattogram.",
        "source": "employee_handbook.pdf",
        "language": "en",
    },
    {
        "question": "Which department does Sadia Islam work in and where is she based?",
        "ground_truth": "Sadia Islam works in Engineering as a Senior Data Engineer and is based in Chattogram.",
        "source": "employees.csv",
        "language": "en",
    },
    {
        "question": "বার্ষিক ছুটি কত দিন?",
        "ground_truth": "প্রতি ক্যালেন্ডার বছরে প্রত্যেক স্থায়ী কর্মী ২০ দিন বার্ষিক ছুটি পাবেন।",
        "source": "hr_policies_bangla.txt",
        "language": "bn",
    },
    {
        "question": "রিমোট কাজ সপ্তাহে কত দিন করা যায়?",
        "ground_truth": "কর্মীরা সপ্তাহে সর্বোচ্চ ৩ দিন রিমোট কাজ করতে পারেন।",
        "source": "hr_policies_bangla.txt",
        "language": "bn",
    },
    {
        "question": "শিক্ষা বাজেট কত টাকা?",
        "ground_truth": "প্রতি কর্মীর জন্য বছরে ৫০,০০০ টাকার শিক্ষা বাজেট রয়েছে।",
        "source": "hr_policies_bangla.txt",
        "language": "bn",
    },
    {
        "question": "What is the capital of France?",
        "ground_truth": "The provided documents do not contain this information.",
        "source": None,
        "language": "en",
        "unanswerable": True,
    },
]


def build_pdf(path: Path) -> None:
    """Render the handbook PDF with PyMuPDF (ASCII fonts only)."""
    import pymupdf

    document = pymupdf.open()
    for page_number, (heading, bullets) in enumerate(HANDBOOK_PAGES, start=1):
        page = document.new_page(width=595, height=842)  # A4
        y = 80.0
        page.insert_text((60, y), "Acme Bangladesh Ltd — Employee Handbook", fontsize=16, fontname="hebo")
        y += 34
        page.insert_text((60, y), heading, fontsize=13, fontname="hebo")
        y += 26

        for bullet in bullets:
            words = bullet.split()
            line = ""
            lines: list[str] = []
            for word in words:
                candidate = f"{line} {word}".strip()
                if len(candidate) > 88:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            if line:
                lines.append(line)
            page.insert_text((68, y), "•", fontsize=11, fontname="helv")
            for offset, text_line in enumerate(lines):
                page.insert_text((82, y + offset * 16), text_line, fontsize=11, fontname="helv")
            y += len(lines) * 16 + 10

        page.insert_text((60, 800), f"{FOOTER}  |  Page {page_number} of {len(HANDBOOK_PAGES)}", fontsize=8)
    document.set_metadata(
        {
            "title": "Acme Bangladesh Employee Handbook",
            "author": "People Operations",
            "subject": "HR policies",
        }
    )
    document.save(path)
    document.close()


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    build_pdf(SAMPLES_DIR / "employee_handbook.pdf")
    (SAMPLES_DIR / "hr_policies_bangla.txt").write_text(BANGLA_POLICY, encoding="utf-8")
    (SAMPLES_DIR / "employees.csv").write_text(EMPLOYEES_CSV, encoding="utf-8")
    (SAMPLES_DIR / "product_catalog.csv").write_text(PRODUCTS_CSV, encoding="utf-8")
    (SAMPLES_DIR / "company_overview.md").write_text(OVERVIEW_MD, encoding="utf-8")
    (EVAL_DIR / "sample_testset.json").write_text(
        json.dumps(TEST_SET, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Generated sample data:")
    for file in sorted(SAMPLES_DIR.iterdir()):
        print(f"  - {file.relative_to(BASE_DIR)} ({file.stat().st_size} bytes)")
    print(f"  - {(EVAL_DIR / 'sample_testset.json').relative_to(BASE_DIR)} ({len(TEST_SET)} questions)")


if __name__ == "__main__":
    main()
