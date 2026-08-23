"""Generate a messy bilingual SMS CSV for demos, tests and benchmarks.

The generator injects the failure modes real exports have: mojibake, bidi
marks, Arabic-Indic digits, tashkeel, double spaces and truncated tails.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta

BANKS = ["alrajhi", "snb", "riyad bank", "الراجحي", "الأهلي", "بنك الرياض"]
MERCHANTS = [
    "amazon.sa", "jarir bookstore", "starbucks", "carrefour hyper",
    "نون", "هنقرستيشن", "بنده", "صيدلية النهدي",
]
CURRENCIES = ["SAR", "AED", "USD", "ر.س"]

TEMPLATES = [
    "Purchase of {cur} {amount} at {merchant} on card ending {card} on {date} {time}. Available balance {cur} {balance}",
    "Transfer of {cur} {amount} to {name} completed. Ref {ref}. Balance {cur} {balance}",
    "Your OTP is {otp}. Valid for {mins} minutes. Do not share it with anyone",
    "ATM withdrawal {cur} {amount} from account {acct} on {date}. Available balance {cur} {balance}",
    "Salary of {cur} {amount} credited to account {acct} on {date}",
    "Bill payment {cur} {amount} to {merchant} succeeded. Fee {cur} {fee}. Ref {ref}",
    "Your card {card} was declined at {merchant} due to insufficient funds",
    "عملية شراء بمبلغ {amount} {cur} لدى {merchant} بالبطاقة {card} بتاريخ {date} الرصيد المتاح {balance}",
    "تم تحويل مبلغ {amount} {cur} الى {name} رقم المرجع {ref} الرصيد {balance}",
    "رمز التحقق الخاص بك هو {otp} صالح لمدة {mins} دقائق لا تشاركه مع احد",
    "سحب نقدي {amount} {cur} من الحساب {acct} بتاريخ {date} الرصيد المتاح {balance}",
    "تم ايداع راتب بمبلغ {amount} {cur} في الحساب {acct} بتاريخ {date}",
    "خصم رسوم {fee} {cur} على الحساب {acct} رقم العملية {ref}",
    "عزيزنا العميل تم تفعيل خدمة {service} على حسابك {acct} للاستفسار اتصل على {phone}",
    "Dear customer, service {service} was activated on account {acct}. Call {phone} for help",
]

NAMES = ["mohammed ali", "sara a.", "خالد العتيبي", "نورة الشمري", "acme trading llc"]
SERVICES = ["sms alerts", "apple pay", "الحوالات الفورية", "المدفوعات الدولية"]


def _mojibake(text: str) -> str:
    return text.encode("utf-8").decode("cp1252", errors="ignore")


def _to_arabic_digits(text: str) -> str:
    return text.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


def _mess(text: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.08:
        head, sep, tail = text.partition(" ")
        text = head + sep + _mojibake(tail[: len(tail) // 2]) + tail[len(tail) // 2 :]
    if roll > 0.92:
        text = _to_arabic_digits(text)
    if rng.random() < 0.15:
        text = text.replace(" ", "  ", 1)
    if rng.random() < 0.1:
        text = "\u200f" + text + "\u200e"
    if rng.random() < 0.07:
        text = text + " " + rng.choice(["شكرا لك", "thank you", "-" + rng.choice(BANKS)])
    if rng.random() < 0.05:
        text = text[: int(len(text) * 0.8)]
    return text


def generate(count: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    start = datetime(2024, 1, 1)
    rows = []
    for i in range(count):
        template = rng.choice(TEMPLATES)
        when = start + timedelta(minutes=rng.randint(0, 500_000))
        text = template.format(
            cur=rng.choice(CURRENCIES),
            amount=f"{rng.uniform(5, 90_000):,.2f}",
            balance=f"{rng.uniform(50, 500_000):,.2f}",
            fee=f"{rng.uniform(1, 100):.2f}",
            merchant=rng.choice(MERCHANTS),
            card=rng.choice(["****" + str(rng.randint(1000, 9999)), "xxxx" + str(rng.randint(1000, 9999))]),
            acct="***" + str(rng.randint(100, 999)),
            date=when.strftime(rng.choice(["%d/%m/%Y", "%Y-%m-%d"])),
            time=when.strftime("%H:%M"),
            ref=f"{rng.randint(10**8, 10**9 - 1)}",
            otp=str(rng.randint(1000, 999999)),
            mins=rng.choice([5, 10, 15]),
            name=rng.choice(NAMES),
            service=rng.choice(SERVICES),
            phone=f"9665{rng.randint(10**7, 10**8 - 1)}",
        )
        rows.append(
            {
                "id": i,
                "sender": rng.choice(BANKS),
                "received_at": when.isoformat(),
                "text": _mess(text, rng),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--count", type=int, default=5000)
    parser.add_argument("-o", "--out", default="examples/sms_sample.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = generate(args.count, args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
