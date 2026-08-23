"""Generate a messy bilingual multi-domain SMS CSV for demos and benchmarks.

The generator injects the failure modes real exports have: mojibake, bidi
marks, Arabic-Indic digits, tashkeel, double spaces and truncated tails.

Finance templates coexist with e-commerce shipping, travel, support, 2FA and
log alerts to prove the learner is domain-agnostic and only uses the words
that already appear in the messages.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta

SENDERS = [
    "alrajhi", "snb", "riyad bank", "الراجحي", "الأهلي", "بنك الرياض",
    "noon", "amazon.sa", "saudia", "flynas", "support", "nocs",
]
MERCHANTS = [
    "amazon.sa", "jarir bookstore", "starbucks", "carrefour hyper",
    "نون", "هنقرستيشن", "بنده", "صيدلية النهدي",
]
CURRENCIES = ["SAR", "AED", "USD", "ر.س"]

FLIGHTS = ["SV102", "EK413", "XY201", "TK123", "QR890"]
CITIES = ["riyadh", "jeddah", "dubai", "cairo", "dammam", "الرياض", "جده"]
HOSTS = ["srv-prod-07", "api-01", "db-master", "worker-03"]
LOCATIONS = ["clinic a", "tower 2", "gate 5", "branch 12", "عيادة الأمل"]
URLS = [
    "https://track.example.com/abc123",
    "https://support.example.com/ticket/xyz",
    "https://noon.com/orders",
]
STATUSES = ["confirmed", "delayed", "cancelled", "shipped", "delivered"]
AGENTS = ["ahmed k.", "sara m.", "خالد العتيبي"]
NAMES = ["mohammed ali", "sara a.", "خالد العتيبي", "نورة الشمري", "acme trading llc"]
SERVICES = ["sms alerts", "apple pay", "الحوالات الفورية", "المدفوعات الدولية"]

TEMPLATES = [
    # finance
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
    # e-commerce / travel / support / logs (multi-domain)
    "Your order {ref} has been {status}. Track at {url} Estimated delivery {date}",
    "Booking {ref} confirmed. Flight {flight} from {origin} to {destination} on {date} at {time}. Passenger {name}",
    "Ticket {ref} updated. Agent {agent} replied: {text}",
    "Your verification code is {otp}. Login from ip {ip} at {time}",
    "ALERT host {host} cpu {percent}% at {time} service {service} status {status}",
    "Appointment reminder for {name} at {time} on {date}. Location {location}",
    "تم تأكيد حجزك {ref} رحلة {flight} من {origin} الى {destination} بتاريخ {date} الساعه {time}",
    "طلبيتك {ref} في الطريق الى {location} التوصيل المتوقع {date}",
]


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
        text = text + " " + rng.choice(["شكرا لك", "thank you", "-" + rng.choice(SENDERS)])
    if rng.random() < 0.05:
        text = text[: int(len(text) * 0.8)]
    return text


def _pick(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


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
            flight=rng.choice(FLIGHTS),
            origin=rng.choice(CITIES),
            destination=rng.choice(CITIES),
            host=rng.choice(HOSTS),
            location=rng.choice(LOCATIONS),
            url=rng.choice(URLS),
            status=rng.choice(STATUSES),
            agent=rng.choice(AGENTS),
            percent=str(rng.randint(1, 99)),
            ip=f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
            text=rng.choice([
                "we are looking into it",
                "please try again now",
                "issue resolved on our end",
                "شكرا على تواصلك معنا",
            ]),
        )
        rows.append(
            {
                "id": i,
                "sender": rng.choice(SENDERS),
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
