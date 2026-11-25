"""
generate_data.py
Creates a messy, synthetic email dataset (data/emails.csv) for the
rule-based spam detector. Uses a fixed seed so results are reproducible.

Columns: sender, subject, body, date, label   (label: ham / promotion / spam)

Messiness added on purpose:
  * mixed date formats (2025-03-01, 01/03/2025, 1-Mar-25)
  * invalid sender emails (john@@gmail.com, missing '@')
  * missing values, duplicate rows, extra whitespace
  * amounts in both Rs and $, random ALL-CAPS words, multiple '!!!'
"""
import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

N_EMAILS = 3000
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emails.csv")

# ------------------------------------------------------------------ pools
NAMES = ["rahul", "priya", "amit", "neha", "karan", "sneha", "vikas", "pooja",
         "arjun", "divya", "rohit", "anita", "manish", "kavya", "sanjay"]
HAM_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "college.edu",
               "company.com", "iimu.ac.in"]
PROMO_DOMAINS = ["shopzone.com", "fashionhub.in", "foodie.com", "travelbuddy.com",
                 "newsletter.medium.com", "deals.amazon.in", "offers.flipkart.com"]
SPAM_DOMAINS = ["win-prizes.xyz", "bank-secure-login.top", "free-gifts.click",
                "lucky-draw.biz", "account-verify.info", "gmail.com", "cashnow.xyz"]
SHORT_LINKS = ["http://bit.ly/x{}".format, "http://tinyurl.com/p{}".format]

HAM_SUBJECTS = [
    "Meeting at 3 PM", "Assignment submission", "Notes from yesterday's class",
    "Lunch tomorrow?", "Project report draft", "Re: schedule for Monday",
    "Invoice for last month", "Happy birthday!", "Group project update",
    "Please review the document", "Trip plan for the weekend", "Fwd: reading list",
]
HAM_BODIES = [
    "Hi, just a reminder that our meeting is at 3 PM in room 204. Please bring the report.",
    "Please find attached the assignment. Let me know if anything needs to change.",
    "Thanks for the notes. I will share my part of the project by Friday evening.",
    "Are you free for lunch tomorrow? We can try the new cafe near campus.",
    "The invoice of Rs 4,500 for last month is attached. Kindly confirm receipt.",
    "Hope you are doing well. Sharing the draft here, please review and comment.",
    "Happy birthday! Wishing you a great year ahead. Let us catch up soon.",
    "The deadline has moved to next week. Please update your timeline accordingly.",
    "I paid $20 for the tickets. We can settle it when we meet. See the link https://docs.google.com/d/abc for the plan.",
    "Sorry I missed your call. Can we talk after the FINAL lecture today?",
]

PROMO_SUBJECTS = [
    "Flat 20% off on shoes", "Weekly newsletter: top reads", "Your monthly deals are here",
    "Big Billion sale starts tonight", "New arrivals just for you",
    "Free shipping on orders above Rs 999", "Weekend offer: buy 1 get 1",
    "Your order has shipped", "Exclusive member discount",
]
PROMO_BODIES = [
    "Enjoy flat 20% off on all shoes this week. Shop now at https://shopzone.com/shoes",
    "Our weekly newsletter is out. Read the top stories at https://newsletter.medium.com/weekly",
    "SALE: new arrivals for the season. Orders above Rs 999 get free shipping! Visit https://fashionhub.in/new",
    "Buy 1 get 1 on selected items. Offer valid till Sunday. Details at https://offers.flipkart.com/bogo",
    "Hello member, here is your exclusive discount of $10 on your next order. https://deals.amazon.in/m",
    "Your order has shipped and will arrive soon. Track at https://shopzone.com/track Unsubscribe anytime.",
    "Hungry? Get 30% OFF on your next meal! Use code TASTY30 at https://foodie.com/order",
]

SPAM_SUBJECTS = [
    "WIN Rs 50,000 FREE!!!", "URGENT: verify your account", "Congratulations, you are a WINNER!!!",
    "Your bank account is suspended", "Claim your FREE gift now", "Lottery prize of $1,000,000 waiting",
    "Act now: limited offer!!!", "Final notice: payment pending", "You have won an iPhone",
]
SPAM_BODIES = [
    "Dear customer, your account will be blocked. URGENT action required, click here {link} to verify now!!!",
    "CONGRATULATIONS! You have been selected to WIN Rs 50,000. FREE entry, click here {link} today!!!",
    "You won a lottery of $1,000,000. Send your bank details at {link} to claim your PRIZE!!!",
    "Your bank has detected suspicious login. Verify your PASSWORD immediately: {link}",
    "Get a FREE gift card worth Rs 5,000. Limited time offer, click here {link}!!!",
    "URGENT: your KYC is expired. Update now at {link} or your account will be closed.",
    "Earn $500 per day working from home. No experience needed. Register at {link}",
]


# ------------------------------------------------------------------ helpers
def rand_name():
    return random.choice(NAMES) + random.choice(["", ".", "_"]) + str(random.randint(1, 99))


def make_sender(label):
    domains = {"ham": HAM_DOMAINS, "promotion": PROMO_DOMAINS, "spam": SPAM_DOMAINS}[label]
    if label == "promotion":
        return random.choice(["offers", "news", "hello", "deals"]) + "@" + random.choice(domains)
    return rand_name() + "@" + random.choice(domains)


def make_link(label):
    if label == "spam":
        if random.random() < 0.5:
            return random.choice(SHORT_LINKS)(random.randint(100, 999))
        return "http://" + random.choice(SPAM_DOMAINS[:4] + SPAM_DOMAINS[-1:]) + "/verify"
    return "https://" + random.choice(PROMO_DOMAINS) + "/offer"


def make_date():
    """Random 2025 timestamp written in one of three formats, plus HH:MM."""
    start = datetime(2025, 1, 1)
    d = start + timedelta(days=random.randint(0, 364),
                          hours=random.randint(0, 23), minutes=random.randint(0, 59))
    fmt = random.choice(["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y"])
    s = d.strftime(fmt)
    if fmt == "%d-%b-%y":
        s = s.lstrip("0")  # 1-Mar-25
    return s + " " + d.strftime("%H:%M")


def make_email(label):
    if label == "ham":
        subject = random.choice(HAM_SUBJECTS)
        body = random.choice(HAM_BODIES)
    elif label == "promotion":
        subject = random.choice(PROMO_SUBJECTS)
        body = random.choice(PROMO_BODIES)
    else:
        subject = random.choice(SPAM_SUBJECTS)
        body = random.choice(SPAM_BODIES).format(link=make_link("spam"))

    sender = make_sender(label)

    # ---- realistic overlap so the task is not trivially perfect
    r = random.random()
    if label == "spam" and r < 0.10:            # sneaky spam: soft wording
        subject = "Quick question about your account"
        body = "Hello, please confirm your details here " + make_link("spam")
    if label == "ham" and r < 0.05:             # ham that looks a bit shouty
        body += " URGENT: please reply today!"
    if label == "promotion" and r < 0.05:       # very aggressive promotion
        subject = subject.upper() + "!!!"
    if label == "spam" and random.random() < 0.15:
        sender = "support" + str(random.randint(1, 9)) + "@@bank-secure.top"   # invalid

    # ---- messiness
    if random.random() < 0.04:
        sender = random.choice([sender.replace("@", ""), sender.replace("@", "@@"),
                                sender.split("@")[0] + "@", ""])
    if random.random() < 0.05:
        subject = ""
    if random.random() < 0.03:
        body = ""
    date = make_date()
    if random.random() < 0.02:
        date = ""
    if random.random() < 0.10:
        subject = "  " + subject + "   "
        body = "  " + body + "  "
    return [sender, subject, body, date, label]


def main():
    labels = random.choices(["ham", "promotion", "spam"], weights=[0.45, 0.25, 0.30], k=N_EMAILS)
    rows = [make_email(lab) for lab in labels]
    for _ in range(int(0.03 * N_EMAILS)):          # duplicate rows
        rows.append(random.choice(rows)[:])
    random.shuffle(rows)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sender", "subject", "body", "date", "label"])
        w.writerows(rows)
    print(f"Generated {len(rows)} emails -> {OUT_FILE}")


if __name__ == "__main__":
    main()
