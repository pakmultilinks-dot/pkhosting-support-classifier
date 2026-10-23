#!/usr/bin/env python3
"""
Generate a SYNTHETIC dataset of 300 PKHosting-style support queries.

Fully synthetic: every query is written from templates with randomized slots
(fictional domains, plan names, amounts). No real customer data is used.
Each of the 10 categories gets 30 queries: 24 from "core" phrasings and 6 from
"held-out" phrasings that are reserved for the test split, so evaluation also
measures how the classifier handles wording it never saw in training.

Output: data/support_queries.csv  (id, query, category, language, template_group)
Reproducible: fixed seed 42.
"""
import csv
import os
import random

random.seed(42)

DOMAINS = ["mybusiness.pk", "shopease.pk", "lahorefoods.com", "techmart.pk",
           "alnoorstore.pk", "desiwear.com.pk", "brighttutor.pk", "citypharmacy.pk",
           "greenfabrics.com", "studyhub.pk", "royalcaterers.pk", "fitgear.pk"]
PLANS = ["Starter", "Basic", "Business", "Premium", "Economy"]
AMOUNTS = ["1,999", "2,499", "3,499", "4,999", "6,500", "8,999", "12,000"]
CLIENTS = ["Outlook", "Gmail", "Thunderbird", "my phone mail app", "Apple Mail"]
GREET = ["", "", "", "Hi team, ", "Hello support, ", "Assalam o Alaikum, ", "Dear PKHosting team, "]
CLOSE = ["", "", " Please help.", " Kindly resolve this soon.", " Thanks.",
         " Please reply urgently.", " JazakAllah.", " Waiting for your reply."]

# Each category: (core templates, held-out templates). {d}=domain {p}=plan {a}=amount {c}=mail client
CATEGORIES = {
    "billing_invoices": (
        [
            "I was charged PKR {a} twice for my {p} hosting this month. Please check my billing.",
            "Can you send me a copy of my latest invoice? I need it for my records.",
            "My invoice shows a renewal fee of PKR {a} but my plan was supposed to be cheaper. Why?",
            "How do I update the card I use for payments on my account?",
            "Do you accept JazzCash or EasyPaisa for hosting renewal payment?",
            "I paid through bank transfer yesterday but my invoice still shows unpaid. Please confirm.",
            "Where can I download my payment receipts from the client area?",
            "My hosting renewal is due next week. What is the exact amount I have to pay?",
            "I see an extra charge on my bill that I do not understand. Please explain it.",
            "Can I get a discount if I pay for two years of hosting in advance?",
            "The payment gateway rejected my card three times. How else can I pay my invoice?",
            "Meray account mein invoice generate ho gaya hai lekin amount ghalat lag raha hai, check kar dein.",
        ],
        [
            "Please share the payable total for this month including taxes.",
            "I need a proper tax invoice with my company name on it for {d}.",
            "Why did the renewal price increase compared to last year?",
            "My payment is stuck at pending since morning, invoice number is unpaid yet.",
            "Can I split my annual hosting bill into monthly payments?",
            "Bill ki copy email kar dein, mujhe accounts department ko deni hai.",
        ]),
    "refunds_cancellations": (
        [
            "I want to cancel my {p} hosting plan and get a refund. I bought it 10 days ago.",
            "Please cancel the hosting for {d} and refund the remaining amount.",
            "How long does a refund take to reach my bank account?",
            "I was promised a 30 day money back guarantee. I want to claim it.",
            "Cancel my subscription please, I am moving my website to another provider.",
            "I accidentally renewed for a year. Can I cancel and get my money back?",
            "What is your refund policy for domain registrations?",
            "Please stop my auto renewal and refund the last charge of PKR {a}.",
            "I no longer need this hosting account. Kindly process a full refund.",
            "The service did not work for my project, so I want to discontinue and claim a refund.",
            "Mujhe hosting cancel karni hai aur paisay wapas chahiyein, procedure bata dein.",
            "Close my account for {d} and tell me how much refund I will get.",
        ],
        [
            "I would like to withdraw from the service and request my payment back.",
            "Terminate this plan today. Also confirm in writing when the reimbursement is done.",
            "Is a partial refund possible if I cancel halfway through the year?",
            "Mera plan cancel kar dein, mujhe refund kab tak milega?",
            "I changed my mind about the purchase, please reverse the transaction.",
            "Do I get anything back if I shut down the account before expiry?",
        ]),
    "domain_dns": (
        [
            "I want to register the domain name for my new shop. Is the .pk extension available?",
            "How do I point my domain {d} to my new hosting account?",
            "I changed the nameservers yesterday but my site still shows the old page. How long does DNS take?",
            "Please help me add an MX record so my email works with Google Workspace.",
            "My domain expired last week. Can I still renew it or is it lost?",
            "How can I transfer my domain from another registrar to PKHosting?",
            "I need to add a CNAME record for my subdomain shop.{d}.",
            "The DNS records for {d} are not propagating. It has been 24 hours.",
            "What are your nameservers? I need to update them at my registrar.",
            "Can I renew my domain for 5 years at once? What will it cost?",
            "Domain ki expiry date check kar dein, renewal kab hai {d} ki?",
            "I want to create a subdomain blog.{d} for my website, how do I do that?",
        ],
        [
            "My domain shows a registrar lock when I try to move it. Please unlock it.",
            "Please verify my TXT record for domain ownership is set correctly.",
            "The A record I added points to the wrong IP. How do I correct it?",
            "My www address works but the address without www does not. Fix the DNS please.",
            "Is domain privacy protection included or do I pay extra for it?",
            "Nameserver change kar diya hai lekin website purani wali khul rahi hai abhi tak.",
        ]),
    "email_problems": (
        [
            "My business email is not receiving any new emails since yesterday.",
            "Emails I send from {d} go straight to the spam folder of my clients.",
            "How do I set up my email account in {c}?",
            "My mailbox is full and I cannot receive emails. How do I increase the quota?",
            "I forgot the password of my email account info@{d}. Please reset it.",
            "Outgoing emails bounce back with an error about the server. Please check.",
            "How many email accounts can I create on the {p} plan?",
            "Webmail loads very slowly and sometimes gives a blank page.",
            "I need to create a new email address for our sales team on {d}.",
            "My emails are delayed by hours. Customers are complaining.",
            "Email aa nahi rahi meri, inbox khali hai do din se. Check karein please.",
            "The mail server is rejecting my login even though the password is correct.",
        ],
        [
            "Please help me configure SPF and DKIM so my mails are not marked spam.",
            "I cannot send attachments bigger than a few MB. What is the limit?",
            "My email works on webmail but not in {c}. What settings should I use?",
            "Someone is sending spam from my domain. Please secure my email accounts.",
            "Auto reply set karna hai meri email par chuttiyon ke liye, tareeqa bata dein.",
            "All my old emails disappeared from the inbox after I changed phones.",
        ]),
    "website_down_errors": (
        [
            "My website {d} is down. It shows a 500 internal server error.",
            "My site is not opening at all, just a blank white screen.",
            "I get 'Error establishing a database connection' on my WordPress site.",
            "The website was fine in the morning and now it shows 502 bad gateway.",
            "My homepage loads but all inner pages give a 404 error.",
            "Customers cannot access my online store, the site keeps timing out.",
            "There is a 'This site can't be reached' message when I open {d}.",
            "My WordPress admin page is broken after an update. Site shows critical error.",
            "The site shows 'Bandwidth limit exceeded'. What does that mean?",
            "Meri website khul nahi rahi, error aa raha hai. Jaldi check karein please.",
            "After installing a plugin my whole website crashed. Please restore it.",
            "Is your server down? All my hosted sites are unreachable right now.",
        ],
        [
            "Visitors see a security warning and the page refuses to load completely.",
            "My site redirects to a strange page that is not mine. It worked yesterday.",
            "Pages show 503 service unavailable every evening around 8 pm.",
            "The checkout page of my store returns an error and orders are lost.",
            "Website band ho gayi hai bilkul, customers call kar rahe hain. Kuch karein.",
            "My site loads the hosting default page instead of my content.",
        ]),
    "setup_migration": (
        [
            "I just bought the {p} plan. How do I start building my website?",
            "Can you migrate my WordPress site from my old host to PKHosting for free?",
            "How do I install WordPress on my new hosting account?",
            "I need step by step guidance to upload my website files using FTP.",
            "My developer built the site on his computer. How do we put it on my hosting?",
            "Where is my cPanel and how do I create my first website on {d}?",
            "Can you move my site and emails together without any downtime?",
            "I have a backup file of my old website. Please help me restore it here.",
            "How do I connect my existing domain to this new hosting plan?",
            "What do I need to provide for a migration? My old host uses cPanel too.",
            "Naya hosting liya hai, website banana start karna hai. Guide kar dein.",
            "Please set up a staging copy of my site so I can test changes safely.",
        ],
        [
            "I want to shift two websites from another company to one account here.",
            "Is there a website builder included or should I use WordPress?",
            "Upload karna hai mujhe apni files, FileZilla ki settings bata dein.",
            "How long does a full migration usually take for a small business site?",
            "Can your team install WooCommerce and a theme for my store?",
            "I purchased hosting today. What is the first thing I should do in cPanel?",
        ]),
    "ssl_security": (
        [
            "My website shows 'Not secure' in the browser. How do I get an SSL certificate?",
            "The SSL certificate for {d} expired yesterday. Please renew it.",
            "How do I install the free SSL on my hosting account?",
            "My site was hacked and now shows strange Japanese pages on Google.",
            "Chrome blocks my site saying the connection is not private. Please fix it.",
            "I need an SSL certificate for my online store checkout page.",
            "Google marked my website as dangerous. How can I clean it?",
            "The padlock icon disappeared after I moved my site. HTTPS is not working.",
            "Do you offer malware scanning and removal for hacked websites?",
            "My site redirects to a spam site sometimes. I think it is infected.",
            "SSL lagana hai meri website par, bina is ke customers trust nahi karte.",
            "Please force HTTPS on {d}, the http version still opens.",
        ],
        [
            "The certificate on my subdomain is invalid while the main domain is fine.",
            "My site is on a blacklist. What steps remove it?",
            "The SSL certificate chain on {d} shows as incomplete in online checkers.",
            "My website was hacked last night and Google now shows a red warning page to visitors.",
            "Padlock nahi aa raha browser mein, certificate ka issue lagta hai.",
            "How do I check whether my WordPress site has malware right now?",
        ]),
    "performance_speed": (
        [
            "My website has become very slow. Pages take 10 seconds to load.",
            "The hosting shows high CPU usage and my site keeps getting suspended.",
            "My WordPress site speed score is very poor. How can I improve it on this hosting?",
            "Images on my store load slowly and customers leave the site.",
            "My account reached its resource limit again. What is using so much?",
            "The site is fast at night but very slow during the day. Why?",
            "How do I enable caching for my website on your hosting?",
            "My admin dashboard takes forever to save any change.",
            "I get 'resource limit reached' errors several times a week on the {p} plan.",
            "Can you check what is slowing down {d} from the server side?",
            "Website bohat slow ho gayi hai, koi check kar ke bata dein wajah.",
            "My traffic increased and now the site cannot handle visitors. Options?",
        ],
        [
            "Page loading improved on mobile but desktop is still sluggish.",
            "Pages of my store load slowly for visitors outside Pakistan. Would a CDN make them faster?",
            "Database queries on my store are timing out during peak hours.",
            "Speed test shows a very high server response time. Please investigate.",
            "Site ki speed barhani hai, upgrade ke baghair koi hal hai?",
            "My site uses too much RAM according to cPanel. Which plugin is responsible?",
        ]),
    "account_login": (
        [
            "I forgot my cPanel password and cannot log in. Please reset it.",
            "My client area login is not working even with the correct password.",
            "I am locked out of my account after too many login attempts.",
            "How do I change the email address registered on my account?",
            "Someone else manages my website. How do I give him access to cPanel?",
            "I cannot receive the password reset email for my account.",
            "My hosting account was suspended. How do I get it reactivated?",
            "Where do I find my FTP login details in the client area?",
            "Two factor authentication is enabled and I lost my phone. Help me log in.",
            "Please update my phone number and address in my profile.",
            "cPanel ka password bhool gaya hoon, reset kar dein please.",
            "My account shows 'suspended for billing' but I already paid. Restore it.",
        ],
        [
            "Login page keeps refreshing and never lets me into the dashboard.",
            "I want to transfer my hosting account ownership to my business partner.",
            "Reset link expire ho jata hai har bar, naya tareeqa bata dein login ka.",
            "Why was my account flagged for verification after signup?",
            "I need to merge two accounts I created by mistake into one.",
            "The username in my welcome email does not work for cPanel sign in.",
        ]),
    "plans_upgrades": (
        [
            "What is the difference between your {p} and Business hosting plans?",
            "How much disk space do I get on the cheapest shared hosting plan?",
            "My business is growing. How do I upgrade my current hosting plan?",
            "Do your plans include a free domain for the first year?",
            "I run a small online store. Which hosting plan do you recommend?",
            "What are your current prices for shared hosting in PKR?",
            "Can I host two websites on one plan or do I need separate accounts?",
            "Do you offer monthly billing or only yearly plans?",
            "My site needs more storage. What does an upgrade cost?",
            "Is email hosting included in all plans or only the higher ones?",
            "Sab se sasta hosting plan konsa hai aap ke paas choti website ke liye?",
            "If I upgrade today, will my website face any downtime?",
        ],
        [
            "I am comparing hosts. Why should I choose PKHosting over others?",
            "Does the Business plan support more traffic than my current one?",
            "Plan upgrade karna hai lekin data loss ka dar hai, confirm kar dein.",
            "Are there any discounts for students or new businesses right now?",
            "How many visitors per month can the {p} plan handle roughly?",
            "I need hosting for a school website with 500 students. Suggest a plan.",
        ]),
}

def fill(t):
    return (t.replace("{d}", random.choice(DOMAINS))
             .replace("{p}", random.choice(PLANS))
             .replace("{a}", random.choice(AMOUNTS))
             .replace("{c}", random.choice(CLIENTS)))

def noise(text):
    if random.random() < 0.45:
        text = random.choice(GREET) + text
    if random.random() < 0.55:
        text = text.rstrip(".") + random.choice(CLOSE)
    if random.random() < 0.15:
        text = text.lower()
    if random.random() < 0.10:  # occasional typo: swap two letters in a long word
        words = text.split()
        idx = [i for i, w in enumerate(words) if len(w) > 5]
        if idx:
            i = random.choice(idx)
            w = words[i]
            j = random.randint(1, len(w) - 3)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            text = " ".join(words)
    return text.strip()

def lang_of(text):
    urdu_markers = ["kar dein", "bata dein", "nahi", "chahiye", "ho gayi", "ka ", "ki ",
                    "mera", "meri", "mujhe", "konsa", "bohat", "wala"]
    hits = sum(1 for m in urdu_markers if m in text.lower())
    return "roman_urdu" if hits >= 2 else "english"

def main():
    rows, seen = [], set()
    per_class = 30
    for cat, (core, held) in CATEGORIES.items():
        made = {"core": 0, "heldout": 0}
        targets = {"core": 24, "heldout": 6}
        pools = {"core": core, "heldout": held}
        held_cycle = random.sample(held, len(held))  # each held-out phrasing used exactly once
        guard = 0
        while sum(made.values()) < per_class and guard < 4000:
            guard += 1
            group = "core" if made["core"] < targets["core"] else "heldout"
            if group == "heldout":
                template = held_cycle[made["heldout"] % len(held_cycle)]
            else:
                template = random.choice(pools[group])
            text = noise(fill(template))
            if text in seen:
                continue
            seen.add(text)
            made[group] += 1
            rows.append({"query": text, "category": cat,
                         "language": lang_of(text), "template_group": group})
    random.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = i
    out = os.path.join(os.path.dirname(__file__), "..", "data", "support_queries.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "query", "category", "language", "template_group"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("total:", len(rows))
    print("by category:", dict(Counter(r["category"] for r in rows)))
    print("by language:", dict(Counter(r["language"] for r in rows)))
    print("by group:", dict(Counter(r["template_group"] for r in rows)))
    print("saved:", out)

if __name__ == "__main__":
    main()
