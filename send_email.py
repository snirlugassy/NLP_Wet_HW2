import os
from smtplib import SMTP_SSL
from email.message import EmailMessage

# smtp.gmail.com
# Requires SSL: Yes
# Requires TLS: Yes (if available)
# Requires Authentication: Yes
# Port for SSL: 465
# Port for TLS/STARTTLS: 587

SUBJECT = "NLP HW2 Notification"
ADDR = "lugassysnir@gmail.com"

message_template = """To: {}
Subject: {}

{}
"""

def send_email(subject, body):
    with SMTP_SSL("smtp.gmail.com", port=465) as smtp:
        smtp.ehlo()
        smtp.login(user=ADDR, password=os.environ.get("SMTP_PWD"))
        smtp.sendmail(msg=message_template.format(ADDR, subject, body), to_addrs=ADDR, from_addr=ADDR)


if __name__ == "__main__":
    send_email("Test subject 1", "Test email 1")