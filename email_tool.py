"""Send an email with PDF attachment via SMTP."""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

load_dotenv()


def send_plan_email(receiver_email: str, pdf_path: str, subject: str = "Your Trip Plan") -> str:
    """Send the trip plan PDF to a receiver email address.
    Args:
        receiver_email: Recipient's email address
        pdf_path: Absolute path to the PDF file
        subject: Email subject line
    Returns:
        Success or error message.
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.hostinger.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")

    if not smtp_user or not smtp_pass:
        return "ERROR: SMTP credentials not found in .env"

    if not os.path.exists(pdf_path):
        return f"ERROR: PDF file not found: {pdf_path}"

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = receiver_email
    msg["Subject"] = subject

    body = "Hello!\n\nPlease find your trip plan attached as a PDF.\n\nBest regards,\nTrip Planner Agent"
    msg.attach(MIMEText(body, "plain"))

    # Attach PDF
    filename = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={filename}")
    msg.attach(part)

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return f"SUCCESS: Email sent to {receiver_email} with attachment {filename}"
    except smtplib.SMTPAuthenticationError:
        return "ERROR: SMTP authentication failed. Check SMTP_USER and SMTP_PASS in .env"
    except Exception as e:
        return f"ERROR: Failed to send email: {e}"
