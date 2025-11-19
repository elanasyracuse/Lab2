"""
Script to send personalized weekly RAG research digests to all active subscribers.
This script should be scheduled to run weekly (e.g., via the orchestrator).
"""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import time
from typing import Dict, List

# Import necessary local components
try:
    from database_manager import DatabaseManager
    from digest_generator import EmailDigestBot
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Make sure database_manager.py and digest_generator.py are in the same directory.")
    import sys
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Email Configuration (PLACEHOLDERS) ---
SMTP_SERVER = 'smtp.gmail.com' # Change this for other providers (e.g., smtp.sendgrid.net)
SMTP_PORT = 587 # Standard TLS port
SENDER_EMAIL = 'YOUR_SENDER_EMAIL@gmail.com' # Replace with your email
SMTP_PASSWORD = 'YOUR_APP_PASSWORD' # Replace with your app-specific password (not primary password)

def send_digest_email(recipient_email: str, subject: str, html_content: str):
    """
    Mocks sending the email using SMTP. 
    In a live environment, this is where the actual SMTP connection happens.
    """
    if SENDER_EMAIL == 'YOUR_SENDER_EMAIL@gmail.com' or SMTP_PASSWORD == 'YOUR_APP_PASSWORD':
        logger.warning(f"Mocking email send to {recipient_email}. Please update SMTP credentials for live sending.")
        # Simulating a successful send
        time.sleep(0.5) 
        return True

    try:
        # Create message container
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email

        # Attach HTML content
        part1 = MIMEText(html_content, 'html')
        msg.attach(part1)

        # Connect to SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Start TLS encryption
            server.login(SENDER_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Successfully sent digest to {recipient_email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email to {recipient_email}: {e}")
        return False

def generate_and_send_digests():
    """
    Main function to run the weekly digest delivery process.
    """
    logger.info("--- Starting Weekly Digest Delivery Process ---")
    
    db = DatabaseManager()
    digest_bot = EmailDigestBot(db)
    
    subscribers = db.get_all_subscribers()
    total_sent = 0
    
    if not subscribers:
        logger.warning("No active subscribers found in the database. Exiting.")
        db.close()
        return
        
    logger.info(f"Found {len(subscribers)} active subscribers.")
    
    for subscriber in subscribers:
        recipient_email = subscriber['email']
        preferences = subscriber['preferences']
        
        logger.info(f"Generating personalized digest for {recipient_email} with preferences: {preferences}")
        
        # 1. Generate the personalized HTML content
        html_content = digest_bot.generate_digest_html(preferences)
        
        # 2. Define the subject line
        subject = f"Your Weekly RAG Digest - Top {len(preferences) if preferences else 'Relevant'} Research Papers"
        
        # 3. Send the email
        if send_digest_email(recipient_email, subject, html_content):
            total_sent += 1
            
        # IMPORTANT: Add a small delay to avoid hitting email provider rate limits
        time.sleep(1) 
        
    logger.info(f"--- Digest delivery complete. Sent {total_sent} of {len(subscribers)} emails. ---")
    db.close()

if __name__ == "__main__":
    generate_and_send_digests()