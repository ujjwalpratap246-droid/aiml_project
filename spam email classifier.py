import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import imaplib
import email
from email.header import decode_header
import os

# --- Load dataset ---
df = pd.read_csv("emails.csv")

# Vectorize the messages
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df['message'])

# Labels: spam=1, ham=0 (detected from filename)
y = df['file'].str.contains("spam", case=False).astype(int)

# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Backend ML classifier ---
def classify_message(msg: str) -> str:
    """Classify a given message as 'Ham' (not spam) or 'Spam'."""
    if not msg.strip():
        return "Message is empty!"
    user_vector = vectorizer.transform([msg])
    prediction = model.predict(user_vector)[0]
    return "Ham ✅ (Not Spam)" if prediction == 0 else "Spam 🚨"

# --- Email login settings ---
# ⚠️ Replace with your email + app password, or safer: put them into env variables!
EMAIL_USER = "akutosaisolos69@gmail.com"
EMAIL_PASS = "armo zpyn gjtk oyac"  # this should be the Gmail App Password (16 chars)
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993
if not EMAIL_USER or not EMAIL_PASS:
    raise ValueError("❌ Missing EMAIL_USER or EMAIL_PASS")

# --- Helper: Extract plain text body safely ---
def extract_body(msg):
    """Safely extract plain text body from email message"""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_dispo = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_dispo:
                try:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    body = part.get_payload()
                break
    else:
        try:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        except Exception:
            body = msg.get_payload()
    return body

# --- Fetch emails and classify ---
def fetch_and_classify_emails():
    print("🔌 Connecting to Gmail...")
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)

    try:
        mail.login(EMAIL_USER, EMAIL_PASS)
        print(f"✅ Logged in as {EMAIL_USER}")
    except Exception as e:
        print("❌ Login failed:", e)
        return

    mail.select("inbox")
    status, messages = mail.search(None, '(UNSEEN)')
    if status != "OK":
        print("❌ Could not search inbox.")
        return

    email_ids = messages[0].split()
    print(f"📧 Found {len(email_ids)} unread emails.")

    if not email_ids:
        print("📭 No unread emails to classify.")
    else:
        for eid in email_ids:
            status, msg_data = mail.fetch(eid, "(RFC822)")
            if status != "OK":
                print("⚠️ Failed to fetch email", eid)
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")

                    body = extract_body(msg)
                    prediction = classify_message(body)

                    print(f"\n✉️ Subject: {subject}")
                    print(f"   Prediction: {prediction}")

                    # ⭐ If spam, move to Spam folder
                    if "Spam" in prediction:
                        result = mail.copy(eid, "[Gmail]/Spam")
                        if result[0] == "OK":
                            mail.store(eid, "+FLAGS", "\\Deleted")  # mark original as deleted
                            mail.expunge()
                            print("   🚨 Email moved to Gmail Spam folder.")
                        else:
                            print("   ⚠️ Failed to move email.")

                    print("-" * 50)

    mail.logout()
    print("🔒 Logged out of Gmail.")
    
# --- MAIN ---
if __name__ == "__main__":
    # Test with a fake message
    msg = "Congratulations! You’ve won a free ticket!"
    print("Manual test:", classify_message(msg))

    # Now fetch and classify live emails
    fetch_and_classify_emails()
