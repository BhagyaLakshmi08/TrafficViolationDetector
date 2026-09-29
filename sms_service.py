from twilio.rest import Client
from dotenv import load_dotenv
import os


# Load variables from .env
load_dotenv()


def send_sms(message, phone_number):

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")

    print("\nChecking Twilio configuration...")

    # Check credentials
    if not account_sid:
        print("ERROR: TWILIO_ACCOUNT_SID is missing.")
        return False

    if not auth_token:
        print("ERROR: TWILIO_AUTH_TOKEN is missing.")
        return False

    if not twilio_number:
        print("ERROR: TWILIO_PHONE_NUMBER is missing.")
        return False

    print("Account SID found.")
    print("Auth Token found.")
    print("Twilio phone number:", twilio_number)

    try:

        # Create Twilio client
        client = Client(
            account_sid,
            auth_token
        )

        # Send SMS
        sms = client.messages.create(
            body=message,
            from_=twilio_number,
            to=phone_number
        )

        print("\n==============================")
        print("SMS SENT SUCCESSFULLY!")
        print("==============================")
        print("Message SID:", sms.sid)
        print("Receiver:", phone_number)

        return True

    except Exception as e:

        print("\n==============================")
        print("SMS FAILED")
        print("==============================")
        print("Error:", e)

        return False


# -------------------------------------------------
# TEST
# -------------------------------------------------

if __name__ == "__main__":

    print("--------------------------------")
    print("TWILIO SMS TEST")
    print("--------------------------------")

    receiver_number = input(
        "Enter receiver phone number: "
    ).strip()

    message = (
        "Traffic Violation Alert!\n"
        "Vehicle: L656XH\n"
        "Violation: red_light\n"
        "Fine: Rs.1000\n"
        "Please follow traffic rules."
    )

    send_sms(
        message,
        receiver_number
    )