
import re
from datetime import datetime
from database import db

# Use the existing MongoDB database and collections
vehicles = db["vehicles"]
violations = db["violations"]


def normalize_plate(vehicle_number):
    """Remove spaces and standardize a vehicle number."""
    return re.sub(r"\s+", "", str(vehicle_number)).upper()


def find_owner(vehicle_number):
    """Find vehicle owner details in MongoDB."""
    plate = normalize_plate(vehicle_number)

    # Try exact match first
    owner = vehicles.find_one({"plate_number": plate})

    # Fallback: handle spaces or letter-case differences
    if owner is None:
        for record in vehicles.find({}):
            if normalize_plate(record.get("plate_number", "")) == plate:
                owner = record
                break

    if owner is None:
        return None

    return {
        "vehicle_number": owner.get("plate_number", plate),
        "owner_name": owner.get("owner_name", "Unknown"),
        "phone_number": owner.get(
            "phone_number",
            owner.get("phone", "")
        ),
        "vehicle_type": owner.get("vehicle_type", "Unknown")
    }


def add_owner(vehicle_number, owner_name, phone_number,
              vehicle_type="Unknown"):
    """Add a vehicle owner to MongoDB."""
    plate = normalize_plate(vehicle_number)

    if vehicles.find_one({"plate_number": plate}):
        print("This vehicle number already exists.")
        return False

    vehicles.insert_one({
        "plate_number": plate,
        "owner_name": owner_name.strip(),
        "phone": phone_number.strip(),
        "vehicle_type": vehicle_type
    })

    print("Owner added successfully.")
    return True


def record_violation(vehicle_number, owner_name, phone_number,
                     violation_type, fine_amount):
    """Save a violation in MongoDB."""
    violations.insert_one({
        "plate_number": normalize_plate(vehicle_number),
        "owner_name": owner_name,
        "phone": phone_number,
        "violation_type": violation_type,
        "fine_amount": fine_amount,
        "status": "Unpaid",
        "date": datetime.now().isoformat(timespec="seconds")
    })

    print("Violation saved to MongoDB.")


if __name__ == "__main__":
    print("Testing MongoDB owner lookup...")

    try:
        db.client.admin.command("ping")
        print("MongoDB connected successfully.")

        owner = find_owner("AP39AB1234")

        if owner:
            print("Vehicle:", owner["vehicle_number"])
            print("Owner:", owner["owner_name"])
            print("Phone:", owner["phone_number"])
        else:
            print("Owner not found in MongoDB.")

    except Exception as error:
        print("Database error:", error)
