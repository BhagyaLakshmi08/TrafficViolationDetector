from owner_database import find_owner
from fine_calculator import calculate_fine


def process_violation(vehicle_number, violation_type):

    vehicle_number = vehicle_number.upper().strip()

    print()
    print("--------------------------------")
    print("PROCESSING VIOLATION")
    print("--------------------------------")

    print("Vehicle:", vehicle_number)

    # Find owner from MongoDB
    owner = find_owner(vehicle_number)

    if not owner:

        print(
            "Owner not found for plate:",
            vehicle_number
        )

        return None

    owner_name = owner["owner_name"]
    phone_number = owner["phone_number"]

    print("Owner:", owner_name)
    print("Phone:", phone_number)

    # Calculate fine
    fine = calculate_fine(violation_type)

    print("Violation:", violation_type)
    print("Fine: ₹", fine)

    # Create SMS message
    message = (
        "Traffic Violation Alert!\n"
        f"Vehicle: {vehicle_number}\n"
        f"Owner: {owner_name}\n"
        f"Violation: {violation_type}\n"
        f"Fine: ₹{fine}\n"
        "Please follow traffic rules."
    )

    return {
        "vehicle_number": vehicle_number,
        "owner_name": owner_name,
        "phone_number": phone_number,
        "violation_type": violation_type,
        "fine": fine,
        "message": message
    }