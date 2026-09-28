from owner_database import find_owner
from fine_calculator import calculate_fine, get_violation_name


def process_violation(plate_number, violation_type):
    """
    Process a detected traffic violation.

    Steps:
    1. Find vehicle owner.
    2. Identify violation.
    3. Calculate fine.
    4. Generate notification message.
    """

    # Find owner
    owner = find_owner(plate_number)

    if owner is None:
        print("Owner not found for plate:", plate_number)
        return None

    owner_name, phone_number = owner

    # Get violation information
    violation_name = get_violation_name(violation_type)
    fine_amount = calculate_fine(violation_type)

    # Create notification message
    message = (
        "TRAFFIC VIOLATION ALERT\n\n"
        f"Owner: {owner_name}\n"
        f"Vehicle: {plate_number}\n"
        f"Violation: {violation_name}\n"
        f"Fine: ₹{fine_amount}\n\n"
        "Please follow traffic rules."
    )

    # Display information
    print("\n========== VIOLATION ==========")
    print("Owner:", owner_name)
    print("Phone:", phone_number)
    print("Vehicle:", plate_number)
    print("Violation:", violation_name)
    print("Fine: ₹", fine_amount)
    print("\nMessage:")
    print(message)
    print("===============================\n")

    return {
        "owner_name": owner_name,
        "phone_number": phone_number,
        "plate_number": plate_number,
        "violation": violation_name,
        "fine": fine_amount,
        "message": message
    }


if __name__ == "__main__":

    # Test data
    plate = "AP39AB1234"
    violation = "red_light"

    result = process_violation(
        plate,
        violation
    )

    if result:
        print("Violation processed successfully.")