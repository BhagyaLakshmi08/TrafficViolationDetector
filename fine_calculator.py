def calculate_fine(violation_type):
    """
    Calculate fine based on the type of traffic violation.
    """

    fines = {
        "red_light": 1000,
        "stop_line": 500,
        "no_helmet": 1000,
        "no_seatbelt": 1000,
        "overspeeding": 2000,
        "wrong_lane": 500,
        "dangerous_driving": 5000
    }

    return fines.get(violation_type.lower(), 0)


def get_violation_name(violation_type):
    """
    Convert internal violation name into readable text.
    """

    names = {
        "red_light": "Red Light Violation",
        "stop_line": "Stop Line Violation",
        "no_helmet": "No Helmet",
        "no_seatbelt": "No Seatbelt",
        "overspeeding": "Overspeeding",
        "wrong_lane": "Wrong Lane",
        "dangerous_driving": "Dangerous Driving"
    }

    return names.get(
        violation_type.lower(),
        "Unknown Violation"
    )


if __name__ == "__main__":

    violation = "red_light"

    fine = calculate_fine(violation)
    name = get_violation_name(violation)

    print("Violation:", name)
    print("Fine: ₹", fine)