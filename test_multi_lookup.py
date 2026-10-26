from owner_database import find_owner

test_plates = [
    "AP39AB1234",
    "TS09CD5678",
    "AP16EF9012"
]

print("\nMULTI-VEHICLE DATABASE TEST")
print("=" * 40)

for plate in test_plates:
    owner = find_owner(plate)

    if owner:
        print("\nPlate:", plate)
        print("Owner:", owner["owner_name"])
        print("Phone:", owner["phone_number"])
        print("Vehicle type:", owner["vehicle_type"])
        print("Status: MATCHED")
    else:
        print("\nPlate:", plate)
        print("Status: NOT FOUND")

print("\nDATABASE TEST COMPLETED")