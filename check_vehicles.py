from database import db

vehicles = db["vehicles"]

print("\nVEHICLES STORED IN MONGODB")
print("-" * 40)

for vehicle in vehicles.find(
    {},
    {"_id": 0, "plate_number": 1, "owner_name": 1, "phone": 1, "vehicle_type": 1}
):
    print(vehicle)