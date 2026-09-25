from pymongo import MongoClient

# ---------------------------------------
# MONGODB CONNECTION
# ---------------------------------------
MONGO_URI = "mongodb+srv://meduribhagyalakshmi2008_db_user:Bhagya1234@skillbridge.vxnsuy7.mongodb.net/?appName=SkillBridge"

client = MongoClient(MONGO_URI)

# Select database
db = client["TrafficViolationDetector"]


# Select collection
vehicles = db["vehicles"]


# ---------------------------------------
# TEST CONNECTION
# ---------------------------------------

try:
    client.admin.command("ping")
    print("MongoDB connected successfully!")

except Exception as e:
    print("MongoDB connection failed:")
    print(e)


# ---------------------------------------
# ADD SAMPLE VEHICLES
# ---------------------------------------

sample_vehicles = [
    {
        "plate_number": "AP39AB1234",
        "owner_name": "Rahul",
        "phone": "9876543210",
        "vehicle_type": "Car"
    },
    {
        "plate_number": "TS09CD5678",
        "owner_name": "Arun",
        "phone": "9876543211",
        "vehicle_type": "Bike"
    },
    {
        "plate_number": "AP16EF9012",
        "owner_name": "Priya",
        "phone": "9876543212",
        "vehicle_type": "Car"
    }
]


# Insert only if they don't already exist

for vehicle in sample_vehicles:

    existing = vehicles.find_one({
        "plate_number": vehicle["plate_number"]
    })

    if existing is None:

        vehicles.insert_one(vehicle)

        print(
            "Added vehicle:",
            vehicle["plate_number"]
        )


# ---------------------------------------
# TEST SEARCH
# ---------------------------------------

test_plate = "AP39AB1234"

vehicle = vehicles.find_one({
    "plate_number": test_plate
})


if vehicle:

    print()
    print("Vehicle found!")
    print("Plate:", vehicle["plate_number"])
    print("Owner:", vehicle["owner_name"])
    print("Phone:", vehicle["phone"])
    print("Vehicle Type:", vehicle["vehicle_type"])

else:

    print("Vehicle not found.")