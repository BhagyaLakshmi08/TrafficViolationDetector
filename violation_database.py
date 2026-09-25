from pymongo import MongoClient
from datetime import datetime


# =====================================================
# MONGODB CONNECTION
# =====================================================

MONGO_URI = "mongodb+srv://meduribhagyalakshmi2008_db_user:Bhagya1234@skillbridge.vxnsuy7.mongodb.net/?appName=SkillBridge"


client = MongoClient(MONGO_URI)

db = client["TrafficViolationDetector"]

vehicles = db["vehicles"]

violations = db["violations"]


# =====================================================
# TEST CONNECTION
# =====================================================

try:

    client.admin.command("ping")

    print("MongoDB connected successfully!")

except Exception as e:

    print("MongoDB connection failed:")
    print(e)

    exit()


# =====================================================
# VIOLATION DETAILS
# =====================================================

plate_number = "AP39AB1234"

violation_type = "Red Light Violation"

fine_amount = 1000


# =====================================================
# FIND VEHICLE OWNER
# =====================================================

vehicle = vehicles.find_one({
    "plate_number": plate_number
})


if vehicle:

    owner_name = vehicle["owner_name"]
    phone = vehicle["phone"]
    vehicle_type = vehicle["vehicle_type"]


    print()
    print("Vehicle found!")
    print("Plate:", plate_number)
    print("Owner:", owner_name)
    print("Phone:", phone)
    print("Vehicle Type:", vehicle_type)


    # =================================================
    # CREATE VIOLATION RECORD
    # =================================================

    violation_record = {

        "plate_number": plate_number,

        "owner_name": owner_name,

        "phone": phone,

        "vehicle_type": vehicle_type,

        "violation_type": violation_type,

        "fine_amount": fine_amount,

        "status": "Unpaid",

        "date": datetime.now()

    }


    # =================================================
    # SAVE TO MONGODB
    # =================================================

    result = violations.insert_one(
        violation_record
    )


    print()
    print("Violation recorded successfully!")

    print(
        "Violation ID:",
        result.inserted_id
    )


    print("Violation:", violation_type)

    print("Fine: ₹", fine_amount)

    print("Status: Unpaid")


else:

    print()
    print(
        "Vehicle not found in database."
    )


# =====================================================
# CLOSE CONNECTION
# =====================================================

client.close()