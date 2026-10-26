
from flask import Flask, render_template, request, url_for
from pathlib import Path
from owner_database import find_owner

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_VIDEO = BASE_DIR / "output" / "plate_recognition.mp4"


@app.route("/")
def dashboard():
    video_exists = OUTPUT_VIDEO.exists()

    return render_template(
        "index.html",
        video_exists=video_exists,
        lookup_done=False,
        owner=None
    )


@app.route("/lookup", methods=["GET"])
def lookup():
    plate = request.args.get("plate", "").strip()
    owner = None
    lookup_done = False

    if plate:
        lookup_done = True

        try:
            owner = find_owner(plate)
        except Exception as error:
            app.logger.exception("Vehicle lookup failed")
            return (
                "Vehicle lookup failed. Check your MongoDB connection "
                "and owner_database.py configuration.",
                500
            )

    return render_template(
        "index.html",
        video_exists=OUTPUT_VIDEO.exists(),
        lookup_done=lookup_done,
        owner=owner
    )


@app.route("/video")
def processed_video():
    if not OUTPUT_VIDEO.exists():
        return "Processed video is not available yet.", 404

    from flask import send_file
    return send_file(OUTPUT_VIDEO, mimetype="video/mp4")


if __name__ == "__main__":
    print("Starting TrafficGuard dashboard...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=5000)
