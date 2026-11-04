import io
import time
import uuid
import threading
import traceback

import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError

from smartrupee.engine import (
    analyze,
    load_ai_model,
    STAGES
)


# ==================================================
# FLASK CONFIGURATION
# ==================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

JOBS = {}
JOBS_LOCK = threading.Lock()


# ==================================================
# LOAD AI MODEL
# ==================================================

try:
    MODEL_LOADED, MODEL_ERROR = load_ai_model()

except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR = str(e)

if MODEL_LOADED:
    print("SmartRupee AI model loaded successfully.")
else:
    print("SmartRupee AI model warning:", MODEL_ERROR)


# ==================================================
# RESULT FORMATTING
# ==================================================

def prepare_currency_prediction(result):

    verdict = result.get("verdict") or {}
    overview = result.get("overview") or {}
    model = overview.get("model") or {}

    denomination = overview.get("denomination", "Unknown")

    if not denomination:
        denomination = "Unknown"

    # Read engine verdict.
    label = str(
        verdict.get("label", "")
    ).strip().upper()

    # Genuine result.
    if label in ("LIKELY GENUINE", "GENUINE"):

        authenticity = "GENUINE"
        sentence = "THIS IS REAL CURRENCY"
        display_verdict = "GENUINE"

    # Possible counterfeit result.
    elif label in (
        "POSSIBLE COUNTERFEIT",
        "LIKELY FAKE",
        "FAKE",
        "COUNTERFEIT"
    ):

        authenticity = "POSSIBLE FAKE"
        sentence = "POSSIBLE FAKE CURRENCY"
        display_verdict = "POSSIBLE COUNTERFEIT"

    # Uncertain result.
    else:

        authenticity = "INCONCLUSIVE"
        sentence = "CURRENCY COULD NOT BE VERIFIED"
        display_verdict = "INCONCLUSIVE"

    return {
        "denomination": denomination,
        "authenticity": authenticity,
        "sentence": sentence,
        "display_verdict": display_verdict,
        "confidence": model.get("confidence"),
        "disclaimer": (
            "SmartRupee is an AI-based visual screening system. "
            "It compares uploaded banknotes with available genuine "
            "reference images. It cannot guarantee authenticity "
            "or officially confirm counterfeit currency. "
            "Physical verification is recommended."
        )
    }


# ==================================================
# BACKGROUND ANALYSIS
# ==================================================

def run(jid, img):

    with JOBS_LOCK:
        if jid not in JOBS:
            return

    def progress(i, detail, extra=None):

        with JOBS_LOCK:

            current = JOBS.get(jid)

            if current is None:
                return

            current["stage"] = i
            current["detail"] = detail

            if isinstance(extra, dict):
                current.update(extra)

    try:

        result = analyze(img, progress)

        if not isinstance(result, dict):
            raise ValueError(
                "Analysis engine returned an invalid result."
            )

        # Add formatted currency prediction.
        result["currency_prediction"] = (
            prepare_currency_prediction(result)
        )

        with JOBS_LOCK:

            current = JOBS.get(jid)

            if current is not None:

                current["result"] = result
                current["stage"] = max(0, len(STAGES) - 1)
                current["detail"] = "Analysis completed"
                current["done"] = True

    except Exception as e:

        traceback.print_exc()

        with JOBS_LOCK:

            current = JOBS.get(jid)

            if current is not None:

                current["error"] = (
                    f"Analysis failed: {str(e)}"
                )

                current["done"] = True


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        stages=STAGES
    )


# ==================================================
# START ANALYSIS
# ==================================================

@app.post("/api/analyze")
def start():

    uploaded_file = request.files.get("image")

    if not uploaded_file:

        return jsonify({
            "error": "No image received."
        }), 400

    try:

        image_bytes = uploaded_file.read()

        if not image_bytes:

            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        im = Image.open(
            io.BytesIO(image_bytes)
        )

        if im.format not in ("JPEG", "PNG"):

            return jsonify({
                "error": "Only JPG, JPEG and PNG images are supported."
            }), 400

        im = ImageOps.exif_transpose(im).convert("RGB")

        image_array = np.array(im)

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

    except (UnidentifiedImageError, OSError):

        return jsonify({
            "error": "The file is not a readable image."
        }), 400

    except Exception as e:

        return jsonify({
            "error": f"Image processing failed: {str(e)}"
        }), 400

    # ==================================================
    # CREATE JOB
    # ==================================================

    with JOBS_LOCK:

        # Remove expired jobs.
        expired = [
            key
            for key, job in JOBS.items()
            if time.time() - job["t"] > 1800
        ]

        for key in expired:
            JOBS.pop(key, None)

        jid = uuid.uuid4().hex

        JOBS[jid] = {
            "t": time.time(),
            "stage": 0,
            "detail": "Queued",
            "done": False,
            "error": None,
            "result": None,
            "preview": None,
            "layout": None,
            "found": None
        }

    # ==================================================
    # START BACKGROUND THREAD
    # ==================================================

    thread = threading.Thread(
        target=run,
        args=(jid, image_bgr),
        daemon=True
    )

    thread.start()

    return jsonify({
        "job_id": jid
    })


# ==================================================
# CHECK ANALYSIS STATUS
# ==================================================

@app.get("/api/status/<jid>")
def status(jid):

    with JOBS_LOCK:

        job = JOBS.get(jid)

        if not job:

            return jsonify({
                "error": "Unknown or expired scan."
            }), 404

        response = {
            "stage": job["stage"],
            "detail": job["detail"],
            "done": job["done"],
            "error": job["error"],
            "preview": job["preview"],
            "layout": job["layout"],
            "found": job["found"]
        }

        if job["done"] and job["result"] is not None:

            response["result"] = job["result"]

    return jsonify(response)


# ==================================================
# FILE SIZE ERROR
# ==================================================

@app.errorhandler(413)
def too_big(_):

    return jsonify({
        "error": "Image is larger than 12 MB."
    }), 413


# ==================================================
# GENERAL ERROR HANDLER
# ==================================================

@app.errorhandler(500)
def internal_error(error):

    traceback.print_exc()

    return jsonify({
        "error": "An internal server error occurred."
    }), 500


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )