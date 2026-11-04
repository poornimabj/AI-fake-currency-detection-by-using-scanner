"""
SmartRupee Analysis Engine
Banknote preprocessing, denomination matching, ORB, SSIM,
OCR and genuine-reference anomaly screening.

Important: This is visual screening, not certified
counterfeit detection.
"""

import os
import re
import glob
import json
import base64
import logging
import threading

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF_DIR = os.path.join(BASE, "data", "references")

MODEL_PATH = os.path.join(
    BASE, "models", "genuine_feature_extractor.keras"
)

ANOMALY_PATH = os.path.join(
    BASE, "models", "genuine_reference_anomaly.keras"
)

DISTRIBUTION_PATH = os.path.join(
    BASE, "models", "genuine_reference_distribution.npz"
)

AUTHENTICITY_200_PATH = os.path.join(
    BASE, "models", "currency_authenticity_200.joblib"
)

EVAL_PATH = os.path.join(BASE, "reports", "evaluation.json")

W, H = 1200, 520

CFG = {
    "ratio": 0.75,
    "inl_ok": 20,
    "inl_bad": 8,
    "ssim_ok": 0.50,
    "ssim_bad": 0.30,
    "min_texture": 12.0,
    "min_edge": 0.01,
    "max_glare": 0.25
}

STAGES = [
    "Image uploaded successfully",
    "Image preprocessing",
    "Banknote feature extraction",
    "Corner-wise analysis",
    "ORB feature matching",
    "SSIM similarity analysis",
    "Genuine-reference comparison",
    "Final reference report"
]

REGIONS = [
    ("top_left", "Top-left corner",
     (0.00, 0.00, 0.24, 0.40), True),

    ("top_right", "Top-right corner",
     (0.76, 0.00, 1.00, 0.40), True),

    ("bottom_left", "Bottom-left corner",
     (0.00, 0.60, 0.24, 1.00), True),

    ("bottom_right", "Bottom-right corner",
     (0.76, 0.60, 1.00, 1.00), True),

    ("thread", "Security-thread band (visual only)",
     (0.46, 0.00, 0.56, 1.00), False),

    ("centre_left", "Central-left design region",
     (0.26, 0.15, 0.44, 0.85), False),

    ("centre_right", "Central-right design region",
     (0.58, 0.15, 0.74, 0.85), False)
]

FEATURE_REGIONS = [
    ("watermark", "Watermark window",
     (0.20, 0.12, 0.34, 0.38), False),

    ("security_thread", "Security thread",
     (0.45, 0.08, 0.50, 0.92), False),

    ("serial_number", "Serial number area",
     (0.76, 0.60, 0.98, 0.78), False),

    ("portrait", "Portrait area",
     (0.35, 0.18, 0.48, 0.70), False)
]

REGION_FEATURES = {
    "top_left": "Security printing and design details",
    "top_right": "Printed design details",
    "bottom_left": "Micro-lettering and surrounding design details",
    "bottom_right": "Serial number and surrounding printed details",
    "thread": "Security thread",
    "centre_left": "Watermark-related area, where applicable",
    "centre_right": "Portrait and surrounding design",
    "watermark": "Watermark visibility requires transmitted light",
    "security_thread": "Embedded security thread (visual inspection only)",
    "serial_number": "Serial number and surrounding printed details",
    "portrait": "Portrait and surrounding design"
}

COL = {
    "green": (80, 200, 80),
    "red": (60, 60, 235),
    "yellow": (0, 200, 240)
}

PHYSICAL = [
    "Tactile (raised-print) features",
    "UV fluorescence",
    "Infrared response",
    "Watermark and see-through registration under transmitted light",
    "Colour-shift ink angle response",
    "Paper and thread feel"
]

VISUAL_INDICATORS = [
    {
        "name": "Watermark inconsistency",
        "assessment": "Inspect against a verified reference under transmitted light."
    },
    {
        "name": "Security-thread inconsistency",
        "assessment": "Check position, continuity and appearance against a same-series reference."
    },
    {
        "name": "See-through register mismatch",
        "assessment": "Requires front and back alignment under transmitted light."
    },
    {
        "name": "Poor micro-lettering",
        "assessment": "Image resolution and magnification can affect this result."
    },
    {
        "name": "Printing irregularities",
        "assessment": "Visible smudging or pattern inconsistency requires reference comparison."
    },
    {
        "name": "Serial-number irregularity",
        "assessment": "Compare format, spacing and print quality against verified references."
    }
]


# ==================================================
# OCR
# ==================================================

OCR_READER = None


def regional_text_assessment(crop_image):

    global OCR_READER

    try:

        if OCR_READER is None:

            import easyocr

            OCR_READER = easyocr.Reader(
                ["en"],
                gpu=False,
                verbose=False
            )

        gray = cv2.cvtColor(
            crop_image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.createCLAHE(
            2.0,
            (8, 8)
        ).apply(gray)

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        results = OCR_READER.readtext(
            gray,
            detail=1,
            paragraph=False
        )

        usable = [
            float(item[2])
            for item in results
            if float(item[2]) >= 0.2
        ]

        if not usable:

            return {
                "status": "No readable text detected",
                "text_confidence": None
            }

        return {
            "status": "Printed text detected",
            "text_confidence": round(max(usable), 3)
        }

    except Exception:

        return {
            "status": "OCR unavailable",
            "text_confidence": None
        }


# ==================================================
# IMAGE UTILITIES
# ==================================================

def b64(img, q=88):

    if img is None or img.size == 0:
        return None

    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, q]
    )

    if not ok:
        return None

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(buf).decode()
    )


def crop(img, box):

    h, w = img.shape[:2]

    x0, y0, x1, y1 = box

    x0 = max(0, min(w, int(x0 * w)))
    x1 = max(0, min(w, int(x1 * w)))
    y0 = max(0, min(h, int(y0 * h)))
    y1 = max(0, min(h, int(y1 * h)))

    return img[y0:y1, x0:x1]


def order_pts(points):

    points = np.asarray(
        points,
        dtype="float32"
    )

    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).ravel()

    return np.array(
        [
            points[np.argmin(total)],
            points[np.argmin(difference)],
            points[np.argmax(total)],
            points[np.argmax(difference)]
        ],
        dtype="float32"
    )


# ==================================================
# IMAGE QUALITY
# ==================================================

def quality(img):

    scale = min(
        1.0,
        800 / max(img.shape[:2])
    )

    resized = (
        cv2.resize(img, None, fx=scale, fy=scale)
        if scale < 1
        else img
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    blur = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(gray.mean())
    glare = float((gray >= 250).mean())
    dark = float((gray <= 35).mean())

    issues = []
    score = 100

    if blur < 40:
        issues.append("Image is blurry")
        score -= 35

    if brightness < 60:
        issues.append("Image is too dark")
        score -= 25

    if brightness > 200 or glare > 0.12:
        issues.append("Glare or overexposure")
        score -= 25

    if dark > 0.25:
        issues.append("Large dark areas reduce visible detail")
        score -= 20

    if min(img.shape[:2]) < 400:
        issues.append("Low resolution")
        score -= 25

    score = max(0, score)

    return {
        "score": score,
        "level": (
            "Good" if score >= 75
            else "Fair" if score >= 50
            else "Poor"
        ),
        "blur": round(blur, 1),
        "blur_status": "Blurry" if blur < 40 else "Acceptable",
        "brightness": round(brightness, 1),
        "glare": round(glare, 3),
        "dark_area": round(dark, 3),
        "lighting": (
            "Poor"
            if brightness < 60 or brightness > 200 or glare > 0.12
            else "Good"
        ),
        "issues": issues
    }


# ==================================================
# BANKNOTE BOUNDARY DETECTION
# ==================================================

def detect_note(img):

    """
    Try multiple contour methods to find a rectangular
    banknote. If no reliable boundary is found, return
    the original image and mark found=False.
    """

    h, w = img.shape[:2]

    scale = min(
        1.0,
        1200 / max(h, w)
    )

    small = (
        cv2.resize(img, None, fx=scale, fy=scale)
        if scale < 1
        else img.copy()
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    total = gray.size
    candidates = []

    # Method 1: Canny edges.
    edges = cv2.Canny(
        gray,
        30,
        100
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=2
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8)
    )

    # Method 2: Otsu threshold.
    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Method 3: Adaptive threshold.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    for mask in (edges, otsu, cv2.bitwise_not(otsu), adaptive):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )[:20]:

            area = cv2.contourArea(contour) / total

            if not 0.12 < area < 0.98:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True
            )

            if len(approx) == 4 and cv2.isContourConvex(approx):

                points = order_pts(
                    approx.reshape(4, 2)
                )

            else:

                rect = cv2.minAreaRect(contour)

                points = order_pts(
                    cv2.boxPoints(rect)
                )

            width = np.linalg.norm(
                points[1] - points[0]
            )

            height = np.linalg.norm(
                points[3] - points[0]
            )

            if min(width, height) < 20:
                continue

            ratio = max(width, height) / max(
                min(width, height),
                1
            )

            if not 1.35 <= ratio <= 3.4:
                continue

            # Prefer large, banknote-shaped candidates.
            score = area - abs(ratio - 2.0) * 0.015

            candidates.append(
                (score, points)
            )

    if not candidates:

        # Do not pretend that an unaligned image was detected.
        return (
            cv2.resize(img, (W, H)),
            False,
            None
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    points = candidates[0][1] / scale

    # Reorder to landscape if needed.
    top_width = np.linalg.norm(
        points[1] - points[0]
    )

    left_height = np.linalg.norm(
        points[3] - points[0]
    )

    if left_height > top_width:

        points = points[[3, 0, 1, 2]]

    destination = np.array(
        [
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1]
        ],
        dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(
        points.astype("float32"),
        destination
    )

    aligned = cv2.warpPerspective(
        img,
        matrix,
        (W, H)
    )

    return aligned, True, points


def prep_gray(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.createCLAHE(
        2.0,
        (8, 8)
    ).apply(gray)

    return cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )


def visibility(region):

    if region is None or region.size == 0:
        return False, "Region could not be extracted."

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY
    )

    texture = float(gray.std())

    edge = float(
        (
            cv2.Canny(gray, 50, 150) > 0
        ).mean()
    )

    glare = float(
        (gray >= 250).mean()
    )

    if glare > CFG["max_glare"]:
        return False, "Strong glare covers much of this region."

    if (
        texture < CFG["min_texture"]
        or edge < CFG["min_edge"]
    ):
        return False, "Too little visible detail in this region."

    return True, ""


# ==================================================
# DENOMINATION AND SIDE FROM FILE NAMES
# ==================================================

def get_reference_metadata(filepath):

    """
    Recognises examples such as:
    10_front.jpg
    rs_20_back.png
    50.jpg
    RBI_100_front.jpeg
    """

    filename = os.path.basename(filepath).lower()

    # Longest denominations first.
    match = re.search(
        r"(?<!\d)(2000|500|200|100|50|20|10)(?!\d)",
        filename
    )

    if not match:
        return None, "unknown"

    denomination = match.group(1)

    if re.search(r"front|obverse", filename):
        side = "front"

    elif re.search(r"back|reverse", filename):
        side = "back"

    else:
        side = "unknown"

    return denomination, side


# ==================================================
# LOAD GENUINE REFERENCES
# ==================================================

_refs = None


def load_refs(force_reload=False):

    global _refs

    if force_reload:
        _refs = None

    if _refs is not None:
        return _refs

    _refs = []

    if not os.path.isdir(REF_DIR):
        print("Reference folder not found:", REF_DIR)
        return _refs

    patterns = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    files = []

    for pattern in patterns:

        files.extend(
            glob.glob(
                os.path.join(
                    REF_DIR,
                    "**",
                    pattern
                ),
                recursive=True
            )
        )

    for filepath in sorted(set(files)):

        denomination, side = get_reference_metadata(filepath)

        if denomination is None:
            print("Skipping reference without denomination:", filepath)
            continue

        image = cv2.imread(filepath)

        if image is None:
            continue

        # Align reference when possible.
        aligned, found, _ = detect_note(image)

        if not found:
            aligned = cv2.resize(
                image,
                (W, H)
            )

        _refs.append(
            {
                "name": os.path.basename(filepath),
                "img": aligned,
                "aligned": found,
                "side": side,
                "denomination": denomination,
                "path": filepath
            }
        )

    print("SmartRupee references loaded:", len(_refs))

    for ref in _refs:
        print(
            "Reference:",
            ref["name"],
            "Denomination:",
            ref["denomination"],
            "Side:",
            ref["side"]
        )

    return _refs


# ==================================================
# ORB MATCHING
# ==================================================

_orb = cv2.ORB_create(
    nfeatures=2500,
    fastThreshold=8
)


def orb_match(test_gray, reference_gray):

    kp_test, desc_test = _orb.detectAndCompute(
        test_gray,
        None
    )

    kp_ref, desc_ref = _orb.detectAndCompute(
        reference_gray,
        None
    )

    result = {
        "kp_test": len(kp_test),
        "kp_ref": len(kp_ref),
        "good": 0,
        "inliers": 0,
        "H": None
    }

    if (
        desc_test is None
        or desc_ref is None
        or len(kp_test) < 4
        or len(kp_ref) < 4
    ):
        return result

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pairs = matcher.knnMatch(
        desc_test,
        desc_ref,
        k=2
    )

    good = []

    for pair in pairs:

        if (
            len(pair) == 2
            and pair[0].distance < CFG["ratio"] * pair[1].distance
        ):
            good.append(pair[0])

    result["good"] = len(good)

    if len(good) >= 4:

        source = np.float32(
            [
                kp_test[m.queryIdx].pt
                for m in good
            ]
        ).reshape(-1, 1, 2)

        destination = np.float32(
            [
                kp_ref[m.trainIdx].pt
                for m in good
            ]
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            4.0
        )

        if homography is not None and mask is not None:

            result["inliers"] = int(mask.sum())
            result["H"] = homography

    return result


def orient_orb_match(test_gray, reference_gray):

    best = None

    for rotation in (0, 1, 2, 3):

        rotated = np.rot90(
            reference_gray,
            rotation
        ).copy()

        for candidate_image in (
            rotated,
            cv2.flip(rotated, 1)
        ):

            match = orb_match(
                test_gray,
                candidate_image
            )

            score = (
                match["inliers"]
                + match["good"] * 0.1
            )

            if best is None or score > best[0]:

                best = (
                    score,
                    match,
                    candidate_image
                )

    return best[1], best[2]


# ==================================================
# SSIM
# ==================================================

def ssim_stage(test_gray, reference_gray, orb):

    test_resized = cv2.resize(
        test_gray,
        (
            reference_gray.shape[1],
            reference_gray.shape[0]
        )
    )

    if orb["H"] is not None and orb["inliers"] >= 8:

        test_resized = cv2.warpPerspective(
            test_gray,
            orb["H"],
            (
                reference_gray.shape[1],
                reference_gray.shape[0]
            )
        )

    similarity, similarity_map = ssim_fn(
        test_resized,
        reference_gray,
        full=True,
        data_range=255
    )

    return (
        float(similarity),
        np.clip(1 - similarity_map, 0, 1)
    )


# ==================================================
# AI MODEL LOADING
# ==================================================

_model = None
_anomaly = None
_distribution = None
_model_err = None
_experimental_feature_refs = None
_experimental_feature_lock = threading.Lock()
_denomination_model_cache = {}
_denomination_model_lock = threading.Lock()
_authenticity_200_classifier = None
_authenticity_200_lock = threading.Lock()
logger = logging.getLogger(__name__)


def load_ai_model(model_path=MODEL_PATH, names_path=None):

    global _model
    global _anomaly
    global _distribution
    global _model_err
    global _experimental_feature_refs

    _model = None
    _anomaly = None
    _distribution = None
    _model_err = None

    required_files = [
        model_path,
        ANOMALY_PATH,
        DISTRIBUTION_PATH
    ]

    missing = [
        path for path in required_files
        if not os.path.isfile(path)
    ]

    if missing:

        _model_err = (
            "Required AI model files are missing: "
            + ", ".join(missing)
        )

        return False, _model_err

    try:

        from tensorflow import keras

        loaded_model = keras.models.load_model(
            model_path
        )

        input_shape = tuple(
            loaded_model.input_shape[1:3]
        )

        if input_shape != (224, 224):

            raise ValueError(
                "Expected model input size 224 x 224, "
                f"found {input_shape}."
            )

        loaded_anomaly = keras.models.load_model(
            ANOMALY_PATH
        )

        distribution = np.load(
            DISTRIBUTION_PATH
        )

        required_keys = [
            "mean",
            "scale",
            "reference_embeddings",
            "reconstruction_threshold",
            "distance_threshold"
        ]

        missing_keys = [
            key for key in required_keys
            if key not in distribution.files
        ]

        if missing_keys:

            raise ValueError(
                "Distribution file missing: "
                + ", ".join(missing_keys)
            )

        _model = loaded_model
        _anomaly = loaded_anomaly
        _distribution = distribution
        _experimental_feature_refs = None

        return True, None

    except Exception as exc:

        _model_err = str(exc)

        return False, _model_err


def _denomination_reference_model(denomination):

    denomination = str(denomination or "").removeprefix("₹")

    if not denomination.isdigit():
        return _anomaly, _distribution, None

    anomaly_path = os.path.splitext(ANOMALY_PATH)[0] + (
        f"_{denomination}.keras"
    )
    distribution_path = os.path.splitext(DISTRIBUTION_PATH)[0] + (
        f"_{denomination}.npz"
    )

    if not os.path.exists(anomaly_path) and not os.path.exists(
        distribution_path
    ):
        return _anomaly, _distribution, None

    if not os.path.isfile(anomaly_path) or not os.path.isfile(
        distribution_path
    ):
        raise FileNotFoundError(
            "Incomplete denomination-specific model files for "
            f"₹{denomination}."
        )

    with _denomination_model_lock:

        if denomination not in _denomination_model_cache:

            from tensorflow import keras

            anomaly = keras.models.load_model(anomaly_path)
            distribution = np.load(distribution_path)

            required_keys = [
                "mean",
                "scale",
                "reference_embeddings",
                "reconstruction_threshold",
                "distance_threshold"
            ]
            missing_keys = [
                key for key in required_keys
                if key not in distribution.files
            ]

            if missing_keys:
                distribution.close()
                raise ValueError(
                    "Denomination distribution file missing: "
                    + ", ".join(missing_keys)
                )

            _denomination_model_cache[denomination] = (
                anomaly,
                distribution
            )

        return (
            *_denomination_model_cache[denomination],
            denomination
        )


def _load_200_authenticity_classifier():

    global _authenticity_200_classifier

    if not os.path.isfile(AUTHENTICITY_200_PATH):
        return None

    with _authenticity_200_lock:

        if _authenticity_200_classifier is None:

            import joblib

            bundle = joblib.load(AUTHENTICITY_200_PATH)

            if (
                not isinstance(bundle, dict)
                or "classifier" not in bundle
                or "metadata" not in bundle
                or not hasattr(
                    bundle["classifier"],
                    "predict_proba"
                )
            ):
                raise ValueError(
                    "The saved ₹200 authenticity classifier is invalid."
                )

            _authenticity_200_classifier = bundle

        return _authenticity_200_classifier


# ==================================================
# AI PREDICTION
# ==================================================

def _experimental_50_feature_references():

    global _experimental_feature_refs

    with _experimental_feature_lock:

        if _experimental_feature_refs is not None:
            return _experimental_feature_refs

        genuine_images = [
            ref["img"]
            for ref in load_refs()
            if ref["denomination"] == "50"
        ]

        fake_images = []
        fake_dir = os.path.join(
            BASE,
            "dataset",
            "train",
            "fake"
        )

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }

        if os.path.isdir(fake_dir):

            for filepath in sorted(
                glob.glob(
                    os.path.join(fake_dir, "**", "*"),
                    recursive=True
                )
            ):

                if not os.path.isfile(filepath):
                    continue

                if os.path.splitext(filepath)[1].lower() not in extensions:
                    continue

                denomination, _ = get_reference_metadata(filepath)

                if denomination != "50":
                    continue

                image = cv2.imread(filepath)

                if image is None:
                    print(
                        "Skipping unreadable experimental fake reference:",
                        filepath
                    )
                    continue

                aligned, found, _ = detect_note(image)

                fake_images.append(
                    aligned
                    if found
                    else cv2.resize(image, (W, H))
                )

        def normalized_features(images):

            if not images:
                return np.empty((0, 0), dtype="float32")

            batch = np.stack([
                cv2.cvtColor(
                    cv2.resize(image, (224, 224)),
                    cv2.COLOR_BGR2RGB
                )
                for image in images
            ]).astype("float32")

            features = np.asarray(
                _model.predict(
                    batch,
                    batch_size=8,
                    verbose=0
                ),
                dtype="float32"
            )

            norms = np.linalg.norm(
                features,
                axis=1,
                keepdims=True
            )

            return features / np.maximum(norms, 1e-8)

        genuine_features = normalized_features(genuine_images)
        fake_features = normalized_features(fake_images)

        if not len(genuine_features) or not len(fake_features):
            _experimental_feature_refs = {
                "genuine": genuine_features,
                "fake": fake_features,
                "leave_one_out": {
                    "correct": 0,
                    "total": 0,
                    "genuine_as_fake": 0,
                    "fake_as_genuine": 0
                }
            }
            return _experimental_feature_refs

        all_features = np.concatenate(
            (genuine_features, fake_features),
            axis=0
        )
        labels = np.concatenate(
            (
                np.ones(len(genuine_features), dtype="int8"),
                np.zeros(len(fake_features), dtype="int8")
            )
        )

        correct = 0
        genuine_as_fake = 0
        fake_as_genuine = 0

        for index, label in enumerate(labels):

            similarities = all_features @ all_features[index]
            similarities[index] = -np.inf
            nearest_label = labels[int(np.argmax(similarities))]

            if nearest_label == label:
                correct += 1
            elif label == 1:
                genuine_as_fake += 1
            else:
                fake_as_genuine += 1

        _experimental_feature_refs = {
            "genuine": genuine_features,
            "fake": fake_features,
            "leave_one_out": {
                "correct": correct,
                "total": len(labels),
                "genuine_as_fake": genuine_as_fake,
                "fake_as_genuine": fake_as_genuine
            }
        }

        return _experimental_feature_refs


def _experimental_50_feature_match(embedding):

    references = _experimental_50_feature_references()

    genuine_features = references["genuine"]
    fake_features = references["fake"]

    if not len(genuine_features) or not len(fake_features):
        return {
            "status": "unavailable",
            "message": (
                "Both genuine and fake ₹50 feature references "
                "are required for this comparison."
            )
        }

    query = embedding / max(
        float(np.linalg.norm(embedding)),
        1e-8
    )

    genuine_similarity = float(
        np.max(genuine_features @ query)
    )
    fake_similarity = float(
        np.max(fake_features @ query)
    )

    validation = references["leave_one_out"]

    return {
        "status": "experimental",
        "nearest_reference_class": (
            "genuine"
            if genuine_similarity >= fake_similarity
            else "fake"
        ),
        "genuine_similarity": round(genuine_similarity, 4),
        "fake_similarity": round(fake_similarity, 4),
        "genuine_reference_count": int(len(genuine_features)),
        "fake_reference_count": int(len(fake_features)),
        "leave_one_out": validation
    }


def model_predict(note, denomination=None):

    global _model
    global _anomaly
    global _distribution

    if (
        _model is None
        or _anomaly is None
        or _distribution is None
    ):

        success, error = load_ai_model()

        if not success:
            return None, error

    try:

        anomaly, distribution, reference_denomination = (
            _denomination_reference_model(denomination)
        )

        rgb = cv2.cvtColor(
            cv2.resize(note, (224, 224)),
            cv2.COLOR_BGR2RGB
        )

        image = rgb.astype(
            "float32"
        )[None, ...]

        embedding = np.asarray(
            _model.predict(
                image,
                verbose=0
            )[0],
            dtype="float32"
        )

        mean = distribution["mean"]
        scale = np.maximum(
            distribution["scale"],
            1e-6
        )

        standardized = (
            embedding - mean
        ) / scale

        reconstruction = np.asarray(
            anomaly.predict(
                standardized[None, ...],
                verbose=0
            )[0],
            dtype="float32"
        )

        reconstruction_error = float(
            np.mean(
                np.square(
                    reconstruction - standardized
                )
            )
        )

        distance = float(
            np.sqrt(
                np.sum(
                    np.square(standardized)
                )
            )
        )

        reconstruction_limit = float(
            np.asarray(
                distribution["reconstruction_threshold"]
            ).ravel()[0]
        )

        distance_limit = float(
            np.asarray(
                distribution["distance_threshold"]
            ).ravel()[0]
        )

        normalized = (
            reconstruction_error
            / max(reconstruction_limit, 1e-6)
        )

        distance_normalized = (
            distance
            / max(distance_limit, 1e-6)
        )

        if (
            normalized <= 1
            and distance_normalized <= 1
        ):

            label = "Consistent with genuine references"

        elif (
            normalized >= 20
            and distance_normalized >= 2.5
        ):

            label = "Unusual visual pattern"

        else:

            label = "Inconclusive"

        result = {
            "label": label,
            "confidence": round(
                float(1 / (1 + normalized)),
                3
            ),
            "reconstruction_error": round(
                reconstruction_error,
                6
            ),
            "distance": round(distance, 3),
            "reconstruction_limit": round(
                reconstruction_limit,
                6
            ),
            "distance_limit": round(
                distance_limit,
                3
            ),
            "model_type": "genuine_reference_anomaly_detection",
            "reference_denomination": reference_denomination,
            "disclaimer": (
                "This is visual reference comparison, not "
                "counterfeit detection. An unusual pattern "
                "does not prove counterfeiting."
            )
        }

        if str(denomination).removeprefix("₹") == "200":

            bundle = _load_200_authenticity_classifier()

            if bundle is not None:

                classifier = bundle["classifier"]
                metadata = bundle["metadata"]
                classes = list(classifier.classes_)
                scores = classifier.predict_proba(
                    embedding[None, ...]
                )[0]
                class_scores = {
                    str(class_name): float(score)
                    for class_name, score in zip(
                        classes,
                        scores
                    )
                }

                fake_score = class_scores.get("fake", 0.0)
                genuine_score = class_scores.get("genuine", 0.0)
                fake_threshold = float(
                    metadata["decision_thresholds"][
                        "fake_score_minimum"
                    ]
                )
                genuine_threshold = float(
                    metadata["decision_thresholds"][
                        "genuine_score_minimum"
                    ]
                )

                if fake_score >= fake_threshold:
                    authenticity_label = "POSSIBLE COUNTERFEIT"
                elif genuine_score >= genuine_threshold:
                    authenticity_label = "LIKELY GENUINE"
                else:
                    authenticity_label = "INCONCLUSIVE"

                result["authenticity_classifier"] = {
                    "label": authenticity_label,
                    "fake_score": round(fake_score, 4),
                    "genuine_score": round(genuine_score, 4),
                    "scores_are_calibrated_probabilities": False,
                    "training_counts": metadata["class_counts"],
                    "validation_status": metadata[
                        "validation_status"
                    ],
                    "warning": (
                        "Experimental: this prediction follows "
                        "user-supplied labels and is not official "
                        "authentication."
                    )
                }

        if str(denomination).removeprefix("₹") == "50":
            try:
                result["experimental_feature_match"] = (
                    _experimental_50_feature_match(embedding)
                )
            except Exception as exc:
                logger.exception(
                    "Experimental ₹50 feature comparison failed."
                )
                result["experimental_feature_match"] = {
                    "status": "unavailable",
                    "message": (
                        "Feature comparison failed: "
                        f"{exc}"
                    )
                }

        return result, None

    except Exception as exc:

        return None, f"AI prediction failed: {exc}"


# ==================================================
# EVALUATION REPORT
# ==================================================

def model_report():

    if not os.path.isfile(EVAL_PATH):
        return None

    try:

        with open(
            EVAL_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return None


# ==================================================
# COMPLETE BANKNOTE ANALYSIS
# ==================================================

def analyze(img, progress):

    all_refs = load_refs()

    progress(0, "Image received and decoded")

    q = quality(img)

    progress(1, "Checking brightness, blur and visibility")

    # Work with a manageable image size.
    scale = min(
        1.0,
        1600 / max(img.shape[:2])
    )

    resized = (
        cv2.resize(img, None, fx=scale, fy=scale)
        if scale < 1
        else img.copy()
    )

    denoised = cv2.fastNlMeansDenoisingColored(
        resized,
        None,
        3,
        3,
        7,
        21
    )

    progress(2, "Detecting banknote boundary")

    note, found, quad = detect_note(denoised)

    matched_denomination = None
    matched_side = None
    matched_reference = None

    compatible_refs = []

    # ----------------------------------------------
    # FIND BEST REFERENCE
    # ----------------------------------------------

    if found and all_refs:

        target_gray = prep_gray(note)

        candidates = []

        for ref in all_refs:

            ref_gray = prep_gray(
                ref["img"]
            )

            match, oriented_ref = orient_orb_match(
                target_gray,
                ref_gray
            )

            score = (
                match["inliers"]
                + match["good"] * 0.1
            )

            if match["inliers"] >= 4:

                candidates.append(
                    (
                        score,
                        ref,
                        match
                    )
                )

        if candidates:

            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            _, best_ref, best_match = candidates[0]

            matched_denomination = best_ref["denomination"]
            matched_side = best_ref["side"]
            matched_reference = best_ref["name"]

            # Include same denomination references.
            compatible_refs = [
                ref for ref in all_refs
                if ref["denomination"] == matched_denomination
                and (
                    ref["side"] == matched_side
                    or ref["side"] == "unknown"
                    or matched_side == "unknown"
                )
            ]

    # ----------------------------------------------
    # LAYOUT AND PREVIEW
    # ----------------------------------------------

    layout = [
        {
            "id": region[0],
            "name": region[1],
            "box": region[2]
        }
        for region in REGIONS + FEATURE_REGIONS
    ]

    progress(
        2,
        "Banknote aligned" if found else "Boundary not detected",
        extra={
            "preview": b64(note),
            "layout": layout,
            "found": found
        }
    )

    progress(3, "Extracting banknote regions")

    items = []

    for rid, name, box, corner in REGIONS + FEATURE_REGIONS:

        region_crop = crop(
            note,
            box
        )

        ok, reason = visibility(
            region_crop
        )

        if not found:

            ok = False

            reason = (
                "The banknote boundary could not be detected. "
                "The region cannot be aligned reliably."
            )

        items.append(
            {
                "id": rid,
                "name": name,
                "box": box,
                "corner": corner,
                "crop": region_crop,
                "visible": ok,
                "why": reason,
                "orb": None,
                "ssim": None,
                "diff": None,
                "ref": None
            }
        )

    # ----------------------------------------------
    # REGION ORB MATCHING
    # ----------------------------------------------

    progress(
        4,
        f"Matching against {len(compatible_refs)} compatible reference images"
    )

    for item in items:

        if not item["visible"] or not compatible_refs:
            continue

        test_gray = prep_gray(
            item["crop"]
        )

        best = None

        for ref in compatible_refs:

            reference_crop = crop(
                ref["img"],
                item["box"]
            )

            reference_gray = prep_gray(
                reference_crop
            )

            match, oriented_ref = orient_orb_match(
                test_gray,
                reference_gray
            )

            score = (
                match["inliers"]
                + match["good"] * 0.1
            )

            if best is None or score > best[0]:

                best = (
                    score,
                    match,
                    oriented_ref,
                    ref["name"]
                )

        if best is not None:

            item["orb"] = best[1]
            item["rg"] = best[2]
            item["ref"] = best[3]
            item["tg"] = test_gray

    # ----------------------------------------------
    # SSIM
    # ----------------------------------------------

    progress(
        5,
        "Computing structural similarity"
    )

    for item in items:

        if item["orb"] is None:
            continue

        similarity, difference = ssim_stage(
            item["tg"],
            item["rg"],
            item["orb"]
        )

        item["ssim"] = similarity

        item["diff"] = cv2.applyColorMap(
            (difference * 255).astype("uint8"),
            cv2.COLORMAP_INFERNO
        )

    # ----------------------------------------------
    # AI MODEL
    # ----------------------------------------------

    progress(
        6,
        "Running genuine-reference anomaly model"
    )

    mp, merr = model_predict(
        note,
        matched_denomination
    )

    progress(7, "Building final report")

    # ----------------------------------------------
    # REGION STATUS
    # ----------------------------------------------

    for item in items:

        orb = item["orb"]
        similarity = item["ssim"]

        if not item["visible"]:

            item["status"] = "yellow"
            item["text"] = item["why"]

        elif orb is None:

            item["status"] = "yellow"
            item["text"] = (
                "No compatible reference comparison was available."
            )

        elif (
            orb["inliers"] >= CFG["inl_ok"]
            and similarity is not None
            and similarity >= CFG["ssim_ok"]
        ):

            item["status"] = "green"
            item["text"] = (
                "This region is visually consistent with the reference."
            )

        elif (
            orb["inliers"] >= CFG["inl_bad"]
            and orb["kp_test"] >= 12
            and orb["kp_ref"] >= 12
            and similarity is not None
            and similarity < CFG["ssim_bad"]
        ):

            item["status"] = "red"
            item["text"] = (
                "This region differs from the corresponding reference."
            )

        else:

            item["status"] = "yellow"
            item["text"] = (
                "Evidence is insufficient for a reliable assessment."
            )

    # ----------------------------------------------
    # OCR
    # ----------------------------------------------

    OCR_REGION_IDS = {
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "serial_number"
    }

    for item in items:

        item["ocr"] = (
            regional_text_assessment(item["crop"])
            if item["id"] in OCR_REGION_IDS
            else None
        )

    # ----------------------------------------------
    # COUNTS
    # ----------------------------------------------

    red = [
        item for item in items
        if item["status"] == "red"
    ]

    assessed = [
        item for item in items
        if item["orb"] is not None
    ]

    corners = [
        item for item in items
        if item["corner"]
    ]

    # ----------------------------------------------
    # FINAL VERDICT
    # ----------------------------------------------

    if q["level"] == "Poor":

        verdict = (
            "INCONCLUSIVE",
            "yellow",
            "Image quality is too poor for reliable screening."
        )

    elif not found:

        verdict = (
            "INCONCLUSIVE",
            "yellow",
            "The banknote boundary could not be detected reliably."
        )

    elif not matched_reference or not compatible_refs:

        verdict = (
            "INCONCLUSIVE",
            "yellow",
            "No sufficiently matching denomination reference was found."
        )

    elif mp is None:

        verdict = (
            "INCONCLUSIVE",
            "yellow",
            "The AI model could not provide a result."
        )

    elif (
        matched_denomination == "200"
        and mp.get("authenticity_classifier") is not None
    ):

        classification = mp["authenticity_classifier"]
        classification_label = classification["label"]

        if classification_label == "LIKELY GENUINE":
            verdict = (
                "LIKELY GENUINE",
                "green",
                "The experimental ₹200 model matched the user-labeled "
                "genuine examples. This is not official authentication."
            )
        elif classification_label == "POSSIBLE COUNTERFEIT":
            verdict = (
                "POSSIBLE COUNTERFEIT",
                "red",
                "The experimental ₹200 model matched the user-labeled "
                "counterfeit examples. This is not proof of counterfeiting; "
                "verify the note with a bank."
            )
        else:
            verdict = (
                "INCONCLUSIVE",
                "yellow",
                "The experimental ₹200 model did not reach its decision "
                "threshold. Physical verification is recommended."
            )

    elif (
        mp["label"] == "Consistent with genuine references"
        and len(assessed) >= 5
        and len(red) == 0
    ):

        verdict = (
            "LIKELY GENUINE",
            "green",
            "The note is visually consistent with available genuine "
            "references. This is not proof of authenticity."
        )

    else:

        verdict = (
            "INCONCLUSIVE",
            "yellow",
            "There is insufficient consistent visual evidence. "
            "Physical verification is recommended."
        )

    # ----------------------------------------------
    # ANNOTATED IMAGE
    # ----------------------------------------------

    annotated = note.copy()

    marker_number = 0
    display_as_genuine = verdict[0] == "LIKELY GENUINE"

    for item in items:

        x0, y0, x1, y1 = item["box"]

        p0 = (
            int(x0 * W),
            int(y0 * H)
        )

        p1 = (
            int(x1 * W) - 2,
            int(y1 * H) - 2
        )

        if display_as_genuine:
            continue

        cv2.rectangle(
            annotated,
            p0,
            p1,
            COL[item["status"]],
            3
        )

        if item["status"] == "red":

            marker_number += 1
            item["marker"] = marker_number

            cv2.circle(
                annotated,
                (p0[0] + 26, p0[1] + 26),
                22,
                COL["red"],
                -1
            )

            cv2.putText(
                annotated,
                str(marker_number),
                (p0[0] + 15, p0[1] + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )

    # ----------------------------------------------
    # AVERAGES
    # ----------------------------------------------

    def avg(key):

        values = []

        for item in assessed:

            if key == "ssim":
                value = item["ssim"]
            else:
                value = item["orb"][key]

            if value is not None:
                values.append(value)

        if not values:
            return None

        return round(
            float(np.mean(values)),
            3
        )

    # ----------------------------------------------
    # REGION REPORT
    # ----------------------------------------------

    regions = []

    for item in items:

        data = {
            "id": item["id"],
            "name": item["name"],
            "corner": item["corner"],
            "status": item["status"],
            "text": item["text"],
            "crop": b64(item["crop"]),
            "marker": item.get("marker"),
            "visual_feature": REGION_FEATURES.get(
                item["id"],
                "Requires physical verification"
            )
        }

        if item.get("ocr"):
            data["ocr"] = item["ocr"]

        if item["orb"] is not None:

            data.update({
                "orb": {
                    "good": item["orb"]["good"],
                    "inliers": item["orb"]["inliers"],
                    "kp": item["orb"]["kp_test"]
                },
                "ssim": round(item["ssim"], 3),
                "diff": b64(item["diff"]),
                "ref": item["ref"]
            })

        regions.append(data)

    # ----------------------------------------------
    # FINAL RESPONSE
    # ----------------------------------------------

    return {

        "verdict": {
            "label": verdict[0],
            "color": verdict[1],
            "reason": verdict[2]
        },

        "annotated": b64(annotated),

        "quality": q,

        "found": found,

        "alignment": {
            "perspective_corrected": bool(found),
            "boundary_detected": bool(found),
            "status": (
                "Aligned"
                if found
                else "Original frame used because the boundary "
                     "could not be detected reliably"
            )
        },

        "overview": {

            "denomination": (
                f"₹{matched_denomination}"
                if matched_denomination
                else "Unknown"
            ),

            "side": matched_side or "Unknown",

            "matched_reference": matched_reference,

            "model": mp,

            "model_error": merr,

            "model_report": model_report(),

            "refs": len(compatible_refs),

            "orb": {
                "mean_good": avg("good"),
                "mean_inliers": avg("inliers")
            },

            "ssim": avg("ssim"),

            "coverage": {
                "assessed": len(assessed),
                "total": len(items),
                "corners": sum(
                    1 for item in corners
                    if item["orb"] is not None
                )
            }
        },

        "regions": regions,

        "suspicious": [
            region for region in regions
            if region["status"] == "red"
        ],

        "physical": PHYSICAL,

        "inspection_criteria": VISUAL_INDICATORS
    }