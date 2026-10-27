import joblib
import pandas as pd

MODEL_PATH = "models/landcover_rf.joblib"

def load_ai_bundle():
    """Load the trained Random Forest model and evaluation metrics."""
    return joblib.load(MODEL_PATH)

def classify_spectral_patch(b02: float, b03: float, b04: float, b08: float) -> dict:
    bundle = load_ai_bundle()
    model = bundle["model"]
    features = bundle["features"]

    ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
    ndwi = (b03 - b08) / (b03 + b08 + 1e-6)

    X_input = pd.DataFrame([[b02, b03, b04, b08, ndvi, ndwi]], columns=features)
    predicted_class = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    confidence = float(max(probabilities))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "NDVI": round(ndvi, 3),
        "NDWI": round(ndwi, 3),
        "class_probabilities": dict(zip(bundle["classes"], [round(float(p), 3) for p in probabilities]))
    }

def evaluate_temporal_change(date1_bands: dict, date2_bands: dict, overlaps_sensitive_zone: bool = True) -> dict:
    res_t1 = classify_spectral_patch(**date1_bands)
    res_t2 = classify_spectral_patch(**date2_bands)

    c1 = res_t1["predicted_class"]
    c2 = res_t2["predicted_class"]
    avg_conf = round((res_t1["confidence"] + res_t2["confidence"]) / 2.0, 2)

    if avg_conf < 0.60:
        flag_type = "Low-confidence finding (Verification Required)"
    elif c1 == "Water" and c2 in ["Built-up", "Bare Soil"] and overlaps_sensitive_zone:
        flag_type = "Possible water-body encroachment / land filling"
    elif c1 != "Built-up" and c2 == "Built-up" and overlaps_sensitive_zone:
        flag_type = "Possible drainage obstruction / new impervious build-up"
    elif c1 == c2:
        flag_type = "No major land-cover class transition detected"
    else:
        flag_type = f"Land-cover transition ({c1} -> {c2}); review context"

    return {
        "t1_result": res_t1,
        "t2_result": res_t2,
        "transition": f"{c1} -> {c2}",
        "combined_confidence": avg_conf,
        "screening_flag": flag_type,
        "disclaimer": "Screening classification only (10m Sentinel-2 resolution). Requires official survey & human verification."
    }