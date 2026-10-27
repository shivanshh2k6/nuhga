import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix

os.makedirs("src", exist_ok=True)
os.makedirs("data/demo", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("docs", exist_ok=True)

with open("src/__init__.py", "w") as f:
    f.write("# NUHGA 2.0 source package\n")

# Realistic multi-vertex boundary for Shahpura Lake catchment (~23.203 N, 77.425 E)
pilot_boundary = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [{
        "type": "Feature",
        "properties": {
            "id": "PILOT-BPL-SHAHPURA",
            "name": "Shahpura Lake Sub-Catchment Priority Zone",
            "source": "ISRO CartoDEM Hydrology Delineation & BMC Ward Survey",
            "authority_status": "Delineated Catchment Boundary"
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [77.4120, 23.1900], [77.4165, 23.1895], [77.4230, 23.1890],
                [77.4320, 23.1915], [77.4385, 23.1960], [77.4410, 23.2040],
                [77.4395, 23.2110], [77.4325, 23.2160], [77.4240, 23.2170],
                [77.4170, 23.2140], [77.4115, 23.2060], [77.4105, 23.1980],
                [77.4120, 23.1900]
            ]]
        }
    }]
}

# Detailed contours of Shahpura Lake (Water Spread & Full Tank Level)
water_bodies = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "feature_id": "WB-SHAHPURA-CORE",
                "name": "Shahpura Lake (Core Water Spread / FTL)",
                "source": "EPCO State Wetland Inventory / Landsat-Sentinel Series",
                "date": "2025-11",
                "authority_status": "Protected State Wetland Asset",
                "buffer_m": 30
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4202, 23.2005], [77.4208, 23.2025], [77.4215, 23.2045],
                    [77.4225, 23.2065], [77.4242, 23.2080], [77.4260, 23.2082],
                    [77.4278, 23.2075], [77.4290, 23.2055], [77.4295, 23.2035],
                    [77.4292, 23.2015], [77.4285, 23.1995], [77.4275, 23.1980],
                    [77.4260, 23.1968], [77.4245, 23.1965], [77.4230, 23.1970],
                    [77.4215, 23.1982], [77.4202, 23.2005]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "feature_id": "WB-MANISHA-RETENTION",
                "name": "Manisha Sector Retention Basin",
                "source": "MP Forest Dept City Biodiversity Index",
                "date": "2024-10",
                "authority_status": "Mapped Natural Asset",
                "buffer_m": 15
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4318, 23.2015], [77.4335, 23.2018], [77.4348, 23.2032],
                    [77.4342, 23.2045], [77.4322, 23.2042], [77.4312, 23.2028],
                    [77.4318, 23.2015]
                ]]
            }
        }
    ]
}

# Natural meandering drainage streams
drainage_corridors = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "feature_id": "DRN-NORTH-PRIMARY",
                "name": "Northern Chunabhatti Trunk Stormwater Inlet",
                "source": "Bhopal Municipal Corporation Master Drainage Plan",
                "date": "2025-06",
                "alignment_type": "Official Mapped Drainage Corridor",
                "buffer_m": 15
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [77.4238, 23.2162], [77.4241, 23.2135], [77.4239, 23.2110],
                    [77.4246, 23.2095], [77.4252, 23.2082]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "feature_id": "DRN-WEST-RUNOFF",
                "name": "Bhadbhada Ridge Surface Flow Pathway",
                "source": "ISRO CartoDEM Terrain Flow Accumulation Analysis",
                "date": "2026-01",
                "alignment_type": "Modelled Topographic Pathway (Requires Survey)",
                "buffer_m": 10
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [77.4125, 23.2062], [77.4150, 23.2052], [77.4172, 23.2045],
                    [77.4188, 23.2032], [77.4208, 23.2025]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "feature_id": "DRN-SOUTH-SPILLWAY",
                "name": "Kaliasot River Outflow & Surplus Spillway",
                "source": "Water Resources Department (WRD) Alignment",
                "date": "2025-06",
                "alignment_type": "Official Mapped Drainage Corridor",
                "buffer_m": 20
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [77.4248, 23.1965], [77.4255, 23.1942], [77.4262, 23.1920],
                    [77.4268, 23.1895]
                ]
            }
        }
    ]
}

# Realistic parcel footprints for screening tests
sample_plots = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "plot_id": "KHASRA-412/1 (Shoreline Filling Violation)",
                "applicant": "Apex Infra City Hub (Simulated Commercial)",
                "submitted_drainage_plan": False,
                "description": "Encroaches 42m into northeastern Shahpura Lake conservation boundary."
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4272, 23.2062], [77.4292, 23.2060], [77.4290, 23.2045],
                    [77.4270, 23.2048], [77.4272, 23.2062]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "plot_id": "KHASRA-188/3 (Trunk Drain Obstruction)",
                "applicant": "Green Meadows Residency (Simulated Housing)",
                "submitted_drainage_plan": True,
                "description": "Constructed directly over the Northern Chunabhatti Trunk Stormwater channel."
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4230, 23.2115], [77.4252, 23.2115], [77.4252, 23.2132],
                    [77.4230, 23.2132], [77.4230, 23.2115]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "plot_id": "KHASRA-95/A (Buffer Proximity Zone)",
                "applicant": "Civic Health Sub-Center (Simulated Public Sector)",
                "submitted_drainage_plan": True,
                "description": "Outside high-water line, but within mandatory 30m NGT buffer zone."
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4210, 23.1970], [77.4225, 23.1970], [77.4225, 23.1979],
                    [77.4210, 23.1979], [77.4210, 23.1970]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "plot_id": "KHASRA-620/8 (Compliant Upland Development)",
                "applicant": "Bhopal Smart Office Suites (Simulated Mixed-Use)",
                "submitted_drainage_plan": True,
                "description": "Safe elevated zone with no water bodies, buffers, or natural drainage corridors."
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4140, 23.1925], [77.4168, 23.1925], [77.4168, 23.1950],
                    [77.4140, 23.1950], [77.4140, 23.1925]
                ]]
            }
        }
    ]
}

# Spectral AI candidate encroachment detections
sample_ai_flags = {
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "properties": {
                "flag_id": "FLAG-SHP-2026-01",
                "classification": "Suspected Wetland Reclamation / Land Filling",
                "change_detected": "Open Water -> Compacted Built-up Surface",
                "imagery_dates": "Sentinel-2 Multi-Spectral: Nov 2021 vs Feb 2026",
                "confidence": 0.94,
                "priority": "High Risk",
                "reason": "Abrupt NDWI drop (+0.41 to -0.22) & NDBI increase along the eastern shoreline bund."
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4276, 23.2060], [77.4290, 23.2058], [77.4288, 23.2046],
                    [77.4274, 23.2050], [77.4276, 23.2060]
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "flag_id": "FLAG-SHP-2026-02",
                "classification": "Suspected Drainage Culvert Paving",
                "change_detected": "Riparian Vegetation -> Impervious Concrete",
                "imagery_dates": "Sentinel-2 Multi-Spectral: Nov 2021 vs Feb 2026",
                "confidence": 0.88,
                "priority": "High Risk",
                "reason": "NDVI drop (+0.58 to +0.12) directly intersecting Northern Trunk Inlet alignment."
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.4234, 23.2118], [77.4248, 23.2118], [77.4248, 23.2128],
                    [77.4234, 23.2128], [77.4234, 23.2118]
                ]]
            }
        }
    ]
}

geojson_files = {
    "data/demo/pilot_boundary.geojson": pilot_boundary,
    "data/demo/water_bodies.geojson": water_bodies,
    "data/demo/drainage_corridors.geojson": drainage_corridors,
    "data/demo/sample_plots.geojson": sample_plots,
    "data/demo/sample_ai_flags.geojson": sample_ai_flags,
}

for path, content in geojson_files.items():
    with open(path, "w") as f:
        json.dump(content, f, indent=2)

# Train the Random Forest Model with synthetic Sentinel-2 signatures
np.random.seed(42)
n_per_class = 200

water_b = np.column_stack([np.random.normal(0.08, 0.015, n_per_class), np.random.normal(0.11, 0.015, n_per_class), np.random.normal(0.06, 0.015, n_per_class), np.random.normal(0.03, 0.01, n_per_class)])
built_b = np.column_stack([np.random.normal(0.18, 0.025, n_per_class), np.random.normal(0.20, 0.025, n_per_class), np.random.normal(0.25, 0.025, n_per_class), np.random.normal(0.23, 0.025, n_per_class)])
veg_b = np.column_stack([np.random.normal(0.05, 0.012, n_per_class), np.random.normal(0.09, 0.015, n_per_class), np.random.normal(0.05, 0.012, n_per_class), np.random.normal(0.42, 0.04, n_per_class)])
bare_b = np.column_stack([np.random.normal(0.14, 0.02, n_per_class), np.random.normal(0.17, 0.02, n_per_class), np.random.normal(0.22, 0.025, n_per_class), np.random.normal(0.26, 0.025, n_per_class)])

raw_X = np.clip(np.vstack([water_b, built_b, veg_b, bare_b]), 0.01, 0.99)
labels = np.array(["Water"]*n_per_class + ["Built-up"]*n_per_class + ["Vegetation"]*n_per_class + ["Bare Soil"]*n_per_class)

df = pd.DataFrame(raw_X, columns=["B02", "B03", "B04", "B08"])
df["NDVI"] = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
df["NDWI"] = (df["B03"] - df["B08"]) / (df["B03"] + df["B08"])
df["class_label"] = labels

train_idx, test_idx = [], []
for c_idx in range(4):
    start = c_idx * n_per_class
    split = start + int(n_per_class * 0.75)
    train_idx.extend(range(start, split))
    test_idx.extend(range(split, start + n_per_class))

FEATURES = ["B02", "B03", "B04", "B08", "NDVI", "NDWI"]
X_train, y_train = df.iloc[train_idx][FEATURES], df.iloc[train_idx]["class_label"]
X_test, y_test = df.iloc[test_idx][FEATURES], df.iloc[test_idx]["class_label"]

rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

metrics_bundle = {
    "model": rf_model,
    "features": FEATURES,
    "classes": list(rf_model.classes_),
    "precision_macro": round(float(precision_score(y_test, y_pred, average="macro")), 3),
    "recall_macro": round(float(recall_score(y_test, y_pred, average="macro")), 3),
    "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(rf_model.classes_)).tolist(),
    "sample_test_df": df.iloc[test_idx].reset_index(drop=True)
}
joblib.dump(metrics_bundle, "models/landcover_rf.joblib")

print("SUCCESS: Natural Shahpura geometries & upgraded AI model successfully generated!")