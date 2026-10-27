import geopandas as gpd

# Local UTM Zone 43N for accurate meter calculations in Bhopal, MP
METRIC_CRS = "EPSG:32643"

def screen_plot(
    plot: gpd.GeoDataFrame,
    water_bodies: gpd.GeoDataFrame,
    drainage: gpd.GeoDataFrame,
    buffer_distance_m: float = 30.0,
    submitted_drainage_plan: bool = True
) -> dict:
    """
    Screen a proposed plot against mapped water bodies, protective buffers,
    and drainage corridors. Does NOT confer legal permission.
    """
    if plot is None or plot.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "water_body_conflicts": 0,
            "drainage_conflicts": 0,
            "buffer_warnings": 0,
            "findings": ["Submitted plot geometry is empty or missing."],
            "note": "Cannot complete assessment without valid plot geometry."
        }

    for name, layer in [("plot", plot), ("water_bodies", water_bodies), ("drainage", drainage)]:
        if layer.crs is None or layer.geometry.isna().all():
            return {
                "status": "INSUFFICIENT_DATA",
                "water_body_conflicts": 0,
                "drainage_conflicts": 0,
                "buffer_warnings": 0,
                "findings": [f"Layer '{name}' is missing a valid Coordinate Reference System (CRS) or geometry."],
                "note": "Further documentation or verified GIS records required."
            }

    plot_m = plot.to_crs(METRIC_CRS)
    water_m = water_bodies.to_crs(METRIC_CRS)
    drainage_m = drainage.to_crs(METRIC_CRS)

    plot_geom = plot_m.geometry.union_all()
    if not plot_geom.is_valid:
        plot_geom = plot_geom.buffer(0)

    # 1. Direct Water Body Overlap
    water_hits = water_m[water_m.geometry.intersects(plot_geom)]
    
    # 2. Direct Drainage Corridor Overlap (with 5m channel width tolerance)
    drainage_hits = drainage_m[drainage_m.geometry.buffer(5.0).intersects(plot_geom)]

    # 3. Protective Buffer Check around Water Bodies
    water_buffered = water_m.copy()
    water_buffered["geometry"] = water_m.geometry.buffer(buffer_distance_m)
    buffer_hits = water_buffered[
        water_buffered.geometry.intersects(plot_geom) & (~water_m.geometry.intersects(plot_geom))
    ]

    findings = []
    for _, row in water_hits.iterrows():
        findings.append(
            f"DIRECT WATER OVERLAP: Intersects '{row['name']}' (ID: {row['feature_id']}, Source: {row['source']}, Date: {row['date']})."
        )
    for _, row in drainage_hits.iterrows():
        findings.append(
            f"DRAINAGE CORRIDOR CONFLICT: Intersects '{row['name']}' ({row['alignment_type']}, Source: {row['source']})."
        )
    for _, row in buffer_hits.iterrows():
        findings.append(
            f"BUFFER ZONE WARNING: Plot falls within the {buffer_distance_m}m investigation buffer of '{row['name']}'."
        )

    if not submitted_drainage_plan:
        findings.append("MISSING DRAINAGE PLAN: Applicant has not submitted a site surface-runoff continuity plan.")

    if not water_hits.empty or not drainage_hits.empty or not buffer_hits.empty or not submitted_drainage_plan:
        status = "REVIEW_REQUIRED"
    else:
        status = "NO_MAPPED_CONFLICT_IDENTIFIED"
        findings.append("No direct spatial intersection found in currently loaded reference layers.")

    return {
        "status": status,
        "water_body_conflicts": int(len(water_hits)),
        "drainage_conflicts": int(len(drainage_hits)),
        "buffer_warnings": int(len(buffer_hits)),
        "submitted_drainage_plan": submitted_drainage_plan,
        "findings": findings,
        "note": (
            "Preliminary technical screening only. This report does not confer legal permission "
            "to build and does not guarantee flood safety. Final decision rests with the authorised municipal authority."
        )
    }