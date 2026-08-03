"""
noisy_feature_extraction_v1.py

Shared feature extraction module for the urban noise proxy pipeline.
Import this module in all scripts to ensure consistent feature computation:

    from noisy_feature_extraction_v1 import (
        FEATURE_COLS, get_osm_data, extract_features,
        load_and_join_noise, get_utm_crs, CITY_NAMES, CITY_CENTRES
    )

FEATURE SET (20 features):
  Global city fingerprints (6 — constants per city, activate in unified model):
    circuity_global, dead_end_fraction, avg_block_length_m,
    avg_node_degree_global, street_orientation_entropy_global,
    link_node_ratio_global

  Network/road composition (9):
    major_road_fraction, medium_road_fraction, local_road_fraction,
    pedestrian_fraction, link_node_ratio, road_density,
    width_weighted_road_density, class_weighted_road_density (2^class, cap=100)

  Acoustic environment (5):
    canyon_ratio, building_height_std, street_width_m,
    building_density, green_fraction, water_fraction

WEIGHTING:
  class_weighted_road_density uses 2^class weighting (capped at 100).
  Justified by power-law traffic flow distribution (Lämmer et al., 2006).
  Motorway:residential ratio = 25x (close to actual ~30x).

CHANGELOG:
  v1.0 — Initial extraction from noise_proxy_pipeline_v5.py
         Dropped: betweenness_centrality, straightness, highway_class,
                  oneway, lanes, maxspeed, abs_grade, seg_length_m
         Added:   building_height_std, water_fraction, dead_end_fraction,
                  avg_block_length_m, avg_node_degree_global,
                  street_orientation_entropy_global, circuity_global,
                  link_node_ratio_global
         Weighting: class_weighted_road_density -> 2^class (capped 100)
"""

"""
Noise Proxy Model Pipeline v5

USAGE:
  # Single city run (reads city from NOISE_FILE name):
  python noise_proxy_pipeline_v5.py

  # Cross-city transfer test:
  python noise_proxy_pipeline_v5.py --transfer noise_contours_pittsburgh.gpkg noise_contours_amsterdam.gpkg

CITY SWITCHING:
  Change only one line: NOISE_FILE = "noise_contours_<city>.gpkg"
  Everything else (city name, CRS, cache, output files) is auto-derived.
"""

import os
import re
import sys
import warnings
import joblib

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    print("  Note: xgboost not installed. Run: pip install xgboost")
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")


DOWNLOAD_RADIUS_M = 2000  # consistent with OpeNoise tile size

# Buffer around each segment for noise polygon matching
BUFFER_DIST_M = 25

# Where to save all outputs
OUTPUT_DIR = "."

# =============================================================================
# AUTO-DERIVED FROM FILENAME — do not edit
# =============================================================================
CITY_NAMES = {
    "pittsburgh": "Pittsburgh, Pennsylvania, USA",
    "amsterdam":  "Amsterdam, Netherlands",
    "zurich":     "Zurich, Switzerland",
    "singapore":  "Singapore",
    "bangkok":    "Bangkok, Thailand",
    "london":     "London, United Kingdom",
    "tokyo":      "Tokyo, Japan",
    "newyork":    "New York City, New York, USA",
    "chicago":    "Chicago, Illinois, USA",
    "sydney":     "Sydney, New South Wales, Australia",
}

# Explicit centre coordinates — must match what prepare_openoise_clipped.py used.
# This ensures the pipeline downloads OSM data from the SAME location that
# OpeNoise was run on. If these differ, noise and segments won't overlap.
CITY_CENTRES = {
    "pittsburgh": (40.4406,  -79.9959),
    "amsterdam":  (52.3676,    4.9041),
    "zurich":     (47.3769,    8.5417),
    "singapore":  ( 1.2897,  103.8501),
    "bangkok":    (13.7563,  100.5018),
    "london":     (51.5074,   -0.1278),
    "tokyo":      (35.6762,  139.6503),
    "newyork":    (40.7128,  -74.0060),
    "chicago":    (41.8781,  -87.6298),
    "sydney":     (-33.8688, 151.2093),
    "delhi":      (28.6139,   77.2090),
    "seoul":      (37.5665,  126.9780),
    "singapore":  ( 1.2897,  103.8501),
    "mexicocity":  (19.4326,  -99.1332),
    "mogadishu":   ( 2.0469,   45.3182),
    "detroit":     (42.305179452631585,  -83.10850391263159), # for this one, we read the mean corodinate from sensor data
    "denver":      (39.7392, -104.9903),
    "phoenix":      (33.4484, -112.0740),
    "cleveland":      (41.4993, -81.6944),
    "jakarta":      (-6.2088, 106.8456),
    "atlanta":      (33.7490, -84.3880),
    "istanbul":      (41.0082, 28.9784),
}

HIGHWAY_ORDER = {
    
    "motorway": 9, "motorway_link": 8,
    "trunk": 7, "trunk_link": 6,
    "primary": 6, "primary_link": 5,
    "secondary": 5, "secondary_link": 4,
    "tertiary": 4, "tertiary_link": 3,
    "residential": 2, "living_street": 1,
    "pedestrian": 0, "footway": 0,
    "path": 0, "cycleway": 0,
    "unclassified": 2, "service": 1,
}

SURFACE_MAP = {
    "asphalt": 0, "concrete": 0,
    "paving_stones": 1, "cobblestone": 1, "sett": 1,
    "unhewn_cobblestone": 1,
    "gravel": 2, "compacted": 2, "fine_gravel": 2,
    "ground": 2, "dirt": 2,
}

# =============================================================================
# HELPERS
# =============================================================================
def _scalar(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    if hasattr(val, "__len__") and not isinstance(val, str):
        try:
            val = list(val)[0]
        except Exception:
            return None
    return val


def _hw_class(val):
    val = _scalar(val)
    return HIGHWAY_ORDER.get(str(val), 2) if val is not None else 2


def _lanes(val):
    val = _scalar(val)
    if val is None:
        return 1.0
    try:
        f = float(val)
        return f if not np.isnan(f) else 1.0
    except Exception:
        return 1.0


def _speed(val):
    val = _scalar(val)
    if val is None:
        return 30.0
    try:
        f = float(str(val).replace("mph", "").replace("km/h", "").strip())
        return f if not np.isnan(f) else 30.0
    except Exception:
        return 30.0


def _surface(val):
    val = _scalar(val)
    return SURFACE_MAP.get(str(val) if val else None, 0)


def get_centre_from_noise_file(noise_file):
    """
    Derive the centre lat/lon from the noise contour GeoPackage itself.
    This is the most reliable approach — the noise file IS the ground truth
    of where OpeNoise ran, so OSM download should match it exactly.
    Returns (lat, lon) in WGS84.
    """
    noise = gpd.read_file(noise_file)
    noise_wgs = noise.to_crs("EPSG:4326")
    bounds    = noise_wgs.total_bounds          # minx, miny, maxx, maxy
    centre_lon = (bounds[0] + bounds[2]) / 2
    centre_lat = (bounds[1] + bounds[3]) / 2
    print("      Centre from noise file: lat={:.5f}, lon={:.5f}".format(
        centre_lat, centre_lon))
    return centre_lat, centre_lon


def get_utm_crs(noise_file=None, place_name=None):
    """
    Derive UTM CRS.
    Priority: (1) noise file centroid, (2) CITY_CENTRES dict, (3) geocode.
    Always prefer the noise file — it guarantees alignment with OpeNoise tile.
    """
    if noise_file and os.path.exists(noise_file):
        lat, lon = get_centre_from_noise_file(noise_file)
    elif _city_slug in CITY_CENTRES:
        lat, lon = CITY_CENTRES[_city_slug]
        print("      Centre from CITY_CENTRES: lat={}, lon={}".format(lat, lon))
    else:
        gdf  = ox.geocode_to_gdf(place_name or CITY)
        c    = gdf.geometry.centroid.iloc[0]
        lat, lon = c.y, c.x
        print("      Geocoded centre: lat={:.4f}, lon={:.4f}".format(lat, lon))
    zone = int((lon + 180) / 6) + 1
    hemi = "6" if lat >= 0 else "7"
    epsg = "EPSG:32{}{}".format(hemi, str(zone).zfill(2))
    print("      Auto CRS: {} (UTM zone {})".format(epsg, zone))
    return epsg, lat, lon


# =============================================================================
# STEP 1: Download OSM data
# =============================================================================
def get_osm_data(city, lat, lon):
    """
    Download streets and buildings using a fixed radius from city centre.
    lat/lon derived from noise file centroid — guarantees alignment with OpeNoise.
    Point+radius keeps memory bounded for large cities (NYC = 1.2M buildings).
    """
    print("[1/5] Fetching OSM data for: {} (radius={}m)".format(
        city, DOWNLOAD_RADIUS_M))
    print("      Centre: lat={:.5f}, lon={:.5f}".format(lat, lon))

    # Centre is passed in from get_utm_crs which derives it from the noise file
    # so OSM download always covers the same area OpeNoise ran on

    G = ox.graph_from_point(
        (lat, lon), dist=DOWNLOAD_RADIUS_M,
        network_type="all", retain_all=True   # keep peripheral segments near rivers/boundaries
    )

    # Add elevation and slope via Google Elevation API (OSMnx 2.x)
    # Free tier: 40,000 requests/day — more than enough for a 3km tile (~20k nodes)
    # Get a free API key at: https://console.cloud.google.com -> Elevation API
    # Set GOOGLE_ELEVATION_API_KEY environment variable, or paste key below.
    import os as _os
    _api_key = _os.environ.get("GOOGLE_ELEVATION_API_KEY", "")
    if not _api_key:
        for _fname in ["google_map_api.txt", "google_maps_api.txt",
                       "google_api.txt", "api_key.txt"]:
            if _os.path.exists(_fname):
                _api_key = open(_fname).read().strip()
                if _api_key:
                    print("      Using API key from {}".format(_fname))
                    break
    if _api_key:
        try:
            G = ox.add_node_elevations_google(G, api_key=_api_key)
            G = ox.add_edge_grades(G, add_absolute=True)
            _grades = pd.Series(
                [d.get("grade_abs", 0) for _, _, d in G.edges(data=True)]
            ).fillna(0)
            print("      Road slope: mean={:.3f}, max={:.3f}".format(
                _grades.mean(), _grades.max()))
        except Exception as e:
            print("      Elevation API error: {}".format(str(e)[:80]))
    else:
        # No API key — try downloading a free SRTM GeoTIFF manually.
        # Instructions:
        #   1. Go to https://dwtkns.com/srtm30m/ (free, no login for most tiles)
        #   2. Click on your city tile to download the .tif
        #   3. Save as e.g. dem_pittsburgh.tif in your working directory
        #   4. The pipeline will auto-detect it below
        import glob as _glob
        _dem_candidates = (
            _glob.glob("dem_{}.tif".format(_city_slug)) +
            _glob.glob("dem_{}*.tif".format(_city_slug)) +
            _glob.glob("*.tif")
        )
        if _dem_candidates:
            _dem_path = _dem_candidates[0]
            try:
                G = ox.add_node_elevations_raster(G, _dem_path)
                G = ox.add_edge_grades(G, add_absolute=True)
                _grades = pd.Series(
                    [d.get("grade_abs", 0) for _, _, d in G.edges(data=True)]
                ).fillna(0)
                print("      Road slope from {}: mean={:.3f}, max={:.3f}".format(
                    _dem_path, _grades.mean(), _grades.max()))
            except Exception as e:
                print("      Raster elevation error: {}".format(str(e)[:80]))
        else:
            print("      No elevation data — set GOOGLE_ELEVATION_API_KEY env var")
            print("      or place a DEM .tif file in the working directory.")
            print("      Slope set to 0 for this run (abs_grade feature inactive).")

    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()

    buildings = ox.features_from_point(
        (lat, lon), dist=DOWNLOAD_RADIUS_M,
        tags={"building": True}
    )
    buildings = buildings[
        buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    # Green landuse — parks, forests, grass, meadows, gardens
    # Used for green_fraction feature (vegetation absorbs 3-5 dB per 10m depth)
    try:
        green = ox.features_from_point(
            (lat, lon), dist=DOWNLOAD_RADIUS_M,
            tags={"landuse": ["grass", "forest", "meadow", "greenfield",
                              "recreation_ground", "village_green"],
                  "leisure": ["park", "garden", "nature_reserve"],
                  "natural": ["wood", "scrub", "heath", "grassland"]}
        )
        green = green[
            green.geometry.type.isin(["Polygon", "MultiPolygon"])
        ].copy()
    except Exception:
        green = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Water bodies — rivers, canals, lakes reflect and extend noise propagation
    try:
        water = ox.features_from_point(
            (lat, lon), dist=DOWNLOAD_RADIUS_M,
            tags={"natural":  ["water", "bay"],
                  "waterway": ["river", "canal", "stream"],
                  "landuse":  ["reservoir", "basin"]}
        )
        water = water[
            water.geometry.type.isin(["Polygon", "MultiPolygon"])
        ].copy()
    except Exception:
        water = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    print("      {} segments | {} buildings | {} green | {} water".format(
        len(edges), len(buildings), len(green), len(water)))
    return edges, buildings, green, water, G


# =============================================================================
# STEP 2: Extract morphological features
# =============================================================================
# =============================================================================
# FEATURE HELPER FUNCTIONS
# =============================================================================

def _buf_nearby(sindex, gdf, geom, buffer_m):
    """Shared spatial index lookup — buffer geom, find nearby rows in gdf."""
    buf        = geom.buffer(buffer_m)
    candidates = list(sindex.intersection(buf.bounds))
    nearby     = gdf.iloc[candidates]
    return buf, nearby[nearby.geometry.intersects(buf)]


def _canyon_ratio(edges_m, buildings_m, buffer_m=40):
    """
    Computes three features simultaneously (same loop, zero extra cost):

    canyon_ratio       = mean_building_height / street_width
    street_width_m     = estimated street width in metres
    building_height_std = std dev of building heights within buffer

    building_height_std captures acoustic diffusion — uniform height creates
    regular reflections and potential standing waves; mixed heights create
    diffuse scattering that reduces noise trapping. Zero extra computation
    since we already iterate over nearby buildings.
    """
    if "height" in buildings_m.columns:
        buildings_m["h"] = pd.to_numeric(buildings_m["height"], errors="coerce")
    else:
        buildings_m["h"] = np.nan
    if "building:levels" in buildings_m.columns:
        lvl = pd.to_numeric(buildings_m["building:levels"], errors="coerce")
        buildings_m["h"] = buildings_m["h"].fillna(lvl * 3.0)
    buildings_m["h"] = buildings_m["h"].fillna(9.0)

    sindex   = buildings_m.sindex
    ratios   = []
    widths   = []
    h_stds   = []
    for geom in tqdm(edges_m.geometry, desc="      canyon ratio + width + h_std", unit="seg"):
        _, nearby = _buf_nearby(sindex, buildings_m, geom, buffer_m)
        b     = geom.bounds
        width = max(min(b[2]-b[0], b[3]-b[1]), 3.0)
        widths.append(width)
        if len(nearby) == 0:
            ratios.append(0.0)
            h_stds.append(0.0)
            continue
        h_vals = nearby["h"].values
        ratios.append(float(h_vals.mean()) / width)
        h_stds.append(float(h_vals.std()) if len(h_vals) > 1 else 0.0)
    return (pd.Series(ratios, index=edges_m.index),
            pd.Series(widths, index=edges_m.index),
            pd.Series(h_stds, index=edges_m.index))


def _building_density(edges_m, buildings_m, buffer_m=100):
    """
    Fraction of buffer area covered by building footprints.
    High density = more reflective surfaces = more diffuse noise environment.
    """
    buf_area = np.pi * buffer_m ** 2
    sindex   = buildings_m.sindex
    densities = []
    for geom in tqdm(edges_m.geometry, desc="      building density", unit="seg"):
        buf, nearby = _buf_nearby(sindex, buildings_m, geom, buffer_m)
        if len(nearby) == 0:
            densities.append(0.0)
            continue
        footprint = nearby.geometry.intersection(buf).area.sum()
        densities.append(footprint / buf_area)
    return pd.Series(densities, index=edges_m.index)


def _road_class_fractions(edges_m, buffer_m=200):
    """
    Four-way compositional road class split — fractions sum to exactly 1.0.

    major_road_fraction   class >= 5   motorway, trunk, primary, secondary
                          dominant noise sources, high traffic volume
    medium_road_fraction  class 3-4    tertiary, tertiary_link
                          moderate traffic, transition zone between major and local
    local_road_fraction   class 1-2    residential, service, living_street, unclassified
                          low traffic, background noise contributor
    pedestrian_fraction   class == 0   footway, path, cycleway, pedestrian
                          no motor traffic — acoustic gaps in the environment

    Compositional (simplex) structure means the four fractions are not
    independent — knowing three determines the fourth. In practice RF handles
    this correctly since it uses splits not correlations. The compositional
    structure makes the feature set interpretable as a road type "recipe".

    Single spatial index loop — four outputs for the cost of one query.
    """
    sindex = edges_m.sindex
    hw     = edges_m["highway_class"].values

    major_fracs  = []
    medium_fracs = []
    local_fracs  = []
    ped_fracs    = []

    for geom in tqdm(edges_m.geometry, desc="      road class fractions", unit="seg"):
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex.intersection(buf.bounds))
        if not candidates:
            major_fracs.append(0.0); medium_fracs.append(0.0)
            local_fracs.append(0.0); ped_fracs.append(0.0)
            continue
        nearby_geoms = edges_m.geometry.iloc[candidates]
        mask         = nearby_geoms.intersects(buf).values
        idx          = np.array(candidates)[mask]
        if len(idx) == 0:
            major_fracs.append(0.0); medium_fracs.append(0.0)
            local_fracs.append(0.0); ped_fracs.append(0.0)
            continue
        hw_nearby = hw[idx]
        n         = len(hw_nearby)
        major_fracs.append( (hw_nearby >= 5).sum() / n)
        medium_fracs.append(((hw_nearby >= 3) & (hw_nearby <= 4)).sum() / n)
        local_fracs.append( ((hw_nearby >= 1) & (hw_nearby <= 2)).sum() / n)
        ped_fracs.append(   (hw_nearby == 0).sum() / n)

    idx_out = edges_m.index
    return (pd.Series(major_fracs,  index=idx_out),
            pd.Series(medium_fracs, index=idx_out),
            pd.Series(local_fracs,  index=idx_out),
            pd.Series(ped_fracs,    index=idx_out))


def _green_fraction(edges_m, green_m, buffer_m=100):
    """
    Fraction of buffer area covered by OSM green landuse.
    Vegetation attenuates 3-5 dB per 10m depth (ISO 9613-2).
    """
    buf_area = np.pi * buffer_m ** 2
    if len(green_m) == 0:
        return pd.Series(0.0, index=edges_m.index)
    sindex    = green_m.sindex
    fractions = []
    for geom in tqdm(edges_m.geometry, desc="      green fraction", unit="seg"):
        buf, nearby = _buf_nearby(sindex, green_m, geom, buffer_m)
        if len(nearby) == 0:
            fractions.append(0.0)
            continue
        fractions.append(min(nearby.geometry.intersection(buf).area.sum() / buf_area, 1.0))
    return pd.Series(fractions, index=edges_m.index)


def _intersection_density(edges_m, G, buffer_m=200):
    """
    Number of intersections (degree >= 3 nodes) per km2 within buffer.
    High density = more conflict points = more braking/acceleration events = more noise.
    Scale-invariant (per unit area).
    """
    import networkx as nx
    # Intersections = nodes with degree >= 3 in undirected graph
    G_undir = G.to_undirected()
    inter_nodes = {n for n, d in G_undir.degree() if d >= 3}
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    inter_gdf = nodes_gdf[nodes_gdf.index.isin(inter_nodes)].copy()

    if len(inter_gdf) == 0:
        return pd.Series(0.0, index=edges_m.index)

    buf_area_km2 = np.pi * (buffer_m / 1000) ** 2
    sindex       = inter_gdf.sindex
    densities    = []
    for geom in tqdm(edges_m.geometry, desc="      intersection density", unit="seg"):
        _, nearby = _buf_nearby(sindex, inter_gdf, geom, buffer_m)
        densities.append(len(nearby) / buf_area_km2)
    return pd.Series(densities, index=edges_m.index)


def _straightness(edges_m):
    """
    Ratio of straight-line (Euclidean) distance between endpoints to actual
    path length of each segment. Range [0, 1].
    1.0 = perfectly straight (acts as noise channel, long propagation distances).
    < 1.0 = curved (breaks line-of-sight, increases diffraction losses).
    Novel acoustic feature — curved streets attenuate noise faster than straight ones.
    """
    from shapely.geometry import Point
    values = []
    for geom in tqdm(edges_m.geometry, desc="      straightness", unit="seg"):
        coords     = list(geom.coords)
        if len(coords) < 2:
            values.append(1.0)
            continue
        path_len   = geom.length
        if path_len < 0.01:
            values.append(1.0)
            continue
        eucl_dist  = Point(coords[0]).distance(Point(coords[-1]))
        values.append(min(eucl_dist / path_len, 1.0))
    return pd.Series(values, index=edges_m.index)


def _betweenness_centrality(G, edges_m, k=500):
    """
    Approximate edge betweenness centrality using k random source nodes.
    k=500 gives good approximation in ~10-20x less time than full computation.
    Dimensionless and scale-invariant — same meaning across all cities.
    High centrality = many shortest paths use this segment = more traffic = more noise.
    """
    import networkx as nx
    import time
    n_nodes = G.number_of_nodes()
    k_use   = min(k, n_nodes)
    print("      betweenness centrality: k={} of {} nodes...".format(k_use, n_nodes))
    t0 = time.time()
    bc = nx.edge_betweenness_centrality(G, normalized=True, weight="length", k=k_use)
    print("      done in {:.0f}s".format(time.time() - t0))

    bc_vals = []
    for _, row in tqdm(edges_m.iterrows(), total=len(edges_m),
                       desc="      mapping centrality", unit="seg"):
        u   = row.get("u")
        v   = row.get("v")
        key = row.get("key", 0)
        bc_vals.append(bc.get((u, v, key), bc.get((v, u, key), 0.0)))
    return pd.Series(bc_vals, index=edges_m.index)


def _intersection_density(edges_m, G, buffer_m=200):
    """
    Number of intersections (degree >= 3 nodes) per km² within buffer.

    Directly models the junction noise contribution in CNOSSOS-EU:
    the standard explicitly includes a dist_intersection parameter that
    increases emission near junctions due to braking, acceleration, and
    idling behaviour. More intersections per unit area = more stop-start
    events = higher noise above the free-flow emission level.

    Also captures network complexity independently of road class —
    a residential grid with many intersections differs acoustically
    from a residential cul-de-sac network even with the same road class mix.

    Fast computation: node degrees already available from G, no spatial
    loop needed. Junction positions projected to metric CRS for accurate
    buffer intersection.

    Replaces betweenness_centrality (2-3 min, 0.018 importance) with a
    faster and more physically direct feature (~10 seconds total).
    """
    import networkx as nx
    G_undir     = G.to_undirected()
    inter_nodes = {n for n, d in G_undir.degree() if d >= 3}

    # Get intersection node positions in metric CRS
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    inter_gdf    = nodes_gdf[nodes_gdf.index.isin(inter_nodes)].copy()

    if len(inter_gdf) == 0:
        return pd.Series(0.0, index=edges_m.index)

    # Ensure same CRS as edges_m
    inter_gdf    = inter_gdf.to_crs(edges_m.crs)
    buf_area_km2 = np.pi * (buffer_m / 1000) ** 2
    sindex_i     = inter_gdf.sindex
    densities    = []

    for geom in tqdm(edges_m.geometry,
                     desc="      intersection density", unit="seg"):
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex_i.intersection(buf.bounds))
        if not candidates:
            densities.append(0.0)
            continue
        nearby = inter_gdf.iloc[candidates]
        nearby = nearby[nearby.geometry.within(buf)]
        densities.append(len(nearby) / buf_area_km2)
    return pd.Series(densities, index=edges_m.index)


def _link_node_ratio(edges_m, buffer_m=200):
    """
    Network connectivity = edges / nodes within buffer.
    Scale-invariant: ~1.0 for cul-de-sac networks, ~2.0+ for Manhattan grid.
    High connectivity -> more through-traffic routes -> more distributed noise.
    Low connectivity (dead ends) -> less through traffic -> quieter.

    Fast vectorised implementation — avoids iterrows() on nearby edges.
    Uses pre-extracted u/v arrays for node counting via numpy unique.
    """
    sindex = edges_m.sindex

    # Pre-extract node arrays for fast lookup (avoid per-row pandas overhead)
    u_arr = edges_m["u"].values if "u" in edges_m.columns else np.zeros(len(edges_m))
    v_arr = edges_m["v"].values if "v" in edges_m.columns else np.zeros(len(edges_m))

    ratios = []
    for geom in tqdm(edges_m.geometry, desc="      link-node ratio", unit="seg"):
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex.intersection(buf.bounds))
        if not candidates:
            ratios.append(1.0)
            continue
        nearby_geoms = edges_m.geometry.iloc[candidates]
        mask         = nearby_geoms.intersects(buf).values
        idx          = np.array(candidates)[mask]
        if len(idx) == 0:
            ratios.append(1.0)
            continue
        n_edges  = len(idx)
        # Count unique node endpoints using numpy (fast)
        n_nodes  = len(np.unique(np.concatenate([u_arr[idx], v_arr[idx]])))
        ratios.append(n_edges / max(n_nodes, 1))
    return pd.Series(ratios, index=edges_m.index)


def _width_weighted_road_density(edges_m, buffer_m=200):
    """
    Road surface area density: sum(length_i x width_i) / buffer_area.

    Dimensionless ratio [0, ~1] representing what fraction of the buffer
    area is covered by road surface.

    Physical justification (independent of traffic assumptions):
    - Wider roads present larger reflective surfaces -> more sound energy
    - Wider roads carry more traffic lanes -> more source distribution
    - Road surface area directly relates to impervious cover and heat island
      effects that correlate with noise levels

    Width comes from street_width_m already computed in _canyon_ratio,
    so this adds zero extra spatial computation.
    """
    buf_area_m2 = np.pi * buffer_m ** 2
    sindex      = edges_m.sindex

    # Use already-computed street_width_m if available, else estimate from lanes
    if "street_width_m" in edges_m.columns:
        widths = edges_m["street_width_m"].fillna(6.0).values
    else:
        # Fallback: estimate width from lane count (3.5m per lane + 2m margins)
        lanes  = edges_m["lanes"].fillna(1.0).values if "lanes" in edges_m.columns else np.ones(len(edges_m))
        widths = np.clip(lanes * 3.5 + 2.0, 3.0, 50.0)

    lengths = edges_m.geometry.length.values

    densities = []
    for geom in tqdm(edges_m.geometry,
                     desc="      width-weighted road density", unit="seg"):
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex.intersection(buf.bounds))
        if not candidates:
            densities.append(0.0)
            continue
        nearby_geoms = edges_m.geometry.iloc[candidates]
        mask         = nearby_geoms.intersects(buf).values
        idx          = np.array(candidates)[mask]
        if len(idx) == 0:
            densities.append(0.0)
            continue
        road_area = (lengths[idx] * widths[idx]).sum()
        densities.append(road_area / buf_area_m2)
    return pd.Series(densities, index=edges_m.index)


# Power exponent for class_weighted_road_density.
# p=1.0: linear (motorway/residential ratio = 4.5x)
# p=2.0: quadratic (ratio = 20x, close to actual traffic volume ratio ~30x)
# p=2.5: (ratio = 43x, upper bound of acoustic justification)
_CLASS_WEIGHT_POWER = 2.5

def _class_weighted_road_density(edges_m, buffer_m=200):
    """
    Road density weighted by highway class raised to power p.

    weight_i = class_i ^ p

    p=1 (linear): motorway weight 9, residential weight 2, ratio 4.5x
    p=2 (quadratic): motorway weight 81, residential weight 4, ratio 20x

    A motorway carries ~30x more vehicles than a residential street,
    so p=2 better reflects actual traffic volume ratios than linear.
    Amplifying the difference helps the model better predict noise gradients
    between major arterials and quiet residential streets.

    density = sum(length_i x class_i^p) / buffer_area
    """
    buf_area_m2 = np.pi * buffer_m ** 2
    sindex      = edges_m.sindex
    # Exponential weighting: 2^class — each road class doubles the weight
    # class 0 (pedestrian) = 1, residential(2) = 4, primary(6) = 64, motorway(9) = 512
    # Motorway:residential ratio = 128x (vs actual ~30x traffic volume)
    # Capped at 100 to prevent motorways from completely dominating
    hw_raw  = edges_m["highway_class"].values.astype(float)
    hw_vals = 10.0 ** hw_raw   # 10^class, no cap — empirically best
    lengths = edges_m.geometry.length.values

    densities = []
    for geom in tqdm(edges_m.geometry,
                     desc="      class-weighted road density (10^class)", unit="seg"):
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex.intersection(buf.bounds))
        if not candidates:
            densities.append(0.0)
            continue
        nearby_geoms = edges_m.geometry.iloc[candidates]
        mask         = nearby_geoms.intersects(buf).values
        idx          = np.array(candidates)[mask]
        if len(idx) == 0:
            densities.append(0.0)
            continue
        weighted = (hw_vals[idx] * lengths[idx]).sum()
        densities.append(weighted / buf_area_m2)
    return pd.Series(densities, index=edges_m.index)


def _local_dead_end_fraction(edges_m, buffer_m=200):
    """
    Fraction of nodes within buffer that are dead ends (degree=1).

    Local version of dead_end_fraction — varies per segment, not per city.
    High value = cul-de-sac neighbourhood, no through traffic, quiet.
    Low value  = well-connected area, high through-traffic potential, loud.

    Computed using the same u/v arrays as link_node_ratio — minimal extra cost
    since the spatial index query is reused.

    Acoustic interpretation: dead-end streets have no through traffic so
    noise is dominated by local access trips only. A segment with 30% dead-end
    neighbours is in a fundamentally quieter acoustic environment than one
    with 0% dead ends, regardless of which city it's in.
    """
    sindex = edges_m.sindex
    u_arr  = edges_m["u"].values if "u" in edges_m.columns else np.zeros(len(edges_m))
    v_arr  = edges_m["v"].values if "v" in edges_m.columns else np.zeros(len(edges_m))

    fractions = []
    for geom in tqdm(edges_m.geometry,
                     desc="      local dead-end fraction", unit="seg"):
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex.intersection(buf.bounds))
        if not candidates:
            fractions.append(0.0)
            continue
        mask = edges_m.geometry.iloc[candidates].intersects(buf).values
        idx  = np.array(candidates)[mask]
        if len(idx) == 0:
            fractions.append(0.0)
            continue
        # Count all nodes and degree-1 nodes within buffer
        all_nodes  = np.concatenate([u_arr[idx], v_arr[idx]])
        nodes, counts = np.unique(all_nodes, return_counts=True)
        # In the local subgraph, degree-1 nodes appear only once
        n_dead = (counts == 1).sum()
        fractions.append(n_dead / max(len(nodes), 1))
    return pd.Series(fractions, index=edges_m.index)


def _min_road_distance_m(edges_m, buffer_m=300):
    """
    Minimum distance (metres) from each segment to ANY other road segment.

    Models geometric spreading in OpeNoise — a receiver far from all roads
    gets low Lden purely from distance attenuation (divergence loss ~6 dB
    per doubling of distance). This is the primary cause of very low dB
    values in parks, superblocks, and deep residential interiors.

    Low value  = surrounded by roads = high ambient noise floor
    High value = isolated from roads = low Lden from geometric spreading

    Buffer of 300m used (larger than most other features) to capture
    segments that are genuinely isolated from the road network.
    """
    sindex = edges_m.sindex
    min_dists = []
    for i, (idx, row) in enumerate(edges_m.iterrows()):
        geom       = row.geometry
        # Expand search until we find at least one other segment
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex.intersection(buf.bounds))
        # Remove self
        candidates = [c for c in candidates if edges_m.index[c] != idx]
        if not candidates:
            min_dists.append(float(buffer_m))
            continue
        nearby = edges_m.geometry.iloc[candidates]
        dists  = nearby.distance(geom)
        dists  = dists[dists > 0]
        if len(dists) == 0:
            min_dists.append(0.0)
        else:
            min_dists.append(float(dists.min()))
    return pd.Series(min_dists, index=edges_m.index)


def _enclosed_fraction(edges_m, buildings_m, buffer_m=100):
    """
    Fraction of the buffer perimeter enclosed by building footprints.

    Approximates acoustic shielding — a segment enclosed by buildings
    on multiple sides is in an acoustic shadow, receiving attenuated
    noise from distant sources via diffraction over building edges.
    OpeNoise explicitly computes this diffraction path; this feature
    approximates the shielding geometry from OSM building footprints.

    High value (~0.8+) = courtyard / back alley, strong shielding → low Lden
    Low value (~0.1-)  = open street, exposed to distant sources → higher Lden

    Method: intersect building footprints with the buffer boundary ring
    (annulus between buffer_m-5 and buffer_m), compute covered fraction.
    This captures buildings at the perimeter that act as acoustic barriers.
    """
    buf_area   = np.pi * buffer_m ** 2
    sindex_b   = buildings_m.sindex
    fractions  = []

    for geom in tqdm(edges_m.geometry,
                     desc="      enclosed fraction", unit="seg"):
        buf_outer  = geom.buffer(buffer_m)
        buf_inner  = geom.buffer(max(buffer_m - 10, 1))
        ring       = buf_outer.difference(buf_inner)
        ring_area  = ring.area
        if ring_area <= 0:
            fractions.append(0.0)
            continue
        candidates = list(sindex_b.intersection(buf_outer.bounds))
        if not candidates:
            fractions.append(0.0)
            continue
        nearby = buildings_m.geometry.iloc[candidates]
        nearby = nearby[nearby.intersects(ring)]
        if len(nearby) == 0:
            fractions.append(0.0)
            continue
        covered = nearby.intersection(ring).area.sum()
        fractions.append(min(covered / ring_area, 1.0))
    return pd.Series(fractions, index=edges_m.index)


def _road_density(edges_m, buffer_m=200):
    """
    Total road length (metres) per km2 within buffer.
    Higher density = more simultaneous noise sources in the acoustic environment.
    Scale-invariant: expressed per unit area, comparable across cities.
    """
    buf_area_km2 = np.pi * (buffer_m / 1000) ** 2
    sindex       = edges_m.sindex
    densities    = []
    for geom in tqdm(edges_m.geometry, desc="      road density", unit="seg"):
        buf        = geom.buffer(buffer_m)
        candidates = list(sindex.intersection(buf.bounds))
        nearby     = edges_m.iloc[candidates]
        nearby     = nearby[nearby.geometry.intersects(buf)]
        if len(nearby) == 0:
            densities.append(0.0)
            continue
        total_length = nearby.geometry.length.sum()
        densities.append(total_length / buf_area_km2)
    return pd.Series(densities, index=edges_m.index)


def _water_fraction(edges_m, water_m, buffer_m=100):
    """
    Fraction of buffer area covered by water bodies.
    Water surfaces reflect sound and extend noise propagation distances.
    Acoustically opposite to green_fraction — water amplifies, vegetation absorbs.
    Key differentiator: Amsterdam canals, Pittsburgh rivers, Oslo fjord inlet.
    """
    buf_area = np.pi * buffer_m ** 2
    if len(water_m) == 0:
        return pd.Series(0.0, index=edges_m.index)
    sindex    = water_m.sindex
    fractions = []
    for geom in tqdm(edges_m.geometry, desc="      water fraction", unit="seg"):
        buf, nearby = _buf_nearby(sindex, water_m, geom, buffer_m)
        if len(nearby) == 0:
            fractions.append(0.0)
            continue
        fractions.append(min(
            nearby.geometry.intersection(buf).area.sum() / buf_area, 1.0))
    return pd.Series(fractions, index=edges_m.index)


def extract_features(edges, buildings, green, water, G, crs_metric):
    print("[2/5] Extracting morphological features...")

    edges_m     = edges.to_crs(crs_metric).copy()
    buildings_m = buildings.to_crs(crs_metric).copy()
    green_m     = green.to_crs(crs_metric).copy() if len(green) > 0 else green
    water_m     = water.to_crs(crs_metric).copy() if len(water) > 0 else water

    # ── Road attributes ───────────────────────────────────────────────────────
    # highway_class — road type ordinal (0-9), single strongest local predictor
    # Directly encodes traffic volume hierarchy: motorway(9) >> residential(2)
    edges_m["highway_class"] = edges_m["highway"].apply(_hw_class)
    edges_m["lanes"]     = edges_m["lanes"].apply(_lanes) if "lanes" in edges_m.columns else 1.0
    edges_m["maxspeed"]  = edges_m["maxspeed"].apply(_speed) if "maxspeed" in edges_m.columns else 30.0
    if "oneway" in edges_m.columns:
        edges_m["oneway"] = edges_m["oneway"].fillna(False).astype(int)
    else:
        edges_m["oneway"] = 0
    # Segment length in metres — proxy for road source integration length in CNOSSOS
    edges_m["seg_length_m"] = edges_m.geometry.length

    # Global link-node ratio — city-level network topology fingerprint
    # Computed once for entire network, then broadcast to all segments.
    # Captures overall connectivity type: organic/tree-like vs grid vs dense.
    # This is the "domain feature" that helps the model know what city it's in.
    import networkx as _nx
    G_undir = G.to_undirected()
    n_edges_global = G_undir.number_of_edges()
    n_nodes_global = G_undir.number_of_nodes()
    lnr_global     = n_edges_global / max(n_nodes_global, 1)
    edges_m["link_node_ratio_global"] = lnr_global
    print("      Global link-node ratio: {:.3f} ({} edges, {} nodes)".format(
        lnr_global, n_edges_global, n_nodes_global))

    # Note: dead_end_fraction is computed locally per segment buffer (below)
    # not globally — local dead-end proximity better captures quietness potential

    # Average block length — mean edge length in metres for entire network
    # Short blocks = fine-grained grid, more stop-start traffic, more noise
    # Long blocks = sparse network, faster through-traffic
    _edge_lengths = [d.get("length", 0) for _, _, d in G.edges(data=True)
                     if d.get("length", 0) > 0]
    avg_block_length = float(np.mean(_edge_lengths)) if _edge_lengths else 100.0
    edges_m["avg_block_length_m"] = avg_block_length
    print("      Avg block length: {:.1f} m".format(avg_block_length))

    # City-level class-weighted road density (global acoustic baseline)
    # Encodes the overall acoustic intensity of the entire city road network.
    # Uses the same 10^class weighting as the local feature but computed
    # over ALL edges in the graph — not normalised by buffer area but by
    # total city area (pi * DOWNLOAD_RADIUS_M^2).
    # High value = dense network with many high-class roads (Bangkok, NYC)
    # Low value  = sparse or low-class network (Oslo residential core)
    # This is the city-level signal that tells the unified model WHERE
    # the city's absolute noise distribution sits — the missing cross-city
    # domain feature.
    _city_area_km2   = 3.14159 * (DOWNLOAD_RADIUS_M / 1000) ** 2
    _hw_classes      = np.array([
        HIGHWAY_ORDER.get(str(_scalar(d.get("highway"))), 2)
        for _, _, d in G.edges(data=True)
    ], dtype=float)
    _edge_lengths_g  = np.array([
        d.get("length", 0) for _, _, d in G.edges(data=True)
    ], dtype=float)
    _city_cwd        = float(
        np.sum(10.0 ** _hw_classes * _edge_lengths_g) /
        (_city_area_km2 * 1e6)   # convert km2 to m2 for consistency
    )
    edges_m["city_class_weighted_density_global"] = _city_cwd
    print("      City class-weighted density: {:.4f}".format(_city_cwd))

    # Average node degree — mean connections per intersection
    # High = grid-like (Manhattan ~3.8), Low = tree-like (suburban ~2.5)
    # Different from link_node_ratio: captures intersection complexity
    degrees   = [d for _, d in G_undir.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 3.0
    edges_m["avg_node_degree_global"] = avg_degree
    print("      Avg node degree: {:.3f}".format(avg_degree))

    # Street orientation entropy — bearing distribution entropy (bits)
    # Low entropy = strict grid (NYC ~1.2 bits)
    # High entropy = organic network (Bangkok ~3.5 bits)
    # Captures planarity/regularity of street layout — not in any other feature
    try:
        import math as _math
        bearings = []
        for u, v, data in G.edges(data=True):
            if u in G.nodes and v in G.nodes:
                try:
                    u_data = G.nodes[u]; v_data = G.nodes[v]
                    dx = v_data.get("x", 0) - u_data.get("x", 0)
                    dy = v_data.get("y", 0) - u_data.get("y", 0)
                    bearing = (_math.degrees(_math.atan2(dx, dy)) + 360) % 360
                    bearings.append(bearing)
                except Exception:
                    pass
        if bearings:
            # Bin into 36 x 10-degree bins and compute Shannon entropy
            bins   = [0] * 36
            for b in bearings:
                bins[int(b // 10) % 36] += 1
            total  = sum(bins)
            probs  = [c / total for c in bins if c > 0]
            entropy = -sum(p * _math.log2(p) for p in probs)
        else:
            entropy = 3.0  # neutral fallback
    except Exception:
        entropy = 3.0
    edges_m["street_orientation_entropy_global"] = float(entropy)
    print("      Street orientation entropy: {:.3f} bits".format(entropy))

    # Global circuity — average ratio of network distance to Euclidean distance.
    # Captures how direct/winding the overall street network is.
    # Pure grid (NYC) ~1.02-1.05 | Organic network (Bangkok) ~1.15-1.25
    # OSMnx computes this in one call from the graph.
    try:
        # Project graph first — ox.stats.circuity_avg needs metric coords
        # (node x/y are WGS84 degrees but edge length is metres — must match)
        G_proj   = ox.project_graph(G)
        circuity = ox.stats.circuity_avg(G_proj)
        if not (1.0 <= float(circuity) <= 3.0):
            raise ValueError("out of range: {}".format(circuity))
    except Exception:
        try:
            import math as _math
            _nodes_t, _edges_t = ox.graph_to_gdfs(G)
            _edges_t = _edges_t.reset_index()
            def _hav(row):
                try:
                    u = _nodes_t.loc[row["u"]].geometry
                    v = _nodes_t.loc[row["v"]].geometry
                    la1,lo1 = _math.radians(u.y), _math.radians(u.x)
                    la2,lo2 = _math.radians(v.y), _math.radians(v.x)
                    a = (_math.sin((la2-la1)/2)**2 +
                         _math.cos(la1)*_math.cos(la2)*_math.sin((lo2-lo1)/2)**2)
                    return 6371000 * 2 * _math.asin(_math.sqrt(max(0, a)))
                except Exception:
                    return np.nan
            _ed   = _edges_t.apply(_hav, axis=1)
            _pl   = (_edges_t["length"].values if "length" in _edges_t.columns
                     else _edges_t.geometry.length.values)
            mask  = (_ed > 1.0) & _ed.notna()
            circuity = float((_pl[mask] / _ed[mask]).mean()) if mask.sum() > 0 else 1.1
            if not (1.0 <= circuity <= 3.0):
                circuity = 1.1
        except Exception:
            circuity = 1.1
    edges_m["circuity_global"] = float(circuity)
    print("      Global circuity: {:.4f} (1.0=perfect grid, >1=winding)".format(circuity))

    # Road slope — absolute grade (rise/run), range 0.0 (flat) to 0.3+ (steep)
    # CNOSSOS applies gradient correction: uphill grades increase engine noise
    # Pittsburgh has grades up to 20%+, Bangkok is nearly flat — good discriminator
    if "grade_abs" in edges_m.columns:
        edges_m["abs_grade"] = pd.to_numeric(edges_m["grade_abs"], errors="coerce").fillna(0.0)
        print("      Road slope: mean={:.3f}, max={:.3f}".format(
            edges_m["abs_grade"].mean(), edges_m["abs_grade"].max()))
    else:
        edges_m["abs_grade"] = 0.0
        print("      Road slope: not available (elevation API may have been unavailable)")

    # ── Network / traffic features (scale-invariant) ──────────────────────────
    (edges_m["major_road_fraction"],
     edges_m["medium_road_fraction"],
     edges_m["local_road_fraction"],
     edges_m["pedestrian_fraction"])  = _road_class_fractions(edges_m)
    edges_m["intersection_density"]   = _intersection_density(edges_m, G)
    edges_m["link_node_ratio"]        = _link_node_ratio(edges_m)
    edges_m["dead_end_fraction"]      = _local_dead_end_fraction(edges_m)
    edges_m["min_road_distance_m"]    = _min_road_distance_m(edges_m)
    edges_m["road_density"]           = _road_density(edges_m)
    # width_weighted_road_density uses street_width_m which must be computed first
    edges_m["width_weighted_road_density"]   = _width_weighted_road_density(edges_m)
    edges_m["class_weighted_road_density"] = _class_weighted_road_density(edges_m)

    # ── Geometry / acoustic features ─────────────────────────────────────────
    (edges_m["canyon_ratio"],
     edges_m["street_width_m"],
     edges_m["building_height_std"]) = _canyon_ratio(edges_m, buildings_m)
    edges_m["enclosed_fraction"]     = _enclosed_fraction(edges_m, buildings_m)
    edges_m["building_density"]      = _building_density(edges_m, buildings_m)
    edges_m["green_fraction"]   = _green_fraction(edges_m, green_m)
    edges_m["water_fraction"]   = _water_fraction(edges_m, water_m)

    # Runtime feature list (strips comment lines from _FEATURE_COLS_RUNTIME)
    feat_cols = _FEATURE_COLS_RUNTIME
    print("      Features ready: {}".format(feat_cols))
    return edges_m, feat_cols


# =============================================================================
# STEPS 3 & 4: Load noise contours and spatial join
# =============================================================================
def load_and_join_noise(edges_m, noise_file, db_col, buffer_m, crs_metric):
    print("[3/5] Loading OpeNoise output: {}".format(noise_file))
    noise = gpd.read_file(noise_file)

    def to_midpoint(v):
        v = str(v).strip()
        if "-" in v:
            a, b = v.split("-")
            return (float(a) + float(b)) / 2
        elif ">" in v:
            return float(v.replace(">", "").strip()) + 2.5
        try:
            return float(v)
        except Exception:
            return np.nan

    noise["db_mid"] = noise[db_col].apply(to_midpoint)
    noise = noise[["geometry", "db_mid"]].dropna()
    print("      {} noise polygons | dB range: {:.0f}-{:.0f}".format(
        len(noise), noise["db_mid"].min(), noise["db_mid"].max()))

    edges_m = edges_m.to_crs(crs_metric)
    noise   = noise.to_crs(crs_metric)
    print("      CRS aligned to {}".format(crs_metric))
    print("      Segment bounds: {}".format(edges_m.total_bounds.round(0)))
    print("      Noise bounds:   {}".format(noise.total_bounds.round(0)))

    print("[4/5] Spatially joining noise -> segments...")
    print("      buffering and joining via sjoin...")

    edges_m   = edges_m.reset_index(drop=True)
    edges_buf = gpd.GeoDataFrame(
        {"pos": np.arange(len(edges_m)),
         "geometry": edges_m.geometry.buffer(buffer_m)},
        crs=crs_metric
    )

    joined = gpd.sjoin(
        edges_buf, noise[["geometry", "db_mid"]],
        how="left", predicate="intersects"
    )
    print("      sjoin complete: {} rows, {} unique segments hit".format(
        len(joined), joined["pos"].nunique()))

    joined = joined.dropna(subset=["db_mid"])
    if len(joined) == 0:
        print("WARNING: 0 matches -- layers do not overlap.")
        edges_m["noise_db"] = np.nan
        return edges_m

    noise_dict      = joined.groupby("pos")["db_mid"].mean().to_dict()
    edges_m["noise_db"] = edges_m.index.map(noise_dict)

    n         = edges_m["noise_db"].notna().sum()
    n_total   = len(edges_m)
    match_pct = 100.0 * n / max(n_total, 1)
    print("      {}/{} segments matched ({:.0f}%).".format(n, n_total, match_pct))

    if match_pct < 60:
        print()
        print("  !! WARNING: Match rate {:.0f}% is below 60% !!".format(match_pct))
        print("  Likely causes:")
        print("    1. CENTRE MISMATCH — OpeNoise was run from a different centre")
        print("       than the OSM download. Check that prepare_openoise_clipped.py")
        print("       and noise_proxy_pipeline_v5.py used the same coordinates.")
        print("    2. PARTIAL OPENOISE RUN — the noise contour file covers only part")
        print("       of the study area. Re-run OpeNoise with the full grid file.")
        print("    3. CRS MISMATCH — noise contours and segments are in different")
        print("       coordinate systems. Check EPSG codes match.")
        print("  Recommended fix:")
        print("    1. del features_cache_{}.gpkg".format(
              "CITY (check NOISE_FILE setting)"))
        print("    2. Re-run prepare_openoise_clipped.py for this city")
        print("    3. Re-run OpeNoise in QGIS")
        print("    4. Re-run this pipeline")
        print()
    elif match_pct < 80:
        print("  Note: Match rate {:.0f}% — some segments may be outside".format(
              match_pct))
        print("        OpeNoise tile boundary (circle vs square extent). Normal.")

    return edges_m


# =============================================================================
# PUBLIC API
# =============================================================================
# Alias for external import — use FEATURE_COLS in all scripts
_FEATURE_COLS_RUNTIME = [
    "highway_class",
    "city_class_weighted_density_global",
    "dead_end_fraction",
    "avg_node_degree_global",
    "street_orientation_entropy_global",
    "major_road_fraction",
    "medium_road_fraction",
    "local_road_fraction",
    "pedestrian_fraction",
    "avg_block_length_m",
    "intersection_density",
    "link_node_ratio",
    "road_density",
    "width_weighted_road_density",
    "class_weighted_road_density",
    "min_road_distance_m",
    "enclosed_fraction",
    "building_density",
    "green_fraction",
    "water_fraction",
]

FEATURE_COLS = _FEATURE_COLS_RUNTIME