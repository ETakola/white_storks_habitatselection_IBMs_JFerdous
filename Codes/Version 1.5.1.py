# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 12:01:09 2025

@author: jannatul
-fixed day time movement only (5am to 8pm)
-site fidelity: every year and every day same starting point
-nest fidelity: breeding individuals return to the nest every 2 hours
"""


from __future__ import annotations
import os, math, random, warnings
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
from scipy.special import expit
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================
# ===== Default File Paths (Edit as needed) ===========
# =====================================================
LULC_PATH = "D:/4th Semester/thesis/models/GeoTiffs/New/LC_2012_07.tif"
NDVI_PATH = "D:/4th Semester/thesis/models/GeoTiffs/NDVI_NDWI/NDVI_2012_JulAug_30m.tif"
NDWI_PATH = "D:/4th Semester/thesis/models/GeoTiffs/NDVI_NDWI/MNDWI_2012_JulAug_30m.tif"

# Fragmentation layers (optional — leave {} if not available)
FRAG_PATHS = {}  # e.g., {"ED":"ed.tif","PD":"pd.tif","SHEI":"shei.tif","SHDI":"shdi.tif","ENT":"ent.tif"}

POINTS_PATH = "D:/4th Semester/thesis/models/RSF/MPIAB_inside_data_xl.xlsx"
RANDOM_RSF_PATH = "D:/4th Semester/thesis/models/RSF/rsf_random_effects_by_bird.csv"
FIXED_RSF_PATH  = "D:/4th Semester/thesis/models/RSF/rsf_fixed_effects.csv"
OUTPUT_DIR = "D:/4th Semester/thesis/models/IBM/Output"

# =====================================================
# =============== Model Constants =====================
# =====================================================
STEP_MINUTES_DEFAULT = 30
FOOD_RENEW_DAYS = 2
BREED_WINDOW_DAYS = 15

ENERGY_REQUIREMENT = {"physio": 0.75, "foraging": 0.10, "flying": 0.10, "other": 0.05}
REQUIREMENT_TOTAL = float(sum(list(ENERGY_REQUIREMENT.values())))

BREED_MIN_EXCESS = 0.30
DEATH_THRESHOLD = 0.60

DEFAULT_LULC_ENERGY = {
    10: 3.0, 11: 3.0, 20: 2.5, 51: 1.0, 61: 1.5, 62: 0.5, 71: 0.5,
    72: 0.1, 81: 0.1, 82: 0.1, 130: 3.0, 182: 4.0, 190: 1.5, 210: 0.1
}

ENERGY_SCALARS = {
    "ndvi_gain_scale": 0.3,
    "ndwi_gain_scale": 0.6,
    "fragmentation_gain_scale": 0.7,
    "distance_cost_scale": 0.0002,
    "baseline_daily_units": 10.0,
}

# Tiled processing size to avoid MemoryError
TILE = 2048  # reduce if you still hit memory limits

# ===== Activity & nest behavior =====
DAY_START_HOUR = 5          # start of active day
DAY_END_HOUR   = 20         # end of active day

HOME_DECAY = 0.0005         # site fidelity strength (higher → tighter home range)
NEST_PULL_INTERVAL_MIN = 120  # every 2 hours
NEST_PULL_ALPHA = 0.003       # strength of nest-pull reward
NEST_PULL_ACTIVE_ONLY_IF_BRED = True  # True = only after bird is marked 'bred'

BEHAVIOR_PARAMS = {
    "breeding": {
        "step_mu_m": 350.0,     # typical step length during breeding
        "step_sigma_m": 180.0,  # step variability
        "turn_kappa": 3.0,      # higher = straighter persistence
        "nest_pull_alpha": 0.0008,  # smoother home attraction strength
        "max_foraging_radius_m": 28100.0  # ref: https://pmc.ncbi.nlm.nih.gov/articles/PMC4791752/#:~:text=Storks%20travelled%20up%20to%2048.2,indicating%20higher%20reliance%20on%20landfill.
    },
    "nonbreeding": {
        "step_mu_m": 2000.0,
        "step_sigma_m": 800.0,
        "turn_kappa": 1.0,
        "nest_pull_alpha": 0.0,        # no nest pull if not breeding
        "max_foraging_radius_m": 48200.0
    }
}

# ======= Helpers ===========

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def softmax(x, temperature=1.0):
    z = (x - np.nanmax(x)) / max(temperature, 1e-6)
    e = np.exp(z)
    return e / np.sum(e)

def within_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w

# --- Movement helpers ---
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))

def bearing_deg(lon1, lat1, lon2, lat2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brng

def von_mises_weight(prev_bearing_deg, new_bearing_deg, kappa):
    # proportional weight in (0,1]; add tiny epsilon to avoid log(0)
    th1 = math.radians(prev_bearing_deg)
    th2 = math.radians(new_bearing_deg)
    # np.i0 is the modified Bessel function of order 0
    return max(1e-12, math.exp(kappa * math.cos(th2 - th1)) / (2 * math.pi * np.i0(kappa)))

def move_along_bearing(lon, lat, dist_m, bearing_deg):
    """Great-circle move from (lon,lat) by dist_m at bearing_deg. Returns (lon2, lat2)."""
    R = 6371000.0
    d = dist_m / R
    brg = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brg))
    lon2 = lon1 + math.atan2(math.sin(brg) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lon2), math.degrees(lat2))


# ========= Environment ===========

@dataclass
class Environment:
    lulc_path: str
    ndvi_path: str
    ndwi_path: str
    frag_paths: Dict[str, str]
    lulc: xr.DataArray = field(init=False)
    ndvi: xr.DataArray = field(init=False)
    ndwi: xr.DataArray = field(init=False)
    frag: Dict[str, xr.DataArray] = field(init=False, default_factory=dict)
    transform: rasterio.Affine = field(init=False)
    crs: str = field(init=False)
    food_base: np.ndarray = field(init=False)
    food_stock: np.ndarray = field(init=False)
    last_renewal: datetime = field(init=False)

    def __post_init__(self):
        # load rasters 
        if os.path.exists("./data/nests_auto.csv"):
            try:
                nests = pd.read_csv("./data/nests_auto.csv")
                minx, miny, maxx, maxy = (
                    nests["lon"].astype(float).min(),
                    nests["lat"].astype(float).min(),
                    nests["lon"].astype(float).max(),
                    nests["lat"].astype(float).max(),
                )
                buffer = 0.5
                minx -= buffer; miny -= buffer; maxx += buffer; maxy += buffer
                with rasterio.open(self.lulc_path) as src:
                    window = src.window(minx, miny, maxx, maxy)
                    self.lulc = rxr.open_rasterio(self.lulc_path, masked=True, window=window).squeeze()
            except Exception as e:
                print(f"⚠️ Cropping failed, loading full raster: {e}")
                self.lulc = rxr.open_rasterio(self.lulc_path).squeeze()
        else:
            self.lulc = rxr.open_rasterio(self.lulc_path).squeeze()

        self.ndvi = rxr.open_rasterio(self.ndvi_path).squeeze().rio.reproject_match(self.lulc)
        self.ndwi = rxr.open_rasterio(self.ndwi_path).squeeze().rio.reproject_match(self.lulc)

        for k, p in self.frag_paths.items():
            self.frag[k.upper()] = rxr.open_rasterio(p).squeeze().rio.reproject_match(self.lulc)

        with rasterio.open(self.lulc_path) as src:
            self.transform = src.transform
            self.crs = src.crs.to_string() if src.crs else ""

        self._init_food_tiled()

    #  Tiled min/max helper for frag normalization 
    def _min_max_tiled(self, arr: xr.DataArray) -> Tuple[float, float]:
        a = arr.values
        h, w = a.shape
        mn = np.inf
        mx = -np.inf
        for r0 in range(0, h, TILE):
            r1 = min(h, r0 + TILE)
            for c0 in range(0, w, TILE):
                c1 = min(w, c0 + TILE)
                block = a[r0:r1, c0:c1]
                bmn = np.nanmin(block)
                bmx = np.nanmax(block)
                if bmn < mn: mn = bmn
                if bmx > mx: mx = bmx
        if not np.isfinite(mn): mn = 0.0
        if not np.isfinite(mx): mx = 1.0
        if mx - mn < 1e-9:
            mx = mn + 1e-9
        return float(mn), float(mx)

    # Tiled food init 
    def _init_food_tiled(self):
        lulc = self.lulc.values.astype(np.int32, copy=False)
        h, w = lulc.shape
        base = np.zeros((h, w), dtype=np.float32)

        # set base by LULC
        for cls, e in DEFAULT_LULC_ENERGY.items():
            mask = (lulc == cls)
            base[mask] = float(e)

        # NDVI + NDWI in tiles
        ndvi = self.ndvi.values  
        ndwi = self.ndwi.values
        for r0 in range(0, h, TILE):
            r1 = min(h, r0 + TILE)
            for c0 in range(0, w, TILE):
                c1 = min(w, c0 + TILE)

                ndvi_block = ndvi[r0:r1, c0:c1].astype(np.float32, copy=False)
                np.clip(ndvi_block, 0.0, 1.0, out=ndvi_block)
                base[r0:r1, c0:c1] += ENERGY_SCALARS["ndvi_gain_scale"] * ndvi_block

                ndwi_block = ndwi[r0:r1, c0:c1].astype(np.float32, copy=False)
                np.clip(ndwi_block, 0.0, 1.0, out=ndwi_block)
                base[r0:r1, c0:c1] += ENERGY_SCALARS["ndwi_gain_scale"] * ndwi_block

        # Fragmentation composite 
        if self.frag:
            # precompute min/max per layer
            stats = {}
            for name, arr in self.frag.items():
                stats[name] = self._min_max_tiled(arr)

            for r0 in range(0, h, TILE):
                r1 = min(h, r0 + TILE)
                for c0 in range(0, w, TILE):
                    c1 = min(w, c0 + TILE)
                    comps = []
                    for name, arr in self.frag.items():
                        mn, mx = stats[name]
                        block = arr.values[r0:r1, c0:c1].astype(np.float32, copy=False)
                        block = (block - mn) / (mx - mn)
                        comps.append(block)
                    if comps:
                        frag_idx = np.nanmean(np.stack(comps, axis=0), axis=0).astype(np.float32)
                        base[r0:r1, c0:c1] += ENERGY_SCALARS["fragmentation_gain_scale"] * frag_idx

        self.food_base = base
        self.food_stock = base.copy()
        self.last_renewal = None

    def renew_food_if_due(self, now):
        if self.last_renewal is None:
            self.last_renewal = now
            return
        if (now - self.last_renewal).days >= FOOD_RENEW_DAYS:
            self.food_stock = self.food_base.copy()
            self.last_renewal = now

    def sample_cell(self, lon, lat):
        r, c = rowcol(self.transform, lon, lat)
        r, c = int(r), int(c)
        if not within_bounds(self.lulc.shape, r, c):
            return {}
        out = {
            "LULC": float(self.lulc.values[r, c]),
            "NDVI": float(np.clip(self.ndvi.values[r, c], 0, 1)),
            "NDWI": float(np.clip(self.ndwi.values[r, c], 0, 1)),
            "FOOD": float(self.food_stock[r, c]),
        }
        for k, arr in self.frag.items():
            out[k] = float(arr.values[r, c])
        out["row"], out["col"] = r, c
        return out

    def neighbors_within(self, lon, lat, radius_cells=2):
        r0, c0 = rowcol(self.transform, lon, lat)
        r0, c0 = int(r0), int(c0)
        out = []
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                rr, cc = r0 + dr, c0 + dc
                if within_bounds(self.lulc.shape, rr, cc):
                    x, y = xy(self.transform, rr, cc)
                    out.append((rr, cc, x, y))
        return out

    def consume_food(self, r, c, amount):
        if within_bounds(self.food_stock.shape, r, c):
            self.food_stock[r, c] = max(0.0, self.food_stock[r, c] - amount)


# ======RSF Model (fixed + random effects) =========

@dataclass
class RSFModel:
    """Per-bird RSF coefficients (already combined fixed + random)."""
    coeffs: Dict[str, float]

    def _get(self, name: str) -> float:
        aliases = [name, name.upper(), name.lower(), name.capitalize()]
        if name == "(Intercept)":
            aliases += ["Intercept", "(intercept)", "intercept"]
        for a in aliases:
            if a in self.coeffs:
                return self.coeffs[a]
        return 0.0

    def linear_predictor(self, cov: Dict[str, float]) -> float:
        lp = 0.0
        lp += self._get("(Intercept)")

        ndwi = cov.get("NDWI")
        if ndwi is not None and np.isfinite(ndwi):
            lp += self._get("NDWI") * ndwi

        lulc = cov.get("LULC")
        if lulc is not None and np.isfinite(lulc):
            try:
                code = int(lulc)
                for key in (f"LULC_class{code}", f"LULC_{code}"):
                    if key in self.coeffs:
                        lp += self.coeffs[key]
                        break
            except Exception:
                pass

        cov_upper = {k.upper(): v for k, v in cov.items()}
        for k, beta in self.coeffs.items():
            ku = k.upper()
            if ku in {"(INTERCEPT)", "INTERCEPT", "NDWI"} or ku.startswith("LULC_") or ku.startswith("LULC_CLASS"):
                continue
            if ku in cov_upper and np.isfinite(cov_upper[ku]):
                lp += beta * cov_upper[ku]

        return float(lp)


# ====== RSF Helpers =====

def _read_fixed_effects(path: str) -> Dict[str, float]:
    fx = pd.read_csv(path)
    fx.columns = [c.strip().lower() for c in fx.columns]
    if not {"term", "coef"}.issubset(fx.columns):
        raise ValueError("Fixed-effects file must have columns: term, coef")
    fx = fx.dropna(subset=["term", "coef"])
    out = {}
    for _, r in fx.iterrows():
        term = str(r["term"]).strip()
        try:
            val = float(r["coef"])
        except Exception:
            continue
        out[term] = val
    if "intercept" in out and "(Intercept)" not in out:
        out["(Intercept)"] = out.pop("intercept")
    return out

def _wide_or_pivot_randoms(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = [c.lower() for c in df.columns]
    if {"bird_id", "term", "coef"}.issubset(cols_lower):
        bid = next(c for c in df.columns if c.lower() == "bird_id")
        term = next(c for c in df.columns if c.lower() == "term")
        coef = next(c for c in df.columns if c.lower() == "coef")
        wide = df.pivot_table(index=bid, columns=term, values=coef, aggfunc="first").reset_index()
        wide.columns.name = None
        return wide.set_index(bid)
    if "bird_id" in cols_lower:
        bid = next(c for c in df.columns if c.lower() == "bird_id")
        return df.set_index(bid)
    raise ValueError("Random-effects file must include a 'bird_id' column or long format (bird_id, term, coef).")

def _combine_fixed_random_for_bird(fixed: Dict[str, float], rrow: pd.Series) -> Dict[str, float]:
    cols_lower = {str(c).lower() for c in rrow.index}
    has_full_intercept = any(c in cols_lower for c in ["(intercept)", "intercept"])
    has_full_ndwi = "ndwi" in cols_lower
    has_any_lulc = any(str(k).lower().startswith("lulc") for k in rrow.index)

    if has_full_intercept or has_full_ndwi or has_any_lulc:
        combined = {}
        for k, v in rrow.items():
            if pd.isna(v): continue
            if isinstance(v, (int, float, np.floating)):
                combined[str(k)] = float(v)
        if "Intercept" in combined and "(Intercept)" not in combined:
            combined["(Intercept)"] = combined.pop("Intercept")
        # carry over fixed for missing terms
        for fk, fv in fixed.items():
            if fk not in combined:
                combined[fk] = fv
        return combined

    combined = dict(fixed)
    for k, v in rrow.items():
        if pd.isna(v): continue
        k_low = str(k).lower()
        if not k_low.startswith("re_"):
            continue
        base = k_low[3:]
        if base in {"intercept", "(intercept)"}:
            key = "(Intercept)"
        else:
            key = None
            for fk in fixed.keys():
                if fk.lower() == base:
                    key = fk
                    break
            if key is None and base.startswith("lulc"):
                key = base
        try:
            delta = float(v)
        except Exception:
            continue
        if key is None:
            continue
        combined[key] = combined.get(key, 0.0) + delta

    if "Intercept" in combined and "(Intercept)" not in combined:
        combined["(Intercept)"] = combined.pop("Intercept")
    return combined


# ======= Load Birds (fixed + random effects) =========

def load_birds_from_points(points_file: str,
                           fixed_effects_csv: str,
                           random_effects_csv: str,
                           step_minutes: int):
    # --- nests ---
    ext = os.path.splitext(points_file)[1].lower()
    df = pd.read_excel(points_file, dtype=str) if ext in ['.xls', '.xlsx'] else pd.read_csv(points_file, dtype=str)
    df.columns = [c.strip().lower().replace('.', '_').replace(' ', '_') for c in df.columns]

    if {'individual_local_identifier', 'location_long', 'location_lat', 'timestamp'}.issubset(df.columns):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(['individual_local_identifier', 'timestamp'])
        nests = df.groupby('individual_local_identifier').first().reset_index()
        nests = nests.rename(columns={'individual_local_identifier': 'bird_id',
                                      'location_long': 'lon',
                                      'location_lat': 'lat'})
        ensure_dir('./data')
        nests[['bird_id', 'lon', 'lat']].to_csv('./data/nests_auto.csv', index=False)
        print(f"Auto-created ./data/nests_auto.csv with {len(nests)} birds:")
    elif {'bird_id', 'lon', 'lat'}.issubset(df.columns):
        nests = df.copy()
        print(f"Using provided nests CSV with {len(nests)} birds.")
    else:
        raise ValueError("Input must contain either columns.")

    nests['bird_id'] = (nests['bird_id'].astype(str).str.strip()
                        .str.replace(r'\.0$', '', regex=True)
                        .str.upper().str.replace(r'\s+', '', regex=True))

    # --- fixed (global) ---
    fixed = _read_fixed_effects(fixed_effects_csv)

    # --- randoms / per-bird table ---
    rnd = pd.read_csv(random_effects_csv)
    rnd.columns = [c.strip() for c in rnd.columns]
    bid_col = next((c for c in rnd.columns if c.lower() == "bird_id"), None)
    if bid_col is None and {"bird_id", "term", "coef"}.issubset({c.lower() for c in rnd.columns}):
        bid_col = next(c for c in rnd.columns if c.lower() == "bird_id")
    if bid_col is None:
        raise ValueError("Random-effects file must include a 'bird_id' column")
    rnd[bid_col] = (rnd[bid_col].astype(str).str.strip()
                    .str.replace(r'\.0$', '', regex=True).str.upper()
                    .str.replace(r'\s+', '', regex=True))
    rnd_wide = _wide_or_pivot_randoms(rnd)  # index=bird_id

    print("\n Matching IDs:")
    print("  Nests IDs:", nests['bird_id'].tolist()[:5])
    print("  RSF IDs:", list(rnd_wide.index[:5]))
    print(f"  Total overlap: {len(set(nests['bird_id']) & set(rnd_wide.index))} birds")

    birds = []
    for _, row in nests.iterrows():
        bid = str(row['bird_id']).strip().upper()
        if bid not in rnd_wide.index:
            print(f" Skipping bird {bid} — no RSF row found.")
            continue

        rrow = rnd_wide.loc[bid]
        combined = _combine_fixed_random_for_bird(fixed, rrow)
        rsf_model = RSFModel(coeffs=combined)

        b = BirdAgent(
            bird_id=bid,
            rsf=rsf_model,
            start_lon=float(row['lon']),
            start_lat=float(row['lat']),
            step_minutes=step_minutes,
            step_mu_m=float(row.get('step_mu_m', 300.0)),
            step_sigma_m=float(row.get('step_sigma_m', 150.0)),
            turn_kappa=float(row.get('turn_kappa', 2.0)),
        )
        birds.append(b)

    print(f" Loaded {len(birds)} birds for simulation.")
    return birds

# ======= Bird Agent =========

@dataclass
class BirdAgent:
    bird_id: str
    rsf: RSFModel
    start_lon: float
    start_lat: float
    step_minutes: int = STEP_MINUTES_DEFAULT
    alive: bool = True

    lon: float = field(init=False)
    lat: float = field(init=False)
    bearing_deg: float = field(default_factory=lambda: random.uniform(0, 360))

    energy_cum: float = 0.0
    day_energy_gain: float = 0.0
    day_energy_cost: float = 0.0
    bred: bool = False
    step_count: int = 0

    step_mu_m: float = 300.0
    step_sigma_m: float = 150.0
    turn_kappa: float = 2.0
    daily_net_history: List[float] = field(default_factory=list)


    def __post_init__(self):
        self.lon = self.start_lon
        self.lat = self.start_lat

    def reset_to_nest(self):
        """Snap bird back to its nest (for daily/yearly site fidelity)."""
        self.lon = self.start_lon
        self.lat = self.start_lat
        # Optionally reset bearing:
        # self.bearing_deg = random.uniform(0, 360)

    def step(self, env: Environment, now: datetime) -> Dict[str, float]:
        """
        step logics:
          - Sample a real step length (m) and turning deviation (von Mises).
          - Compute new (lon,lat) by great-circle move.
          - Apply RSF + site fidelity +  smooth nest pull every 2h in daytime.
        """
        # ---- behavioural mode ----
        mode = "breeding" if self.bred else "nonbreeding"
        p = BEHAVIOR_PARAMS[mode]
    
        # If user provided per-bird overrides, keep them as fallbacks
        mu = float(getattr(self, "step_mu_m", p["step_mu_m"])) or p["step_mu_m"]
        sg = float(getattr(self, "step_sigma_m", p["step_sigma_m"])) or p["step_sigma_m"]
        kappa = float(getattr(self, "turn_kappa", p["turn_kappa"])) or p["turn_kappa"]
    
        # ---- sample distance and bearing ----
        dist_m = max(0.0, np.random.normal(mu, sg))
        turn_rad = np.random.vonmises(0.0, kappa)  # deviation from current bearing
        new_bearing = (self.bearing_deg + math.degrees(turn_rad)) % 360.0
    
        # ---- propose new coordinate ----
        new_lon, new_lat = move_along_bearing(self.lon, self.lat, dist_m, new_bearing)
    
        # ---- pull back softly if foraging too far from nest (soft arena, not a hard cap) ----
        nest_dist_after = haversine_m(self.start_lon, self.start_lat, new_lon, new_lat)
        arena = p["max_foraging_radius_m"]
        if arena > 0:
            # exponential penalty beyond the arena radius
            if nest_dist_after > arena:
                over = nest_dist_after - arena
                penalty = math.exp(-0.001 * over)  # gentle decay; tweak if needed
                # pull slightly toward nest by shortening the step
                adj_dist = dist_m * penalty
                new_lon, new_lat = move_along_bearing(self.lon, self.lat, adj_dist, new_bearing)
                nest_dist_after = haversine_m(self.start_lon, self.start_lat, new_lon, new_lat)
    
        # ---- read environment at proposed location ----
        cov = env.sample_cell(new_lon, new_lat)

        # --- Handle out-of-bounds movement by bouncing back ---
        if not cov:
            # Try turning 180° (reverse) plus a small random deviation
            new_bearing = (self.bearing_deg + 180 + np.random.uniform(-30, 30)) % 360
        
            #  3 reflection attempts before giving up
            for _ in range(3):
                new_lon, new_lat = move_along_bearing(self.lon, self.lat, dist_m, new_bearing)
                cov = env.sample_cell(new_lon, new_lat)
                if cov:
                    break
                # if still out of bounds, pick a new random bearing
                new_bearing = np.random.uniform(0, 360)
        
            # if still invalid after retries, stay in place
            if not cov:
                new_lon, new_lat = self.lon, self.lat
                cov = env.sample_cell(new_lon, new_lat)
                if not cov:
                    return {}

    
        #  RSF linear predictor 
        lp = self.rsf.linear_predictor(cov)
    
        # site fidelity 
        if self.bred:
            home_weight = math.exp(-HOME_DECAY * nest_dist_after)
        else:
            home_weight = 1.0  # no site fidelity for non-breeders
        score = lp + math.log(home_weight + 1e-12)

    
        # nest pull breeders vs non-breeder
        if now is not None and (DAY_START_HOUR <= now.hour <= DAY_END_HOUR):
            minutes_since_midnight = now.hour * 60 + now.minute
            pull_tick = (minutes_since_midnight % NEST_PULL_INTERVAL_MIN == 0)
        
            #  Breeders: apply nest pull every 2 hours 
            if self.bred and NEST_PULL_ACTIVE_ONLY_IF_BRED and pull_tick:
                alpha = BEHAVIOR_PARAMS["breeding"]["nest_pull_alpha"]
                if alpha > 0:
                    pull_weight = math.exp(-alpha * nest_dist_after)
                    score += math.log(pull_weight + 1e-12)
        
            # Non-breeders:  no nest pull
            elif (not self.bred) and pull_tick:
                pass

    
        #  energy & bookkeeping 
        food_gain = cov.get("FOOD", 0.0)
        take = 0.25 * food_gain  
        env.consume_food(int(cov["row"]), int(cov["col"]), take)
    
        flying_cost = ENERGY_SCALARS["distance_cost_scale"] * dist_m
        slices_per_day = (24 * 60) / self.step_minutes
        physio_cost = (ENERGY_SCALARS["baseline_daily_units"] / slices_per_day) * ENERGY_REQUIREMENT["physio"] / REQUIREMENT_TOTAL
        foraging_cost = (ENERGY_SCALARS["baseline_daily_units"] / slices_per_day) * ENERGY_REQUIREMENT["foraging"] / REQUIREMENT_TOTAL
    
        gain = take
        cost = flying_cost + physio_cost + foraging_cost
        net = gain - cost
    
        # commit new position
        self.lon, self.lat = new_lon, new_lat
        self.bearing_deg = new_bearing
        self.day_energy_gain += gain
        self.day_energy_cost += cost
        self.energy_cum += net
        self.step_count += 1

    
        return {
            "lon": self.lon, "lat": self.lat, "dist_m": dist_m, "bearing": new_bearing,
            "gain": gain, "cost": cost, "net": net, "lulc": cov.get("LULC", np.nan)
        }
    def end_of_day_update(self):
        day_req = ENERGY_SCALARS["baseline_daily_units"]
        net_today = self.day_energy_gain - self.day_energy_cost
    
    
        # daily net energy tracking for window-based breeding rule
        self.daily_net_history.append(float(net_today))
        if len(self.daily_net_history) > 120:  
            self.daily_net_history = self.daily_net_history[-120:]
    
        met_fraction = net_today / max(day_req, 1e-6)
    
        died = self.energy_cum < (DEATH_THRESHOLD * -day_req)  # unchanged death test
    
        # Reset daily tallies
        g, c = self.day_energy_gain, self.day_energy_cost
        self.day_energy_gain = 0.0
        self.day_energy_cost = 0.0
    
        if died:
            self.bred = False
            self.alive = False
    
        bred_today = False
        return g, c, met_fraction, died, bred_today



# ======== Simulation ==========
@dataclass
class SimulationConfig:
    start_year: int = 2012
    end_year: int = 2022
    season_start_mmdd: Tuple[int, int] = (3, 1)   
    season_end_mmdd: Tuple[int, int] = (8, 31)    
    step_minutes: int = STEP_MINUTES_DEFAULT
    neighbor_radius_cells: int = 2
    rng_seed: int = 42

@dataclass
class Scenario:
    lulc_percent_delta: Dict[int, float] = field(default_factory=dict)
    fragmentation_multiplier: float = 1.0

class IBM:
    def __init__(self, env: Environment, birds: List[BirdAgent], cfg: SimulationConfig, scenario: Scenario):
        self.env = env
        self.birds = birds
        self.cfg = cfg
        self.scenario = scenario
        random.seed(cfg.rng_seed)
        np.random.seed(cfg.rng_seed)

    def apply_scenario(self):
        if not self.scenario.lulc_percent_delta:
            return
        lulc = self.env.lulc.values.copy().astype(int)
        unique, counts = np.unique(lulc, return_counts=True)
        uniq_list = unique.tolist()
        count_map = dict(zip(uniq_list, counts.tolist()))

        target_counts = count_map.copy()
        for cls, pct in self.scenario.lulc_percent_delta.items():
            base = count_map.get(cls, 0)
            delta = int(round(base * (pct / 100.0)))
            target_counts[cls] = max(0, base + delta)

        donors = [c for c in uniq_list if c not in self.scenario.lulc_percent_delta]
        for cls, tgt in target_counts.items():
            cur = int((lulc == cls).sum())
            if tgt > cur:
                need = tgt - cur
                donor_counts = {d: int((lulc == d).sum()) for d in donors}
                donor_total = sum(donor_counts.values())
                if donor_total <= 0:
                    continue
                for d in donors:
                    if need <= 0: break
                    give = int(round(need * donor_counts[d] / donor_total))
                    if give <= 0: continue
                    rr, cc = np.where(lulc == d)
                    if len(rr) == 0: continue
                    take_n = min(give, len(rr), need)
                    idx = np.random.choice(len(rr), size=take_n, replace=False)
                    lulc[rr[idx], cc[idx]] = cls
                    need -= take_n

        self.env.lulc.values[:] = lulc
        if self.scenario.fragmentation_multiplier != 1.0:
            for k, arr in self.env.frag.items():
                arr.values[:] = arr.values * self.scenario.fragmentation_multiplier
        self.env._init_food_tiled()

    def run(self, years: List[int], output_dir: str):
        ensure_dir(output_dir)
        for year in years:
            print(f"\n=== Simulating {year} ===")
            self._run_year(year, output_dir)
            

    def _run_year(self, year: int, output_dir: str):
        self.env._init_food_tiled()
        for b in self.birds:
            b.alive = True
            b.energy_cum = 0.0
            b.day_energy_cost = 0.0
            b.day_energy_gain = 0.0
            b.bred = False
            b.step_count = 0 
            b.reset_to_nest()   

        yout_dir = os.path.join(output_dir, "steps", str(year))
        fig_dir = os.path.join(output_dir, "figures", str(year))
        sum_dir = os.path.join(output_dir, "summaries")
        ensure_dir(yout_dir); ensure_dir(fig_dir); ensure_dir(sum_dir)

        season_start = datetime(year, *self.cfg.season_start_mmdd)
        season_end = datetime(year, *self.cfg.season_end_mmdd, 23, 59)
        step = timedelta(minutes=self.cfg.step_minutes)
        t = season_start

        writers = {}
        for b in self.birds:
            fpath = os.path.join(yout_dir, f"{b.bird_id}.csv")
            writers[b.bird_id] = open(fpath, "w", encoding="utf-8")
            writers[b.bird_id].write("timestamp,step,lon,lat,dist_m,bearing,gain,cost,net,lulc\n")

        day_counter = 0
        breed_checked = set()
        deaths = set()

        steps_total = int(((season_end - season_start).total_seconds() // (self.cfg.step_minutes * 60)) + 1)
        pbar = tqdm(total=steps_total)

        last_day = season_start.date()  

        while t <= season_end:
            # daytime-only movement gate
            if (t.hour < DAY_START_HOUR) or (t.hour > DAY_END_HOUR):
                t += step
                pbar.update(1)
                continue

            self.env.renew_food_if_due(t)

            # Daily nest return-ONLY for breeders
            if t.date() != last_day:
                for b in self.birds:
                    if not b.alive:
                        continue
                    # Only confirmed breeders return to nest each new day
                    if b.bred:
                        b.reset_to_nest()
                    # Non-breeders continue from their current position 
                last_day = t.date()

            for b in self.birds:
                if not b.alive:
                    continue
                rec = b.step(self.env, t)
                if rec:
                    writers[b.bird_id].write(
                        f"{t.isoformat()},{b.step_count},"
                        f"{rec['lon']:.6f},{rec['lat']:.6f},{rec['dist_m']:.2f},{rec['bearing']:.2f},"
                        f"{rec['gain']:.4f},{rec['cost']:.4f},{rec['net']:.4f},"
                        f"{int(rec['lulc']) if not math.isnan(rec['lulc']) else ''}\n"
                    )


            # end-of-day energy updates: check at last active slot
            # roll this at 20:00 boundary-only day time movement
            if t.hour == DAY_END_HOUR:
                day_counter += 1
            
                #  End-of-day energy update 
                for b in self.birds:
                    if not b.alive:
                        continue
                    g, c, frac, died, _ = b.end_of_day_update()
                    if died:
                        b.alive = False
                        deaths.add(b.bird_id)
            
                #  Breeding decision after window 
                if day_counter >= BREED_WINDOW_DAYS:
                    for b in self.birds:
                        if not b.alive:
                            continue
                        b.bred = self._should_breed(b)
            t += step
            pbar.update(1)
        pbar.close()

        for f in writers.values():
            f.close()

        rows = []
        for b in self.birds:
            rows.append({"bird_id": b.bird_id, 
                         "alive": b.alive, 
                         "bred": b.bred, 
                         "energy_cum": round(b.energy_cum, 3),
                         "step": b.step_count})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(sum_dir, f"summary_{year}.csv"), index=False)

        self._plot_year_tracks(year, yout_dir, fig_dir)
        self._plot_year_energy(year, sum_dir, fig_dir)
        self._plot_year_tracks_by_status(year, yout_dir, fig_dir, self.birds)


    def _plot_year_tracks(self, year: int, yout_dir: str, fig_dir: str):
        plt.figure(figsize=(8, 8))
        for birdfile in os.listdir(yout_dir):
            if not birdfile.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(yout_dir, birdfile))
            if df.empty:
                continue
            plt.plot(df['lon'], df['lat'], linewidth=0.6, alpha=0.7)
        plt.title(f"Tracks — {year}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        out = os.path.join(fig_dir, f"tracks_{year}.png")
        plt.savefig(out, dpi=200)
        plt.close()

    def _plot_year_energy(self, year: int, sum_dir: str, fig_dir: str):
        df = pd.read_csv(os.path.join(sum_dir, f"summary_{year}.csv"))
        alive = int(df['alive'].sum())
        bred = int(df['bred'].sum())
        plt.figure(figsize=(6, 4))
        plt.bar(["Alive", "Bred"], [alive, bred])
        plt.title(f"Year {year}: Alive & Bred counts")
        plt.tight_layout()
        out = os.path.join(fig_dir, f"energy_summary_{year}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        
    def _should_breed(self, b: BirdAgent) -> bool:
        if not b.alive:
            return False
        window = BREED_WINDOW_DAYS
        day_req = ENERGY_SCALARS["baseline_daily_units"]
        if len(b.daily_net_history) < window:
            return False
        recent = b.daily_net_history[-window:]
        window_net = float(sum(recent))
    
        #  0.3 means 30% of daily baseline per day for breeding
        required = BREED_MIN_EXCESS * day_req * window
        return window_net >= required

    def _plot_year_tracks_by_status(self, year: int, yout_dir: str, fig_dir: str, birds: List[BirdAgent]):
        breeders, nonbreeders = [], []
        for b in birds:
            fpath = os.path.join(yout_dir, f"{b.bird_id}.csv")
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue
            df = pd.read_csv(fpath)
            if b.bred:
                breeders.append(df)
            else:
                nonbreeders.append(df)
    
        # Breeders 
        if breeders:
            plt.figure(figsize=(8, 8))
            for df in breeders:
                plt.plot(df["lon"], df["lat"], linewidth=0.6, alpha=0.7)
            plt.title(f"Breeder Tracks — {year}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"tracks_breeders_{year}.png"), dpi=200)
            plt.close()
    
        # Non-breeders 
        if nonbreeders:
            plt.figure(figsize=(8, 8))
            for df in nonbreeders:
                plt.plot(df["lon"], df["lat"], linewidth=0.6, alpha=0.7)
            plt.title(f"Non-breeder Tracks — {year}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"tracks_nonbreeders_{year}.png"), dpi=200)
            plt.close()



#Single year simulation
def run_simulation():
    env = Environment(LULC_PATH, NDVI_PATH, NDWI_PATH, FRAG_PATHS)
    birds = load_birds_from_points(POINTS_PATH, FIXED_RSF_PATH, RANDOM_RSF_PATH, STEP_MINUTES_DEFAULT)
    cfg = SimulationConfig()
    scenario = Scenario()
    ibm = IBM(env, birds, cfg, scenario)
    ibm.apply_scenario()
    ibm.run([2012], OUTPUT_DIR)

# Multiple year simulation loop
# Require higher device configuration
def run_simulation_multi(years: list[int]):
    
    for year in years:
        print(f"\n🚀 Starting simulation for {year}...")

        # --- Construct year-specific raster paths ---
        LULC = f"D:/4th Semester/thesis/models/GeoTiffs/New/LC_{year}_07.tif"
        NDVI = f"D:/4th Semester/thesis/models/GeoTiffs/NDVI_NDWI/NDVI_{year}_JulAug_30m.tif"
        NDWI = f"D:/4th Semester/thesis/models/GeoTiffs/NDVI_NDWI/MNDWI_{year}_JulAug_30m.tif"

        # --- Check raster existence before proceeding ---
        missing = False
        for f in [LULC, NDVI, NDWI]:
            if not os.path.exists(f):
                print(f"⚠️ Missing raster file: {f}")
                missing = True
        if missing:
            print("   Skipping this year.\n")
            continue

        # --- Initialize environment + birds ---
        env = Environment(LULC, NDVI, NDWI, FRAG_PATHS)
        birds = load_birds_from_points(POINTS_PATH, FIXED_RSF_PATH, RANDOM_RSF_PATH, STEP_MINUTES_DEFAULT)
        cfg = SimulationConfig(start_year=year, end_year=year)
        scenario = Scenario()
        ibm = IBM(env, birds, cfg, scenario)
        ibm.apply_scenario()

        # --- Run simulation for this year ---
        out_dir_year = os.path.join(OUTPUT_DIR, str(year))
        ibm.run([year], out_dir_year)
        print(f"✅ Finished simulation for {year}. Results saved to {out_dir_year}\n")

if __name__ == "__main__":
    run_simulation_multi([2012,2018,2024])



