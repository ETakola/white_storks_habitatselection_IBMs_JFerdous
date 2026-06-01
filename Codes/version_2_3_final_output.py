# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 12:01:09 2025

@author: jannatul

IBM for white stork movement and breeding dynamics

Logic:
- fixed daytime movement only (5am to 8pm)
- breeding decision is made early in the season (day 15 to day 45)
- birds enter breeder behavior once they exceed the early breeding threshold
- final breeding success is evaluated from sustained energy over the last 30 days
- both non-breeders and failed breeders are eligible to shift
- if an unsuccessful bird finds a sufficiently improved site, it shifts center
- if it does not find a suitable site, next season starts from end-of-season location
- successful breeders start next season from breeding center
- a stable final_state is assigned to each bird for yearly summary/export

Model notes
Important:
- ED raster is raw, but RSF uses ED_scaled
- therefore raw ED is converted using the same mean and SD from R model training
"""

from __future__ import annotations
import os
import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyproj import Transformer

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# FILE PATHS
# =========================================================
LULC_PATH = "E:/IBM_stork_resources/LC_2012.tif"
NDVI_PATH = "E:/IBM_stork_resources/NDVI_2012_JulAug.tif"
NDWI_PATH = "E:/IBM_stork_resources/MNDWI_2012_JulAug.tif"

FRAG_PATHS = {
    "SHEI": "E:/IBM_stork_resources/SHEI_2012.tif",
    "ED": "E:/IBM_stork_resources/ED_2012.tif",
}

POINTS_PATH = "E:/IBM_stork_resources/MPIAB_inside_data.xlsx"
RANDOM_RSF_PATH = "E:/IBM_stork_resources/IBM_bird_random_effects_ndvi.csv"
FIXED_RSF_PATH = "E:/IBM_stork_resources/IBM_fixed_effects_ndvi.csv"
OUTPUT_DIR = "E:/IBM_stork_resources/Output"


# =========================================================
# MODEL CONSTANTS
# =========================================================
STEP_MINUTES_DEFAULT = 30
FOOD_RENEW_DAYS = 2

BREED_DECISION_START_DAY = 15
BREED_DECISION_END_DAY = 45
BREED_DECISION_WINDOW_DAYS = 15
BREED_SUCCESS_WINDOW_DAYS = 30

ENERGY_REQUIREMENT = {
    "physio": 0.75,
    "foraging": 0.10,
    "flying": 0.10,
    "other": 0.05,
}
REQUIREMENT_TOTAL = float(sum(ENERGY_REQUIREMENT.values()))

# 30% surplus threshold
BREED_MIN_EXCESS = 0.30

# death occurs if cumulative energy debt exceeds 60% of one daily requirement
DEATH_THRESHOLD = 0.60

DEFAULT_LULC_ENERGY = {
    10: 3.0, 11: 3.0, 20: 2.5, 51: 1.0, 61: 1.5, 62: 0.5, 71: 0.5,
    72: 0.1, 81: 0.1, 82: 0.1, 130: 3.0, 182: 4.0, 190: 1.5, 210: 0.1,
}

ENERGY_SCALARS = {
    "ndvi_gain_scale": 0.3,
    "ndwi_gain_scale": 0.6,
    "fragmentation_gain_scale": 0.7,
    "distance_cost_scale": 0.00015,
    "baseline_daily_units": 10.0,
}

# Replace with exact RSF scaling values from R
ED_MEAN = 18.36658
ED_SD = 8.768729

TILE = 2048


# =========================================================
# ECOLOGICAL DECISION RULES
# =========================================================
PROSPECTING_START_DAY = 45

MIN_SHIFT_YEARS_FAILED = 1
MIN_BETTER_SITE_SCORE = 0.10
MIN_SHIFT_DISTANCE_M = 3000.0

LOYALTY_INITIAL = 1.0
LOYALTY_GAIN_IF_BRED = 0.40
LOYALTY_LOSS_IF_FAILED = 0.15
LOYALTY_MIN = 0.2
LOYALTY_MAX = 3.0


# =========================================================
# ACTIVITY AND NEST BEHAVIOR
# =========================================================
DAY_START_HOUR = 5
DAY_END_HOUR = 20

HOME_DECAY = 0.0005
NEST_PULL_INTERVAL_MIN = 120
NEST_PULL_ACTIVE_ONLY_IF_BRED = True

BEHAVIOR_PARAMS = {
    "breeding": {
        "step_mu_m": 350.0,
        "step_sigma_m": 180.0,
        "turn_kappa": 3.0,
        "nest_pull_alpha": 0.0008,
        "max_foraging_radius_m": 28100.0,
        "candidate_n": 15,
    },
    "nonbreeding": {
        "step_mu_m": 1400.0,
        "step_sigma_m": 600.0,
        "turn_kappa": 1.0,
        "nest_pull_alpha": 0.0,
        "max_foraging_radius_m": 80000.0,
        "candidate_n": 18,
    },
    "prospecting": {
        "step_mu_m": 2200.0,
        "step_sigma_m": 900.0,
        "turn_kappa": 0.8,
        "nest_pull_alpha": 0.0,
        "max_foraging_radius_m": 200000.0,
        "candidate_n": 22,
    },
}


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([], dtype=float)
    z = (x - np.nanmax(x)) / max(temperature, 1e-6)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(e) / len(e)
    return e / s


def within_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def move_along_bearing(lon, lat, dist_m, bearing_deg):
    R = 6371000.0
    d = dist_m / R
    brg = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brg)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lon2), math.degrees(lat2))


# =========================================================
# ENVIRONMENT
# =========================================================
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
    transformer_to_raster: Transformer = field(init=False)
    transformer_to_lonlat: Transformer = field(init=False)
    food_base: np.ndarray = field(init=False)
    food_stock: np.ndarray = field(init=False)
    last_renewal: Optional[datetime] = field(init=False)

    def __post_init__(self):
        self.lulc = rxr.open_rasterio(self.lulc_path, masked=True).squeeze()

        with rasterio.open(self.lulc_path) as src:
            self.transform = src.transform
            self.crs = src.crs.to_string() if src.crs else ""
            raster_crs = src.crs

        self.transformer_to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        self.transformer_to_lonlat = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

        self.ndvi = rxr.open_rasterio(self.ndvi_path, masked=True).squeeze().rio.reproject_match(self.lulc)
        self.ndwi = rxr.open_rasterio(self.ndwi_path, masked=True).squeeze().rio.reproject_match(self.lulc)

        for key, path in self.frag_paths.items():
            self.frag[key.upper()] = rxr.open_rasterio(path, masked=True).squeeze().rio.reproject_match(self.lulc)

        self._init_food_tiled()

    def _lonlat_to_rowcol(self, lon: float, lat: float):
        x, y = self.transformer_to_raster.transform(lon, lat)
        r, c = rowcol(self.transform, x, y)
        return int(r), int(c)

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
                if np.all(np.isnan(block)):
                    continue
                bmn = np.nanmin(block)
                bmx = np.nanmax(block)
                if bmn < mn:
                    mn = bmn
                if bmx > mx:
                    mx = bmx

        if not np.isfinite(mn):
            mn = 0.0
        if not np.isfinite(mx):
            mx = 1.0
        if mx - mn < 1e-9:
            mx = mn + 1e-9

        return float(mn), float(mx)

    def _init_food_tiled(self):
        lulc = self.lulc.values.astype(np.int32, copy=False)
        h, w = lulc.shape
        base = np.zeros((h, w), dtype=np.float32)

        for cls, e in DEFAULT_LULC_ENERGY.items():
            mask = (lulc == cls)
            base[mask] = float(e)

        ndvi = self.ndvi.values
        ndwi = self.ndwi.values

        for r0 in range(0, h, TILE):
            r1 = min(h, r0 + TILE)
            for c0 in range(0, w, TILE):
                c1 = min(w, c0 + TILE)

                ndvi_block = np.array(ndvi[r0:r1, c0:c1], dtype=np.float32, copy=True)
                ndvi_block = np.nan_to_num(ndvi_block, nan=0.0)
                np.clip(ndvi_block, 0.0, 1.0, out=ndvi_block)
                base[r0:r1, c0:c1] += ENERGY_SCALARS["ndvi_gain_scale"] * ndvi_block

                ndwi_block = np.array(ndwi[r0:r1, c0:c1], dtype=np.float32, copy=True)
                ndwi_block = np.nan_to_num(ndwi_block, nan=0.0)
                np.clip(ndwi_block, 0.0, 1.0, out=ndwi_block)
                base[r0:r1, c0:c1] += ENERGY_SCALARS["ndwi_gain_scale"] * ndwi_block

        if self.frag:
            stats = {name: self._min_max_tiled(arr) for name, arr in self.frag.items()}

            for r0 in range(0, h, TILE):
                r1 = min(h, r0 + TILE)
                for c0 in range(0, w, TILE):
                    c1 = min(w, c0 + TILE)
                    comps = []

                    for name, arr in self.frag.items():
                        mn, mx = stats[name]
                        block = np.array(arr.values[r0:r1, c0:c1], dtype=np.float32, copy=True)
                        block = np.nan_to_num(block, nan=mn)
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
        r, c = self._lonlat_to_rowcol(lon, lat)

        if not within_bounds(self.lulc.shape, r, c):
            return {}

        lulc_val = self.lulc.values[r, c]
        ndvi_val = self.ndvi.values[r, c]
        ndwi_val = self.ndwi.values[r, c]

        if np.isnan(lulc_val):
            return {}

        out = {
            "LULC": float(lulc_val),
            "NDVI": float(np.clip(np.nan_to_num(ndvi_val, nan=0.0), 0.0, 1.0)),
            "NDWI": float(np.clip(np.nan_to_num(ndwi_val, nan=0.0), 0.0, 1.0)),
            "FOOD": float(np.nan_to_num(self.food_stock[r, c], nan=0.0)),
        }

        for k, arr in self.frag.items():
            raw_val = float(np.nan_to_num(arr.values[r, c], nan=0.0))
            out[k] = raw_val

            if k.upper() == "ED":
                out["ED_raw"] = raw_val
                out["ED_scaled"] = (raw_val - ED_MEAN) / ED_SD if ED_SD != 0 else 0.0

            if k.upper() == "SHEI":
                out["SHEI_raw"] = raw_val

        out["row"] = r
        out["col"] = c
        return out

    def consume_food(self, r, c, amount):
        if within_bounds(self.food_stock.shape, r, c):
            self.food_stock[r, c] = max(0.0, self.food_stock[r, c] - amount)


# =========================================================
# RSF MODEL
# =========================================================
@dataclass
class RSFModel:
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

        ndvi = cov.get("NDVI")
        if ndvi is not None and np.isfinite(ndvi):
            lp += self._get("ndvi") * ndvi + self._get("NDVI") * ndvi

        lulc = cov.get("LULC")
        if lulc is not None and np.isfinite(lulc):
            try:
                code = int(lulc)
                for key in (
                    f"LULC_class{code}",
                    f"lulc_class{code}",
                    f"LULC_{code}",
                    f"lulc_{code}",
                ):
                    if key in self.coeffs:
                        lp += self.coeffs[key]
                        break
            except Exception:
                pass

        ed_scaled = cov.get("ED_scaled")
        if ed_scaled is not None and np.isfinite(ed_scaled):
            lp += self._get("ED_scaled") * ed_scaled + self._get("ed_scaled") * ed_scaled

        return float(lp)


# =========================================================
# RSF HELPERS
# =========================================================
def _read_fixed_effects(path: str) -> Dict[str, float]:
    fx = pd.read_csv(path)
    fx.columns = [c.strip().lower() for c in fx.columns]

    if "term" not in fx.columns:
        raise ValueError("Fixed-effects file must contain column: term")

    coef_col = None
    for candidate in ["coef", "beta", "estimate", "estim", "value"]:
        if candidate in fx.columns:
            coef_col = candidate
            break

    if coef_col is None:
        raise ValueError(
            f"Fixed-effects file must contain a coefficient column such as coef or beta. "
            f"Found columns: {list(fx.columns)}"
        )

    fx = fx.dropna(subset=["term", coef_col])

    out = {}
    for _, row in fx.iterrows():
        term = str(row["term"]).strip()
        try:
            value = float(row[coef_col])
        except Exception:
            continue
        out[term] = value

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

    raise ValueError("Random-effects file must include a bird_id column.")


def _combine_fixed_random_for_bird(fixed: Dict[str, float], rrow: pd.Series) -> Dict[str, float]:
    cols_lower = {str(c).lower() for c in rrow.index}
    has_full_intercept = any(c in cols_lower for c in ["(intercept)", "intercept"])

    if has_full_intercept:
        combined = {}
        for k, v in rrow.items():
            if pd.isna(v):
                continue
            if isinstance(v, (int, float, np.floating)):
                combined[str(k)] = float(v)

        if "Intercept" in combined and "(Intercept)" not in combined:
            combined["(Intercept)"] = combined.pop("Intercept")

        for fk, fv in fixed.items():
            if fk not in combined:
                combined[fk] = fv

        return combined

    combined = dict(fixed)

    for k, v in rrow.items():
        if pd.isna(v):
            continue

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


# =========================================================
# LOAD BIRDS
# =========================================================
def load_birds_from_points(points_file: str, fixed_effects_csv: str, random_effects_csv: str, step_minutes: int):
    ext = os.path.splitext(points_file)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(points_file, dtype=str)
    else:
        df = pd.read_csv(points_file, dtype=str)

    df.columns = [c.strip().lower().replace(".", "_").replace(" ", "_") for c in df.columns]

    if {"individual_local_identifier", "location_long", "location_lat", "timestamp"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["individual_local_identifier", "timestamp"])
        nests = df.groupby("individual_local_identifier").first().reset_index()
        nests = nests.rename(
            columns={
                "individual_local_identifier": "bird_id",
                "location_long": "lon",
                "location_lat": "lat",
            }
        )
    elif {"bird_id", "lon", "lat"}.issubset(df.columns):
        nests = df.copy()
    else:
        raise ValueError("Input points file must contain either nest-style columns or bird_id/lon/lat.")

    nests["bird_id"] = (
        nests["bird_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper().str.replace(r"\s+", "", regex=True)
    )

    fixed = _read_fixed_effects(fixed_effects_csv)

    rnd = pd.read_csv(random_effects_csv)
    rnd.columns = [c.strip() for c in rnd.columns]

    bid_col = next((c for c in rnd.columns if c.lower() == "bird_id"), None)
    if bid_col is None:
        raise ValueError("Random-effects file must include a bird_id column.")

    rnd[bid_col] = (
        rnd[bid_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper().str.replace(r"\s+", "", regex=True)
    )
    rnd_wide = _wide_or_pivot_randoms(rnd)

    print("\nMatching IDs:")
    print("  Nests IDs:", nests["bird_id"].tolist()[:5])
    print("  RSF IDs:", list(rnd_wide.index[:5]))
    print(f"  Total overlap: {len(set(nests['bird_id']) & set(rnd_wide.index))} birds")

    birds = []
    for _, row in nests.iterrows():
        bid = str(row["bird_id"]).strip().upper()

        if bid in rnd_wide.index:
            rrow = rnd_wide.loc[bid]
            if isinstance(rrow, pd.DataFrame):
                rrow = rrow.iloc[0]
            combined = _combine_fixed_random_for_bird(fixed, rrow)
        else:
            print(f"No random RSF row for bird {bid} — using fixed effects only.")
            combined = dict(fixed)

        bird = BirdAgent(
            bird_id=bid,
            rsf=RSFModel(coeffs=combined),
            start_lon=float(row["lon"]),
            start_lat=float(row["lat"]),
            step_minutes=step_minutes,
        )
        birds.append(bird)

    print(f"Loaded {len(birds)} birds for simulation.")
    return birds


# =========================================================
# BIRD AGENT
# =========================================================
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

    energy_cumulative: float = 0.0
    day_energy_gain: float = 0.0
    day_energy_cost: float = 0.0

    attempted_breeding: bool = False
    bred: bool = False
    final_state: str = "unknown"

    step_count: int = 0
    daily_net_history: List[float] = field(default_factory=list)

    home_range_shifted: bool = False
    original_nest_lon: float = field(init=False)
    original_nest_lat: float = field(init=False)
    failed_breeding_attempts: int = 0
    exploration_mode: bool = False
    shift_year: Optional[int] = None
    shift_history: List[Dict] = field(default_factory=list)

    current_center_lon: float = field(init=False)
    current_center_lat: float = field(init=False)
    current_center_score: float = field(default=-np.inf)
    best_site_lon: float = field(init=False)
    best_site_lat: float = field(init=False)
    best_site_score: float = field(default=-np.inf)
    prospecting_days: int = 0
    breeding_site_loyalty: float = field(default=LOYALTY_INITIAL)
    _last_habitat_quality: float = 0.0

    next_start_lon: float = field(init=False)
    next_start_lat: float = field(init=False)
    season_end_lon: float = field(init=False)
    season_end_lat: float = field(init=False)

    def __post_init__(self):
        self.lon = self.start_lon
        self.lat = self.start_lat

        self.original_nest_lon = self.start_lon
        self.original_nest_lat = self.start_lat

        self.current_center_lon = self.start_lon
        self.current_center_lat = self.start_lat

        self.best_site_lon = self.start_lon
        self.best_site_lat = self.start_lat

        self.next_start_lon = self.start_lon
        self.next_start_lat = self.start_lat
        self.season_end_lon = self.start_lon
        self.season_end_lat = self.start_lat

    def reset_to_center(self):
        self.lon = self.current_center_lon
        self.lat = self.current_center_lat

    def begin_new_year(self, env: Environment):
        self.energy_cumulative = 0.0
        self.day_energy_cost = 0.0
        self.day_energy_gain = 0.0
        self.bred = False
        self.attempted_breeding = False
        self.final_state = "unknown"
        self.step_count = 0
        self.daily_net_history = []
        self.prospecting_days = 0
        self.exploration_mode = False

        self.lon = self.next_start_lon
        self.lat = self.next_start_lat

        cov = env.sample_cell(self.current_center_lon, self.current_center_lat)
        if cov:
            self.current_center_score = self.rsf.linear_predictor(cov)
        else:
            self.current_center_score = -np.inf

        self.best_site_lon = self.current_center_lon
        self.best_site_lat = self.current_center_lat
        self.best_site_score = self.current_center_score
        
    def _get_mode(self, season_day: int) -> str:
        if self.attempted_breeding:
            return "breeding"
        if season_day >= PROSPECTING_START_DAY or self.failed_breeding_attempts >= 1:
            return "prospecting"
        return "nonbreeding"

    def should_attempt_breeding(self) -> bool:
        if len(self.daily_net_history) < BREED_DECISION_WINDOW_DAYS:
            return False
        recent = self.daily_net_history[-BREED_DECISION_WINDOW_DAYS:]
        window_net = float(sum(recent))
        day_req = ENERGY_SCALARS["baseline_daily_units"]
        required = BREED_MIN_EXCESS * day_req * BREED_DECISION_WINDOW_DAYS
        return window_net >= required

    def check_final_breeding_success(self) -> Tuple[bool, float, float]:
        if len(self.daily_net_history) < BREED_SUCCESS_WINDOW_DAYS:
            day_req = ENERGY_SCALARS["baseline_daily_units"]
            required = BREED_MIN_EXCESS * day_req * BREED_SUCCESS_WINDOW_DAYS
            return False, np.nan, required

        recent = self.daily_net_history[-BREED_SUCCESS_WINDOW_DAYS:]
        window_net = float(sum(recent))
        day_req = ENERGY_SCALARS["baseline_daily_units"]
        required = BREED_MIN_EXCESS * day_req * BREED_SUCCESS_WINDOW_DAYS
        return window_net >= required, window_net, required

    def step(self, env: Environment, now: datetime, season_day: int) -> Dict[str, float]:
        mode = self._get_mode(season_day)
        params = BEHAVIOR_PARAMS[mode]

        mu = params["step_mu_m"]
        sigma = params["step_sigma_m"]
        kappa = params["turn_kappa"]
        n_candidates = params["candidate_n"]

        center_lon = self.current_center_lon
        center_lat = self.current_center_lat

        candidates = []

        for _ in range(n_candidates):
            dist_m = max(0.0, np.random.normal(mu, sigma))
            turn_rad = np.random.vonmises(0.0, kappa)
            new_bearing = (self.bearing_deg + math.degrees(turn_rad)) % 360
            new_lon, new_lat = move_along_bearing(self.lon, self.lat, dist_m, new_bearing)

            center_dist = haversine_m(center_lon, center_lat, new_lon, new_lat)
            arena = params["max_foraging_radius_m"]

            if arena > 0 and center_dist > arena:
                over = center_dist - arena
                penalty = math.exp(-0.001 * over)
                adj_dist = dist_m * penalty
                new_lon, new_lat = move_along_bearing(self.lon, self.lat, adj_dist, new_bearing)
                center_dist = haversine_m(center_lon, center_lat, new_lon, new_lat)

            cov = env.sample_cell(new_lon, new_lat)
            if not cov:
                continue

            rsf_score = self.rsf.linear_predictor(cov)

            if mode == "breeding":
                home_weight = math.exp(-HOME_DECAY * center_dist)
                rsf_score += math.log(home_weight + 1e-12)

            if now is not None and self.attempted_breeding and NEST_PULL_ACTIVE_ONLY_IF_BRED:
                minutes_since_midnight = now.hour * 60 + now.minute
                if minutes_since_midnight % NEST_PULL_INTERVAL_MIN == 0:
                    alpha = params["nest_pull_alpha"]
                    if alpha > 0:
                        rsf_score += -alpha * center_dist

            candidates.append(
                {
                    "lon": new_lon,
                    "lat": new_lat,
                    "bearing": new_bearing,
                    "dist_m": dist_m,
                    "center_dist": center_dist,
                    "cov": cov,
                    "score": rsf_score,
                    "mode": mode,
                }
            )

        if not candidates:
            return self._stay_in_place(env)

        scores = np.array([c["score"] for c in candidates], dtype=float)
        temperature = 0.8 if np.any(np.isfinite(scores) & (scores > 0)) else 2.0
        probs = softmax(scores, temperature=temperature)
        chosen = candidates[np.random.choice(len(candidates), p=probs)]

        self._last_habitat_quality = float(chosen["score"])

        if chosen["score"] > self.best_site_score:
            self.best_site_score = float(chosen["score"])
            self.best_site_lon = float(chosen["lon"])
            self.best_site_lat = float(chosen["lat"])

        if chosen["mode"] == "prospecting":
            self.prospecting_days = max(self.prospecting_days, season_day)

        return self._execute_movement(chosen, env)

    def _execute_movement(self, chosen: Dict, env: Environment) -> Dict[str, float]:
        self.lon = chosen["lon"]
        self.lat = chosen["lat"]
        self.bearing_deg = chosen["bearing"]

        food_gain = chosen["cov"].get("FOOD", 0.0)
        take = 0.25 * food_gain
        env.consume_food(int(chosen["cov"]["row"]), int(chosen["cov"]["col"]), take)

        dist_m = chosen["dist_m"]
        slices_per_day = (24 * 60) / self.step_minutes

        flying_cost = ENERGY_SCALARS["distance_cost_scale"] * dist_m
        physio_cost = (
            (ENERGY_SCALARS["baseline_daily_units"] / slices_per_day)
            * ENERGY_REQUIREMENT["physio"]
            / REQUIREMENT_TOTAL
        )
        foraging_cost = (
            (ENERGY_SCALARS["baseline_daily_units"] / slices_per_day)
            * ENERGY_REQUIREMENT["foraging"]
            / REQUIREMENT_TOTAL
        )

        gain = take
        cost = flying_cost + physio_cost + foraging_cost
        net = gain - cost

        self.day_energy_gain += gain
        self.day_energy_cost += cost
        self.energy_cumulative += net
        self.step_count += 1

        return {
            "lon": self.lon,
            "lat": self.lat,
            "dist_m": dist_m,
            "bearing": self.bearing_deg,
            "gain": gain,
            "cost": cost,
            "net": net,
            "lulc": chosen["cov"].get("LULC", np.nan),
            "ndvi":chosen["cov"].get("NDVI", np.nan),
            "ndwi":chosen["cov"].get("NDWI", np.nan),
            "mode": chosen.get("mode", ""),
            "site_score": chosen.get("score", np.nan),
        }

    def _stay_in_place(self, env: Environment) -> Dict[str, float]:
        cov = env.sample_cell(self.lon, self.lat)
        if not cov:
            return {}

        slices_per_day = (24 * 60) / self.step_minutes
        physio_cost = (
            (ENERGY_SCALARS["baseline_daily_units"] / slices_per_day)
            * ENERGY_REQUIREMENT["physio"]
            / REQUIREMENT_TOTAL
        )

        self.energy_cumulative -= physio_cost
        self.day_energy_cost += physio_cost
        self.step_count += 1

        return {
            "lon": self.lon,
            "lat": self.lat,
            "dist_m": 0.0,
            "bearing": self.bearing_deg,
            "gain": 0.0,
            "cost": physio_cost,
            "net": -physio_cost,
            "lulc": cov.get("LULC", np.nan),
            "ndvi":cov.get("NDVI", np.nan),
            "ndwi":cov.get("NDWI", np.nan),
            "mode": "rest",
            "site_score": np.nan,
        }

    def end_of_day_update(self):
        day_req = ENERGY_SCALARS["baseline_daily_units"]
        net_today = self.day_energy_gain - self.day_energy_cost

        self.daily_net_history.append(float(net_today))
        if len(self.daily_net_history) > 180:
            self.daily_net_history = self.daily_net_history[-180:]

        died = self.energy_cumulative < (-DEATH_THRESHOLD * day_req)

        gain_today = self.day_energy_gain
        cost_today = self.day_energy_cost

        self.day_energy_gain = 0.0
        self.day_energy_cost = 0.0

        if died:
            self.bred = False
            self.attempted_breeding = False
            self.alive = False
            self.final_state = "dead"

        return gain_today, cost_today, net_today, died, False

    def end_of_season_update(self, year: int) -> Dict:
        self.season_end_lon = self.lon
        self.season_end_lat = self.lat

        success, window_net, required = self.check_final_breeding_success()
        self.bred = bool(success)

        if self.bred:
            self.final_state = "successful_breeder"
            self.failed_breeding_attempts = 0
            self.exploration_mode = False
            self.breeding_site_loyalty = min(LOYALTY_MAX, self.breeding_site_loyalty + LOYALTY_GAIN_IF_BRED)

            self.next_start_lon = self.current_center_lon
            self.next_start_lat = self.current_center_lat

            return {
                "shifted": False,
                "reason": "successful_breeding",
                "window_net": window_net,
                "required": required,
                "attempted_breeding": self.attempted_breeding,
                "bred": self.bred,
                "final_state": self.final_state,
                "best_site_score": self.best_site_score,
                "center_score": self.current_center_score,
                "site_improvement_raw": self.best_site_score - self.current_center_score,
                "site_improvement_effective": (self.best_site_score - self.current_center_score) - self.breeding_site_loyalty,
                "shift_distance_m": haversine_m(
                    self.current_center_lon, self.current_center_lat,
                    self.best_site_lon, self.best_site_lat
                ),
                "loyalty": self.breeding_site_loyalty,
                "next_start_lon": self.next_start_lon,
                "next_start_lat": self.next_start_lat,
            }

        self.failed_breeding_attempts += 1
        self.exploration_mode = True
        self.breeding_site_loyalty = max(LOYALTY_MIN, self.breeding_site_loyalty - LOYALTY_LOSS_IF_FAILED)

        site_improvement_raw = self.best_site_score - self.current_center_score
        site_improvement_effective = site_improvement_raw - self.breeding_site_loyalty
        shift_distance = haversine_m(
            self.current_center_lon, self.current_center_lat,
            self.best_site_lon, self.best_site_lat
        )

        should_shift = (
            self.failed_breeding_attempts >= MIN_SHIFT_YEARS_FAILED
            and np.isfinite(site_improvement_effective)
            and site_improvement_effective >= MIN_BETTER_SITE_SCORE
            and shift_distance >= MIN_SHIFT_DISTANCE_M
        )

        if should_shift:
            old_lon = self.current_center_lon
            old_lat = self.current_center_lat

            self.current_center_lon = self.best_site_lon
            self.current_center_lat = self.best_site_lat
            self.current_center_score = self.best_site_score

            self.start_lon = self.current_center_lon
            self.start_lat = self.current_center_lat
            self.home_range_shifted = True
            self.shift_year = year

            self.next_start_lon = self.current_center_lon
            self.next_start_lat = self.current_center_lat
            self.final_state = "shifter"

            self.shift_history.append(
                {
                    "year": year,
                    "from_lon": old_lon,
                    "from_lat": old_lat,
                    "to_lon": self.current_center_lon,
                    "to_lat": self.current_center_lat,
                    "reason": "energetic_failure_found_better_site",
                    "site_improvement_raw": site_improvement_raw,
                    "site_improvement_effective": site_improvement_effective,
                    "shift_distance_m": shift_distance,
                    "loyalty": self.breeding_site_loyalty,
                }
            )

            print(
                f"Bird {self.bird_id} shifted breeding center in {year} "
                f"(raw_improvement={site_improvement_raw:.3f}, "
                f"effective_improvement={site_improvement_effective:.3f}, "
                f"distance={shift_distance:.1f} m, loyalty={self.breeding_site_loyalty:.2f})"
            )

            return {
                "shifted": True,
                "reason": "energetic_failure_found_better_site",
                "window_net": window_net,
                "required": required,
                "attempted_breeding": self.attempted_breeding,
                "bred": self.bred,
                "final_state": self.final_state,
                "best_site_score": self.best_site_score,
                "center_score": self.current_center_score,
                "site_improvement_raw": site_improvement_raw,
                "site_improvement_effective": site_improvement_effective,
                "shift_distance_m": shift_distance,
                "loyalty": self.breeding_site_loyalty,
                "next_start_lon": self.next_start_lon,
                "next_start_lat": self.next_start_lat,
            }

        if self.attempted_breeding:
            self.final_state = "failed_breeder"
        else:
            self.final_state = "non_breeder"

        # if bird did not shift to a clearly better breeding center,
        # next season starts from its end-of-season location
        self.next_start_lon = self.season_end_lon
        self.next_start_lat = self.season_end_lat

        return {
            "shifted": False,
            "reason": "energetic_failure_no_better_site",
            "window_net": window_net,
            "required": required,
            "attempted_breeding": self.attempted_breeding,
            "bred": self.bred,
            "final_state": self.final_state,
            "best_site_score": self.best_site_score,
            "center_score": self.current_center_score,
            "site_improvement_raw": site_improvement_raw,
            "site_improvement_effective": site_improvement_effective,
            "shift_distance_m": shift_distance,
            "loyalty": self.breeding_site_loyalty,
            "next_start_lon": self.next_start_lon,
            "next_start_lat": self.next_start_lat,
        }


# =========================================================
# SIMULATION CONFIG
# =========================================================
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


# =========================================================
# IBM
# =========================================================
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
                    if need <= 0:
                        break

                    give = int(round(need * donor_counts[d] / donor_total))
                    if give <= 0:
                        continue

                    rr, cc = np.where(lulc == d)
                    if len(rr) == 0:
                        continue

                    take_n = min(give, len(rr), need)
                    idx = np.random.choice(len(rr), size=take_n, replace=False)
                    lulc[rr[idx], cc[idx]] = cls
                    need -= take_n

        self.env.lulc.values[:] = lulc

        if self.scenario.fragmentation_multiplier != 1.0:
            for _, arr in self.env.frag.items():
                arr.values[:] = arr.values * self.scenario.fragmentation_multiplier

        self.env._init_food_tiled()

    def run(self, years: List[int], output_dir: str):
        ensure_dir(output_dir)
        for year in years:
            print(f"\n=== Simulating {year} ===")
            self._run_year(year, output_dir)

    def _safe_lulc_out(self, rec: Dict[str, float]):
        lulc_val = rec.get("lulc", np.nan)
        return int(lulc_val) if np.isfinite(lulc_val) else ""

    def _update_step_csv_final_state(self, yout_dir: str):
        for b in self.birds:
            fpath = os.path.join(yout_dir, f"{b.bird_id}.csv")
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue

            df_steps = pd.read_csv(fpath)
            df_steps["final_state"] = b.final_state
            df_steps.to_csv(fpath, index=False)

    def _run_year(self, year: int, output_dir: str):
        self.env._init_food_tiled()

        for b in self.birds:
            if b.alive:
                b.begin_new_year(self.env)

        yout_dir = os.path.join(output_dir, "steps", str(year))
        fig_dir = os.path.join(output_dir, "figures", str(year))
        sum_dir = os.path.join(output_dir, "summaries")
        ensure_dir(yout_dir)
        ensure_dir(fig_dir)
        ensure_dir(sum_dir)

        season_start = datetime(year, *self.cfg.season_start_mmdd)
        season_end = datetime(year, *self.cfg.season_end_mmdd, 23, 59)
        step = timedelta(minutes=self.cfg.step_minutes)
        t = season_start

        writers = {}
        for b in self.birds:
            if b.alive:
                fpath = os.path.join(yout_dir, f"{b.bird_id}.csv")
                writers[b.bird_id] = open(fpath, "w", encoding="utf-8")
                writers[b.bird_id].write(
                    "timestamp,step,lon,lat,dist_m,bearing,gain,cost,net,lulc,ndvi,ndwi,mode,site_score,attempted_breeding,bred\n"
                )

        steps_total = int(((season_end - season_start).total_seconds() // (self.cfg.step_minutes * 60)) + 1)
        pbar = tqdm(total=steps_total)
        last_day = season_start.date()

        while t <= season_end:
            if (t.hour < DAY_START_HOUR) or (t.hour > DAY_END_HOUR):
                t += step
                pbar.update(1)
                continue

            self.env.renew_food_if_due(t)

            if t.date() != last_day:
                for b in self.birds:
                    if b.alive and b.attempted_breeding:
                        b.reset_to_center()
                last_day = t.date()

            season_day = (t.date() - season_start.date()).days + 1

            for b in self.birds:
                if not b.alive:
                    continue

                rec = b.step(self.env, t, season_day)

                if rec and b.bird_id in writers:
                    ndvi_val = rec.get("ndvi", np.nan)
                    ndwi_val = rec.get("ndwi", np.nan)
                    site_score_val = rec.get("site_score", np.nan)

                    writers[b.bird_id].write(
                        f"{t.isoformat()},{b.step_count},"
                        f"{rec['lon']:.6f},{rec['lat']:.6f},{rec['dist_m']:.2f},{rec['bearing']:.2f},"
                        f"{rec['gain']:.4f},{rec['cost']:.4f},{rec['net']:.4f},"
                        f"{self._safe_lulc_out(rec)},"
                        f"{ndvi_val:.6f},{ndwi_val:.6f},"
                        f"{rec.get('mode', '')},{site_score_val:.6f},"
                        f"{int(b.attempted_breeding)},{int(b.bred)}\n"
                    )

            if t.hour == DAY_END_HOUR:
                for b in self.birds:
                    if not b.alive:
                        continue
                    _, _, _, died, _ = b.end_of_day_update()
                    if died:
                        b.alive = False

                if BREED_DECISION_START_DAY <= season_day <= BREED_DECISION_END_DAY:
                    for b in self.birds:
                        if not b.alive or b.attempted_breeding:
                            continue
                        if b.should_attempt_breeding():
                            b.attempted_breeding = True

            t += step
            pbar.update(1)

        pbar.close()

        for f in writers.values():
            f.close()

        year_shifts = []
        end_results = {}

        for b in self.birds:
            if b.alive:
                shift_result = b.end_of_season_update(year)
                end_results[b.bird_id] = shift_result

                if shift_result.get("shifted", False):
                    last_shift = b.shift_history[-1]
                    year_shifts.append(
                        {
                            "bird_id": b.bird_id,
                            "from_lon": last_shift["from_lon"],
                            "from_lat": last_shift["from_lat"],
                            "to_lon": last_shift["to_lon"],
                            "to_lat": last_shift["to_lat"],
                            "reason": shift_result["reason"],
                            "energy_cumulative": b.energy_cumulative,
                            "attempted_breeding": b.attempted_breeding,
                            "bred": b.bred,
                            "final_state": b.final_state,
                            "site_improvement_raw": shift_result.get("site_improvement_raw", np.nan),
                            "site_improvement_effective": shift_result.get("site_improvement_effective", np.nan),
                            "shift_distance_m": shift_result.get("shift_distance_m", np.nan),
                            "loyalty": shift_result.get("loyalty", np.nan),
                        }
                    )

        shifted_ids_this_year = {d["bird_id"] for d in year_shifts}

        for b in self.birds:
            if not b.alive and b.final_state == "unknown":
                b.final_state = "dead"

        self._update_step_csv_final_state(yout_dir)

        alive_birds = [b for b in self.birds if b.alive]
        successful_breeders = sum(1 for b in alive_birds if b.final_state == "successful_breeder")
        failed_breeders = sum(1 for b in alive_birds if b.final_state == "failed_breeder")
        non_breeders = sum(1 for b in alive_birds if b.final_state == "non_breeder")
        shifters_this_year = sum(1 for b in alive_birds if b.final_state == "shifter")

        print(f"\nYear {year} diagnostics:")
        print(f"  successful_breeders = {successful_breeders}")
        print(f"  failed_breeders = {failed_breeders}")
        print(f"  non_breeders = {non_breeders}")
        print(f"  shifters = {shifters_this_year}")
        print(f"  alive = {len(alive_birds)}")
        print(f"  shift_events_this_year = {len(year_shifts)}")

        if alive_birds:
            energies = [b.energy_cumulative for b in alive_birds]
            print(
                f"  energy_cumulative: min={np.min(energies):.3f}, "
                f"median={np.median(energies):.3f}, max={np.max(energies):.3f}"
            )

            window_nets = [
                res.get("window_net", np.nan)
                for _, res in end_results.items()
                if np.isfinite(res.get("window_net", np.nan))
            ]
            if window_nets:
                print(
                    f"  last_30d_net: min={np.min(window_nets):.3f}, "
                    f"median={np.median(window_nets):.3f}, max={np.max(window_nets):.3f}"
                )
                print(
                    f"  breeding success threshold = "
                    f"{BREED_MIN_EXCESS * ENERGY_SCALARS['baseline_daily_units'] * BREED_SUCCESS_WINDOW_DAYS:.3f}"
                )

        rows = []
        for b in self.birds:
            res = end_results.get(b.bird_id, {})
            final_window_net = res.get("window_net", np.nan)
            final_required = res.get("required", np.nan)

            rows.append(
                {
                    "bird_id": b.bird_id,
                    "final_state": b.final_state,
                    "alive": b.alive,
                    "attempted_breeding": b.attempted_breeding,
                    "bred": b.bred,
                    "final_30d_net": round(float(final_window_net), 3) if np.isfinite(final_window_net) else np.nan,
                    "breeding_threshold_required": round(float(final_required), 3) if np.isfinite(final_required) else np.nan,
                    "energy_cumulative": round(b.energy_cumulative, 3),
                    "step": b.step_count,
                    "home_range_shifted": b.home_range_shifted,
                    "shifted_this_year": b.bird_id in shifted_ids_this_year,
                    "failed_breeding_attempts": b.failed_breeding_attempts,
                    "breeding_site_loyalty": round(float(b.breeding_site_loyalty), 3),
                    "current_center_lon": round(b.current_center_lon, 6),
                    "current_center_lat": round(b.current_center_lat, 6),
                    "current_center_score": round(float(b.current_center_score), 4) if np.isfinite(b.current_center_score) else np.nan,
                    "best_site_lon": round(b.best_site_lon, 6),
                    "best_site_lat": round(b.best_site_lat, 6),
                    "best_site_score": round(float(b.best_site_score), 4) if np.isfinite(b.best_site_score) else np.nan,
                    "next_start_lon": round(float(getattr(b, "next_start_lon", np.nan)), 6),
                    "next_start_lat": round(float(getattr(b, "next_start_lat", np.nan)), 6),
                }
            )

        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(os.path.join(sum_dir, f"summary_{year}.csv"), index=False)

        if year_shifts:
            shift_df = pd.DataFrame(year_shifts)
            shift_df.to_csv(os.path.join(sum_dir, f"home_range_shifts_{year}.csv"), index=False)
            print(f"  {len(year_shifts)} birds shifted breeding center in {year}")

        self._plot_summary_graph(year, sum_dir, fig_dir)
        self._plot_breeder_tracks(year, yout_dir, fig_dir, self.birds)
        self._plot_nonbreeder_tracks(year, yout_dir, fig_dir, self.birds)
        self._plot_all_alive_tracks(year, yout_dir, fig_dir, self.birds)

        step_counts = [b.step_count for b in self.birds]
        print(
            f"  step_count: min={np.min(step_counts)}, "
            f"median={np.median(step_counts)}, max={np.max(step_counts)}"
        )

    def _plot_summary_graph(self, year: int, sum_dir: str, fig_dir: str):
        df = pd.read_csv(os.path.join(sum_dir, f"summary_{year}.csv"))

        alive = int((df["alive"] == True).sum())
        breeders = int((df["final_state"] == "successful_breeder").sum())
        shifters = int((df["final_state"] == "shifter").sum())

        plt.figure(figsize=(6, 4))
        plt.bar(
            ["Alive", "Breeders", "Shifters"],
            [alive, breeders, shifters]
        )
        plt.title(f"Year {year}: Population status")
        plt.ylabel("Number of birds")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"summary_{year}.png"), dpi=300)
        plt.close()

    def _plot_breeder_tracks(self, year: int, yout_dir: str, fig_dir: str, birds: List[BirdAgent]):
        plt.figure(figsize=(8, 8))
        n = 0

        for b in birds:
            if not (b.alive and b.final_state == "successful_breeder"):
                continue

            fpath = os.path.join(yout_dir, f"{b.bird_id}.csv")
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue

            df = pd.read_csv(fpath)
            if df.empty:
                continue

            plt.plot(df["lon"], df["lat"], color="green", alpha=0.6, linewidth=0.7)
            n += 1

        plt.title(f"Breeder tracks — {year} (n={n})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"tracks_breeders_{year}.png"), dpi=300)
        plt.close()

    def _plot_nonbreeder_tracks(self, year: int, yout_dir: str, fig_dir: str, birds: List[BirdAgent]):
        plt.figure(figsize=(8, 8))
        n = 0

        for b in birds:
            if not b.alive:
                continue

            if b.final_state == "successful_breeder":
                continue

            fpath = os.path.join(yout_dir, f"{b.bird_id}.csv")
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue

            df = pd.read_csv(fpath)
            if df.empty:
                continue

            plt.plot(df["lon"], df["lat"], color="blue", alpha=0.4, linewidth=0.7)
            n += 1

        plt.title(f"Non-breeder tracks — {year} (n={n})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"tracks_nonbreeders_{year}.png"), dpi=300)
        plt.close()

    def _plot_all_alive_tracks(self, year: int, yout_dir: str, fig_dir: str, birds: List[BirdAgent]):
        plt.figure(figsize=(8, 8))

        n_breed = 0
        n_non = 0

        for b in birds:
            if not b.alive:
                continue

            fpath = os.path.join(yout_dir, f"{b.bird_id}.csv")
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue

            df = pd.read_csv(fpath)
            if df.empty:
                continue

            if b.final_state == "successful_breeder":
                plt.plot(df["lon"], df["lat"], color="green", alpha=0.35, linewidth=0.6)
                n_breed += 1
            else:
                plt.plot(df["lon"], df["lat"], color="blue", alpha=0.25, linewidth=0.6)
                n_non += 1

        plt.title(f"All alive tracks — {year} (breeders={n_breed}, non-breeders={n_non})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"tracks_all_alive_{year}.png"), dpi=300)
        plt.close()


# =========================================================
# RUNNERS
# =========================================================
def run_simulation():
    env = Environment(LULC_PATH, NDVI_PATH, NDWI_PATH, FRAG_PATHS)
    birds = load_birds_from_points(POINTS_PATH, FIXED_RSF_PATH, RANDOM_RSF_PATH, STEP_MINUTES_DEFAULT)
    cfg = SimulationConfig()
    scenario = Scenario()
    ibm = IBM(env, birds, cfg, scenario)
    ibm.apply_scenario()
    ibm.run([2012], OUTPUT_DIR)


def run_simulation_multi(years: List[int]):
    print("Loading initial bird population...")
    current_birds = load_birds_from_points(
        POINTS_PATH,
        FIXED_RSF_PATH,
        RANDOM_RSF_PATH,
        STEP_MINUTES_DEFAULT,
    )

    for year in years:
        print(f"\nStarting simulation for {year}...")

        current_birds = [b for b in current_birds if b.alive]
        print(f"  {len(current_birds)} birds alive at start of {year}")

        if len(current_birds) == 0:
            print("  No birds alive - stopping simulation")
            break

        lulc = f"E:/IBM_stork_resources/LC_{year}.tif"
        ndvi = f"E:/IBM_stork_resources/NDVI_{year}_JulAug.tif"
        ndwi = f"E:/IBM_stork_resources/MNDWI_{year}_JulAug.tif"

        frag_paths_year = {
            "SHEI": f"E:/IBM_stork_resources/SHEI_{year}.tif",
            "ED": f"E:/IBM_stork_resources/ED_{year}.tif",
        }

        all_rasters = [lulc, ndvi, ndwi] + list(frag_paths_year.values())
        missing = [f for f in all_rasters if not os.path.exists(f)]

        if missing:
            for f in missing:
                print(f"Missing raster file: {f}")
            print("Skipping this year.\n")
            continue

        env = Environment(lulc, ndvi, ndwi, frag_paths_year)

        test_bird = current_birds[0]
        test_cov = env.sample_cell(test_bird.current_center_lon, test_bird.current_center_lat)
        print("Sample cov at first bird center:", test_cov)

        cfg = SimulationConfig(start_year=year, end_year=year)
        scenario = Scenario()
        ibm = IBM(env, current_birds, cfg, scenario)
        ibm.apply_scenario()

        out_dir_year = os.path.join(OUTPUT_DIR, str(year))
        ibm.run([year], out_dir_year)

        alive_count = sum(1 for b in current_birds if b.alive)
        successful_breeders = sum(1 for b in current_birds if b.alive and b.final_state == "successful_breeder")
        failed_breeders = sum(1 for b in current_birds if b.alive and b.final_state == "failed_breeder")
        non_breeders = sum(1 for b in current_birds if b.alive and b.final_state == "non_breeder")
        shifters = sum(1 for b in current_birds if b.alive and b.final_state == "shifter")

        print(f"Finished simulation for {year}. Results saved to {out_dir_year}")
        print(
            f"  End of {year} status: "
            f"{successful_breeders} successful breeders, "
            f"{failed_breeders} failed breeders, "
            f"{non_breeders} non-breeders, "
            f"{shifters} shifters, "
            f"{alive_count} alive"
        )


if __name__ == "__main__":
    run_simulation_multi([2000, 2001, 2012, 2018, 2022])