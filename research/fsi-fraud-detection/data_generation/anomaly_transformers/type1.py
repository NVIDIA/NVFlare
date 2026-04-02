"""Anomaly Type 1 — Geo Location / Tower Location Mismatch.

Perturbs tower coordinates so they are far from the user's physical location.
Operates on a DataFrame slice in bulk using vectorised numpy operations.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Type1Config:
    """Configuration for anomalous tower perturbation.

    Attributes:
        nor_e_low:  Lower bound for North-or-East perturbation factor.
        nor_e_high: Upper bound for North-or-East perturbation factor.
        sor_w_low:  Lower bound for South-or-West perturbation factor.
        sor_w_high: Upper bound for South-or-West perturbation factor.
    """

    nor_e_low: float
    nor_e_high: float
    sor_w_low: float
    sor_w_high: float

    @classmethod
    def from_site_fields(cls, fields: dict) -> "Type1Config":
        nor_e = fields["anomalous_tower_NorE_perturbation"]["distributions"][0]
        sor_w = fields["anomalous_tower_SorW_perturbation"]["distributions"][0]
        return cls(
            nor_e_low=nor_e["low"],
            nor_e_high=nor_e["high"],
            sor_w_low=sor_w["low"],
            sor_w_high=sor_w["high"],
        )


def _generate_faraway_coords(
    coords: np.ndarray,
    max_coord: float,
    min_coord: float,
    nor_e_perturbations: np.ndarray,
    sor_w_perturbations: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised version of generate_faraway_coord for an array of coordinates."""
    n = len(coords)

    ne_delta = max_coord - coords + nor_e_perturbations
    sw_delta = np.abs(min_coord - coords) + sor_w_perturbations

    # Compute candidate coordinates for both directions
    ne_low = max_coord - ne_delta
    ne_candidates = rng.uniform(low=ne_low, high=max_coord)

    sw_high = sw_delta - np.abs(min_coord)
    sw_candidates = rng.uniform(low=min_coord, high=sw_high)

    # Zero out invalid ranges (delta <= 0)
    ne_valid = ne_delta > 0
    sw_valid = sw_delta > 0

    ne_candidates = np.where(ne_valid, ne_candidates, 0.0)
    sw_candidates = np.where(sw_valid, sw_candidates, 0.0)

    # Where both are valid, randomly pick one; otherwise take whichever is valid
    both_valid = ne_valid & sw_valid
    pick_ne = rng.integers(0, 2, size=n).astype(bool)

    result = np.where(
        both_valid,
        np.where(pick_ne, ne_candidates, sw_candidates),
        np.where(ne_valid, ne_candidates, sw_candidates),
    )
    return result


def apply(df: pd.DataFrame, config: Type1Config, seed: int = 42) -> pd.DataFrame:
    """Apply type-1 anomaly to the given DataFrame rows (in-place).

    Mutates tower lat/lon columns for both DEBITOR and CREDITOR so they are
    far from the corresponding physical geo-coordinates.

    Args:
        df:     DataFrame slice containing the rows to mutate.
        config: Perturbation parameters loaded from site config.
        seed:   RNG seed for reproducibility.

    Returns:
        The mutated DataFrame (same object, modified in-place).
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    for prefix in ("DEBITOR", "CREDITOR"):
        lat_col = f"{prefix}_TOWER_LATITUDE"
        lon_col = f"{prefix}_TOWER_LONGITUDE"

        nor_e_lat = rng.uniform(config.nor_e_low, config.nor_e_high, size=n)
        sor_w_lat = rng.uniform(config.sor_w_low, config.sor_w_high, size=n)
        nor_e_lon = rng.uniform(config.nor_e_low, config.nor_e_high, size=n)
        sor_w_lon = rng.uniform(config.sor_w_low, config.sor_w_high, size=n)

        df[lat_col] = _generate_faraway_coords(
            df[lat_col].to_numpy(dtype=float),
            90.0,
            -90.0,
            nor_e_lat,
            sor_w_lat,
            rng,
        )
        df[lon_col] = _generate_faraway_coords(
            df[lon_col].to_numpy(dtype=float),
            180.0,
            -180.0,
            nor_e_lon,
            sor_w_lon,
            rng,
        )

    df["TYPE1_ANOMALY"] = True
    return df
