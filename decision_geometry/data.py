"""Remote NWB loading and conversion to a compact population tensor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import remfile
import requests
from pynwb import NWBHDF5IO

from .config import (
    ASSET_ID,
    DANDI_API,
    DANDISET_ID,
    DANDISET_VERSION,
    DEFAULT_BIN_SIZE,
    DEFAULT_MAX_UNITS,
    DEFAULT_WINDOW,
    SESSION_ID,
    SUBJECT_ID,
)


@dataclass(frozen=True)
class PopulationDataset:
    """Trial-aligned firing rates and labels from one recording session."""

    rates: np.ndarray
    time: np.ndarray
    choice: np.ndarray
    stimulus_side: np.ndarray
    prior_side: np.ndarray
    contrast: np.ndarray
    rewarded: np.ndarray
    reaction_time: np.ndarray
    trial_ids: np.ndarray
    unit_ids: np.ndarray
    unit_regions: np.ndarray
    subject_id: str = SUBJECT_ID
    session_id: str = SESSION_ID

    def __post_init__(self) -> None:
        if self.rates.ndim != 3:
            raise ValueError("rates must have shape trials x units x time")
        n_trials, n_units, n_bins = self.rates.shape
        if self.time.shape != (n_bins,):
            raise ValueError("time must match the last rates dimension")
        trial_fields = {
            "choice": self.choice,
            "stimulus_side": self.stimulus_side,
            "prior_side": self.prior_side,
            "contrast": self.contrast,
            "rewarded": self.rewarded,
            "reaction_time": self.reaction_time,
            "trial_ids": self.trial_ids,
        }
        for name, values in trial_fields.items():
            if values.shape != (n_trials,):
                raise ValueError(f"{name} must match the first rates dimension")
        if self.unit_regions.shape != (n_units,):
            raise ValueError("unit metadata must match the second rates dimension")
        if self.unit_ids.shape != (n_units,):
            raise ValueError("unit_ids must match the second rates dimension")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rates=self.rates,
            time=self.time,
            choice=self.choice,
            stimulus_side=self.stimulus_side,
            prior_side=self.prior_side,
            contrast=self.contrast,
            rewarded=self.rewarded,
            reaction_time=self.reaction_time,
            trial_ids=self.trial_ids,
            unit_ids=self.unit_ids,
            unit_regions=self.unit_regions,
            subject_id=np.array(self.subject_id),
            session_id=np.array(self.session_id),
        )

    @classmethod
    def from_cache(cls, path: Path) -> "PopulationDataset":
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: data[name] for name in data.files})


def _read_column(table, name: str, dtype=None) -> np.ndarray:
    values = np.asarray(table[name].data[:])
    return values.astype(dtype) if dtype is not None else values


def _resolve_download_url() -> str:
    response = requests.get(
        f"{DANDI_API}/assets/{ASSET_ID}/download/",
        allow_redirects=True,
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    return response.url


def select_unit_indices(
    quality: np.ndarray,
    presence_ratio: np.ndarray,
    firing_rate: np.ndarray,
    max_electrode: np.ndarray,
    n_electrodes: int,
    max_units: int | None,
) -> np.ndarray:
    """Select stable, well-isolated units using documented IBL metrics."""
    if max_units is not None and max_units < 1:
        raise ValueError("max_units must be at least one or None")
    mask = (
        (quality >= 1.0)
        & (presence_ratio >= 0.9)
        & (firing_rate >= 0.1)
        & (firing_rate <= 100.0)
        & (max_electrode >= 0)
        & (max_electrode < n_electrodes)
    )
    selected = np.flatnonzero(mask)
    if max_units is not None and selected.size > max_units:
        rank = np.lexsort((-presence_ratio[selected], -firing_rate[selected]))
        selected = selected[rank[:max_units]]
    return np.sort(selected)


def _stream_population(
    window: tuple[float, float],
    bin_size: float,
    max_units: int | None,
) -> PopulationDataset:
    url = _resolve_download_url()
    remote = remfile.File(url=url)
    h5_file = h5py.File(remote, mode="r")
    io = NWBHDF5IO(file=h5_file, load_namespaces=True)

    try:
        nwb = io.read()
        trials = nwb.trials
        units = nwb.units
        electrodes = nwb.electrodes

        onset = _read_column(trials, "gabor_stimulus_onset_time", float)
        movement = _read_column(trials, "wheel_movement_onset_time", float)
        raw_choice = _read_column(trials, "mouse_wheel_choice", str)
        raw_stimulus = _read_column(trials, "gabor_stimulus_side", str)
        probability_left = _read_column(trials, "probability_left", float)

        valid = (
            np.isfinite(onset)
            & np.isin(raw_choice, ["clockwise", "counter_clockwise"])
            & np.isin(raw_stimulus, ["left", "right"])
        )
        trial_ids = np.flatnonzero(valid)
        onset = onset[valid]

        choice = (raw_choice[valid] == "counter_clockwise").astype(np.int8)
        stimulus_side = (raw_stimulus[valid] == "right").astype(np.int8)
        prior_side = np.full(trial_ids.size, -1, dtype=np.int8)
        prior_side[probability_left[valid] == 0.2] = 1
        prior_side[probability_left[valid] == 0.8] = 0

        quality = _read_column(units, "ibl_quality_score", float)
        presence = _read_column(units, "presence_ratio", float)
        firing_rate = _read_column(units, "firing_rate", float)
        max_electrode = _read_column(units, "max_electrode", int)
        unit_ids = select_unit_indices(
            quality,
            presence,
            firing_rate,
            max_electrode,
            len(electrodes),
            max_units,
        )
        if unit_ids.size == 0:
            raise RuntimeError("No units passed the IBL quality filters")

        electrode_locations = _read_column(electrodes, "location", str)
        unit_regions = electrode_locations[max_electrode[unit_ids]]
        edges = np.arange(window[0], window[1] + bin_size * 0.5, bin_size)
        time = (edges[:-1] + edges[1:]) / 2
        absolute_edges = onset[:, None] + edges[None, :]
        rates = np.empty((trial_ids.size, unit_ids.size, time.size), dtype=np.float32)

        for column, unit_id in enumerate(unit_ids):
            spike_times = np.asarray(units.get_unit_spike_times(int(unit_id)))
            positions = np.searchsorted(spike_times, absolute_edges)
            rates[:, column, :] = np.diff(positions, axis=1) / bin_size

        return PopulationDataset(
            rates=rates,
            time=time.astype(np.float32),
            choice=choice,
            stimulus_side=stimulus_side,
            prior_side=prior_side,
            contrast=_read_column(trials, "gabor_stimulus_contrast", float)[valid].astype(np.float32),
            rewarded=_read_column(trials, "is_mouse_rewarded", bool)[valid],
            reaction_time=(movement[valid] - onset).astype(np.float32),
            trial_ids=trial_ids.astype(np.int32),
            unit_ids=unit_ids.astype(np.int32),
            unit_regions=unit_regions,
            subject_id=str(nwb.subject.subject_id),
            session_id=str(nwb.session_id),
        )
    finally:
        io.close()
        if h5_file.id.valid:
            h5_file.close()
        remote.close()


def load_population(
    cache_path: str | Path = "data/cache/session_population.npz",
    *,
    force_stream: bool = False,
    window: tuple[float, float] = DEFAULT_WINDOW,
    bin_size: float = DEFAULT_BIN_SIZE,
    max_units: int | None = DEFAULT_MAX_UNITS,
) -> PopulationDataset:
    """Load the compact cache or stream and derive it from the published NWB."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_stream:
        return PopulationDataset.from_cache(cache_path)

    dataset = _stream_population(window, bin_size, max_units)
    dataset.save(cache_path)
    return dataset


def provenance() -> dict[str, str]:
    return {
        "dandiset": DANDISET_ID,
        "version": DANDISET_VERSION,
        "asset_id": ASSET_ID,
        "subject_id": SUBJECT_ID,
        "session_id": SESSION_ID,
    }
