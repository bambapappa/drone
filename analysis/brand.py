"""Persistent, place-bound fire incidents from fire and smoke observations.

Fire and smoke are detector signals, not separate operational incidents.  A
confirmed signal opens a BRAND incident at a place; later signals near that
place update the same incident.  Incidents remain active through the end of
the analyzed video because a temporary visual miss does not mean that the
physical fire stopped.

Recorded-film analysis repairs an ordinary adjacent-frame tracking dropout in
the scene layer before incidents are derived.  A remaining segment boundary
therefore means that visual continuity was not proved (for example a hard
cut); its unrelated coordinates stay isolated instead of inventing identity.
The GUI reviews the completed engine result and may correct it, but is not the
primary association mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BrandObservation:
    """One confirmed fire/smoke observation in a single coordinate space."""

    frame_no: int
    signal: str
    pos: tuple[float, float]
    area: float
    scene_segment: int | None
    # Raw frame-pixel position retained for replay rendering.  `pos` is the
    # scene coordinate used for association; it is not a stable screen point
    # while the drone moves.
    frame_pos: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.signal not in {"fire", "smoke"}:
            raise ValueError(f"unknown brand signal: {self.signal}")


@dataclass(frozen=True)
class BrandIncident:
    """One persistent fire site in a linked scene coordinate system."""

    brand_id: str
    scene_segment: int | None
    anchor: tuple[float, float]
    frame_start: int
    frame_end: int
    last_observed_frame: int
    signals: frozenset[str]
    observation_count: int
    observed_frame_count: int
    area_mean: float
    area_peak: float
    t_start: float
    t_end: float
    position_samples: tuple[tuple[int, float, float], ...]


@dataclass
class _MutableIncident:
    brand_id: str
    scene_segment: int | None
    anchor: tuple[float, float]
    frame_start: int
    last_observed_frame: int
    signals: set[str] = field(default_factory=set)
    areas: list[float] = field(default_factory=list)
    observation_count: int = 0
    observed_frames: set[int] = field(default_factory=set)
    position_samples: list[tuple[int, float, float]] = field(default_factory=list)

    def update(self, observation: BrandObservation) -> None:
        count = self.observation_count
        self.anchor = (
            (self.anchor[0] * count + observation.pos[0]) / (count + 1),
            (self.anchor[1] * count + observation.pos[1]) / (count + 1),
        )
        self.last_observed_frame = max(self.last_observed_frame, observation.frame_no)
        self.signals.add(observation.signal)
        self.areas.append(observation.area)
        self.observation_count += 1
        self.observed_frames.add(observation.frame_no)
        point = observation.frame_pos or observation.pos
        self.position_samples.append((observation.frame_no, point[0], point[1]))


class BrandIncidentTracker:
    """Deterministically associates observations with persistent fire sites."""

    def __init__(self, association_radius: float):
        if association_radius <= 0:
            raise ValueError("association_radius must be positive")
        self.association_radius = association_radius
        self._incidents: list[_MutableIncident] = []

    def observe(self, observations: list[BrandObservation]) -> None:
        """Associate every observation with at most one incident.

        Greedy nearest-neighbour assignment is deterministic because both
        observations and incidents have stable sort keys. An incident can
        consume at most one observation of each signal per frame, which makes
        simultaneous same-signal observations the honest evidence for a
        second site. One fire and one smoke observation may update the same
        incident because they are two sensors for the same operational BRAND.
        """
        claimed: set[tuple[str, str]] = set()
        ordered = sorted(
            observations,
            key=lambda item: (
                item.frame_no,
                item.pos[0],
                item.pos[1],
                item.signal,
                -item.area,
            ),
        )
        for observation in ordered:
            candidates = []
            for incident in self._incidents:
                if (
                    incident.brand_id,
                    observation.signal,
                ) in claimed or incident.scene_segment != observation.scene_segment:
                    continue
                distance = math.dist(incident.anchor, observation.pos)
                if distance <= self.association_radius:
                    candidates.append((distance, incident.brand_id, incident))
            if candidates:
                _, _, incident = min(candidates, key=lambda item: (item[0], item[1]))
            else:
                # A single strongest signal jumping in the frame is not proof
                # of a second physical site (the circling-drone film measured
                # large centroid/GMC drift). Keep the established incident.
                # A new site requires simultaneous same-signal evidence: the
                # existing incident has already consumed that signal this
                # frame, so it is absent from fallbacks and a new id is made.
                fallbacks = [
                    incident
                    for incident in self._incidents
                    if incident.scene_segment == observation.scene_segment
                    and (incident.brand_id, observation.signal) not in claimed
                ]
                if fallbacks:
                    incident = min(
                        fallbacks,
                        key=lambda item: (math.dist(item.anchor, observation.pos), item.brand_id),
                    )
                else:
                    incident = _MutableIncident(
                        brand_id=f"brand-{len(self._incidents):06d}",
                        scene_segment=observation.scene_segment,
                        anchor=observation.pos,
                        frame_start=observation.frame_no,
                        last_observed_frame=observation.frame_no,
                    )
                    self._incidents.append(incident)
            incident.update(observation)
            claimed.add((incident.brand_id, observation.signal))

    def danger_targets(self, frame_no: int, scene_segment: int | None) -> dict[str, tuple[float, float]]:
        """Return every incident active in this frame's coordinate system."""
        return {
            incident.brand_id: incident.anchor
            for incident in self._incidents
            if incident.frame_start <= frame_no and incident.scene_segment == scene_segment
        }

    def incidents(self, end_frame: int, fps: float) -> list[BrandIncident]:
        if fps <= 0:
            return []
        result = []
        for incident in self._incidents:
            result.append(
                BrandIncident(
                    brand_id=incident.brand_id,
                    scene_segment=incident.scene_segment,
                    anchor=incident.anchor,
                    frame_start=incident.frame_start,
                    frame_end=end_frame,
                    last_observed_frame=incident.last_observed_frame,
                    signals=frozenset(incident.signals),
                    observation_count=incident.observation_count,
                    observed_frame_count=len(incident.observed_frames),
                    area_mean=sum(incident.areas) / len(incident.areas),
                    area_peak=max(incident.areas),
                    t_start=incident.frame_start / fps,
                    t_end=(end_frame + 1) / fps,
                    position_samples=tuple(incident.position_samples),
                )
            )
        return result
