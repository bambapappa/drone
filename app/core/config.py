from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Drone Insats-PoC"
    debug: bool = False
    version: str = "0.1.0"

    # --- Video source ---
    # File path, RTSP/HTTP URL or camera index ("0"). Empty = pick first file in video_dir.
    source: str = ""
    video_dir: str = "videos"
    loop: bool = True  # restart file sources when they end
    # Crop frames to this region before any analysis: "x,y,w,h" normalized 0..1.
    # For split-screen IR/visual footage: select the visual half so people
    # aren't double-counted. Empty = full frame.
    analysis_roi: str = ""
    # Regions to EXCLUDE from analysis (";"-separated "x,y,w,h", normalized,
    # in the analyzed/cropped frame's coordinates). For picture-in-picture IR
    # insets: detections and smoke/fire analysis inside are discarded while
    # the video still shows the full frame. Ex: IGNORE_REGIONS=0.66,0,0.34,0.44
    ignore_regions: str = ""

    # --- Detection model ---
    model: str = "yolo11n.pt"  # any Ultralytics .pt; class names are introspected
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.50
    # Class names treated as humans (covers COCO and VisDrone-style models)
    human_classes: str = "person,pedestrian,people"
    # Threat flagging is deferred past PoC 1 (see DECISIONS B7): plumbing kept,
    # default off. Set e.g. "knife" with a COCO model to re-enable.
    threat_classes: str = ""

    # --- Output stream ---
    max_fps: float = 24.0
    out_width: int = 960
    jpeg_quality: int = 70

    # --- Box display smoothing (flow feed-forward + EMA + slew limit) ---
    smooth_tau_pos: float = 0.12  # s; position time constant
    smooth_tau_size: float = 0.18  # s; size time constant
    smooth_slew: float = 3.0  # box-dimensions per second max correction glide speed

    # --- Re-ID registry ---
    reid_sim_thresh: float = 0.86
    reid_max_gap_s: float = 60.0  # forget gallery entries older than this for matching
    reid_max_dist_frac: float = 0.45  # max stabilized travel (fraction of diag) per second gap

    # --- Behavior analysis (fixed thresholds => predictable on unseen footage) ---
    beh_window_s: float = 6.0
    beh_min_history_s: float = 3.0
    beh_still_speed: float = 0.10  # body-heights per second
    beh_still_time_s: float = 4.0
    beh_toward_speed: float = 0.25
    beh_toward_angle_deg: float = 40.0
    beh_toward_time_s: float = 1.5
    beh_prone_aspect: float = 1.4

    # --- Situation assessment (fire/smoke heuristics) ---
    hazard_min_area: float = 0.004  # fraction of frame area
    hazard_hold_s: float = 2.0
    smoke_flow_ema: float = 0.15
    base_margin: float = 0.08  # keep suggestion this far from frame edge
    base_hysteresis: float = 0.15  # move marker only if target shifts more than this

    # --- Threat alarm ---
    threat_hold_s: float = 2.0

    def human_class_set(self) -> set[str]:
        return {c.strip().lower() for c in self.human_classes.split(",") if c.strip()}

    def threat_class_set(self) -> set[str]:
        return {c.strip().lower() for c in self.threat_classes.split(",") if c.strip()}

    @staticmethod
    def _parse_rect(v: str, name: str, min_wh: float = 0.05) -> tuple[float, float, float, float]:
        parts = [float(p) for p in v.split(",")]
        if len(parts) != 4:
            raise ValueError(f"{name} ska vara 'x,y,w,h' (normaliserat 0..1)")
        x, y, w, h = parts
        if not (0 <= x < 1 and 0 <= y < 1 and min_wh <= w <= 1 and min_wh <= h <= 1):
            raise ValueError(f"{name} utanför giltigt intervall (0..1, w/h ≥ {min_wh})")
        if x + w > 1.0001 or y + h > 1.0001:
            raise ValueError(f"{name} sticker utanför bilden (x+w respektive y+h ≤ 1)")
        return x, y, w, h

    @field_validator("analysis_roi")
    @classmethod
    def _validate_roi(cls, v: str) -> str:
        if v.strip():
            cls._parse_rect(v, "ANALYSIS_ROI")
        return v

    @field_validator("ignore_regions")
    @classmethod
    def _validate_ignore(cls, v: str) -> str:
        for part in v.split(";"):
            if part.strip():
                cls._parse_rect(part, "IGNORE_REGIONS", min_wh=0.01)
        return v

    def roi_tuple(self) -> tuple[float, float, float, float] | None:
        if not self.analysis_roi.strip():
            return None
        return self._parse_rect(self.analysis_roi, "ANALYSIS_ROI")

    def ignore_list(self) -> list[tuple[float, float, float, float]]:
        return [
            self._parse_rect(p, "IGNORE_REGIONS", min_wh=0.01)
            for p in self.ignore_regions.split(";")
            if p.strip()
        ]


settings = Settings()
