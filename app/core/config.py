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

    # --- Detection model ---
    model: str = "yolo11n.pt"  # any Ultralytics .pt; class names are introspected
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.50
    # Class names treated as humans (covers COCO and VisDrone-style models)
    human_classes: str = "person,pedestrian,people"
    # Class names flagged as threats (COCO has knife; swap model for weapons etc.)
    threat_classes: str = "knife"

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


settings = Settings()
