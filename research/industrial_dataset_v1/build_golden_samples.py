"""Build ten illustrative IWM-Episodes v1 golden records.

The records validate the schema and coverage design. They reference fictitious URIs
and must not be presented as captured factory evidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "golden_samples_v1.json"
TZ = timezone(timedelta(hours=7))


def stream(stream_id, uri, modality, rate, clock_error, **extra):
    record = {
        "stream_id": stream_id,
        "uri": uri,
        "modality": modality,
        "sampling_rate_hz": rate,
        "clock_error_ms": clock_error,
        "validity_uri": None,
        "unit": None,
        "sensor_type": None,
        "view": None,
        "fps": None,
        "channels": None,
    }
    record.update(extra)
    return record


def make_episode(
    idx,
    asset_type,
    asset_id,
    duration,
    mode,
    load,
    telemetry,
    event,
    severity,
    action,
    actor,
    outcome,
    future_state,
    *,
    audio=True,
    spatial=None,
    quality="na",
    downtime=0.0,
    recovery=None,
    ood_tags=None,
    alarms=None,
    product_id=None,
):
    start = datetime(2026, 7, 23, 1, 0, tzinfo=timezone.utc) + timedelta(minutes=idx * 5)
    end = start + timedelta(seconds=duration)
    prefix = f"golden/{idx:04d}"
    telemetry_streams = [
        stream(
            f"{asset_id}.{sensor_id}",
            f"iwm://{prefix}/telemetry/{sensor_id}.parquet",
            "telemetry" if modality == "telemetry" else modality,
            rate,
            15.0,
            unit=unit,
            sensor_type=sensor_type,
        )
        for sensor_id, sensor_type, unit, rate, modality in telemetry
    ]
    video = [
        stream(
            f"{asset_id}.cam_side",
            f"iwm://{prefix}/video/side.mp4",
            "rgb",
            30.0,
            8.0,
            view="side",
            fps=30.0,
            channels=3,
        )
    ]
    audio_streams = []
    if audio:
        audio_streams.append(
            stream(
                f"{asset_id}.mic_machine",
                f"iwm://{prefix}/audio/machine.wav",
                "audio",
                48000.0,
                6.0,
                channels=1,
            )
        )
    event_end = duration * 0.8 if event != "STATE.NORMAL_OPERATION" else duration
    action_start = duration * 0.55
    if action == "ACTION.NO_ACTION":
        action_start = 0.0
    return {
        "schema_version": "1.0.0",
        "episode_id": f"IWM-GOLD-{idx:04d}",
        "source_id": f"illustrative-source-{idx:04d}",
        "site_id": f"demo_site_{1 + idx % 2}",
        "line_id": f"demo_line_{1 + idx % 3}",
        "asset_id": asset_id,
        "asset_type": asset_type,
        "product_id": product_id,
        "batch_id": f"demo_batch_{idx // 3:02d}" if product_id else None,
        "time": {
            "start_time": start.isoformat().replace("+00:00", "Z"),
            "end_time": end.isoformat().replace("+00:00", "Z"),
            "duration_s": duration,
            "original_timezone": "Asia/Ho_Chi_Minh",
            "clock_source": "ptp",
            "max_clock_error_ms": 15.0,
        },
        "observations": {
            "video": video,
            "audio": audio_streams,
            "telemetry": telemetry_streams,
            "other": [],
        },
        "machine_state": {
            "operating_mode": mode,
            "load_pct": load,
            "setpoints": {"nominal_load_pct": load},
            "alarms": alarms or [],
        },
        "actions": [
            {
                "action_id": f"ACT-{idx:04d}",
                "type": action,
                "actor": actor,
                "start_offset_s": action_start,
                "end_offset_s": min(duration, action_start + (0.1 if action != "ACTION.NO_ACTION" else 0.0)),
                "parameters": {},
                "safety_authorized": True,
                "executed": action != "ACTION.NO_ACTION",
            }
        ],
        "events": [
            {
                "event_id": f"EVT-{idx:04d}",
                "taxonomy_id": event,
                "start_offset_s": 0.0 if event == "STATE.NORMAL_OPERATION" else duration * 0.1,
                "end_offset_s": event_end,
                "severity": severity,
                "confidence": 0.98 if event == "STATE.NORMAL_OPERATION" else 0.92,
                "label_source": "multi_source",
                "reviewed": True,
                "root_cause_status": "confirmed" if severity >= 2 else "unknown",
                "spatial": spatial,
            }
        ],
        "outcome": {
            "status": outcome,
            "quality_disposition": quality,
            "downtime_s": downtime,
            "recovery_time_s": recovery,
            "future_state": future_state,
        },
        "data_quality": {
            "missing_fraction": 0.0,
            "drift_flags": [],
            "status": "pass",
            "validator_version": "1.0.0",
        },
        "governance": {
            "data_owner": "IWM-Episodes illustrative examples",
            "source_type": "synthetic",
            "license": "CC-BY-4.0",
            "permitted_use": "public",
            "redistribution_allowed": True,
            "privacy_review": "pass",
            "redactions": [],
            "provenance": ["programmatic golden sample", "no real factory media"],
        },
        "split": {
            "split": ["train", "validation", "test_public", "test_private"][idx % 4],
            "split_group": f"demo_site_{1 + idx % 2}/{asset_id}/day_{idx:02d}",
            "ood_tags": ood_tags or [],
        },
        "world_model_targets": {
            "reasoner": ["event_classification", "state_explanation"],
            "generator": ["future_state_prediction"],
            "action": ["inverse_dynamics"] if action != "ACTION.NO_ACTION" else ["forward_dynamics"],
        },
    }


SAMPLES = [
    make_episode(
        1, "centrifugal_pump", "pump_07", 10.0, "running", 82.0,
        [("pressure", "pressure", "bar", 100.0, "telemetry"), ("current", "motor_current", "A", 1000.0, "telemetry"), ("vibration", "vibration", "mm/s", 25600.0, "vibration")],
        "FAULT.MECHANICAL.CAVITATION", 2, "ACTION.SETPOINT_CHANGE", "operator", "recovered",
        {"vibration_rms_mm_s": 4.2, "pressure_bar": 3.3}, recovery=8.5,
    ),
    make_episode(
        2, "induction_motor", "motor_12", 12.0, "running", 70.0,
        [("speed", "rotational_speed", "rpm", 100.0, "telemetry"), ("current", "motor_current", "A", 1000.0, "telemetry"), ("vibration", "vibration", "m/s2", 25600.0, "vibration")],
        "FAULT.MECHANICAL.BEARING_IMBALANCE", 2, "ACTION.MAINTENANCE_INSPECTION", "maintenance", "degraded_continuing",
        {"inspection_required": True, "speed_rpm": 1450.0}, downtime=0.0,
    ),
    make_episode(
        3, "belt_conveyor", "conveyor_03", 8.0, "running", 64.0,
        [("speed", "belt_speed", "m/s", 100.0, "telemetry"), ("torque", "drive_torque", "Nm", 1000.0, "telemetry"), ("photoeye", "presence", "bool", 100.0, "telemetry")],
        "FAULT.MECHANICAL.JAM", 3, "ACTION.STOP", "plc", "stopped",
        {"belt_speed_m_s": 0.0, "jam_cleared": False}, downtime=44.0,
        spatial={"camera_id": "conveyor_03.cam_side", "boxes_xyxy": [[410, 250, 730, 610]], "mask_uri": None},
        alarms=[{"code": "CONV_TORQUE_HIGH", "start_offset_s": 1.2, "end_offset_s": 8.0}],
    ),
    make_episode(
        4, "inspection_station", "vision_cell_02", 3.0, "inspection", 35.0,
        [("conveyor_speed", "belt_speed", "m/s", 100.0, "telemetry"), ("light", "illumination", "lux", 10.0, "telemetry"), ("trigger", "camera_trigger", "bool", 100.0, "telemetry")],
        "QUALITY.SURFACE_CRACK", 2, "ACTION.REJECT_PRODUCT", "plc", "quality_fail",
        {"reject_gate": "closed", "product_removed": True}, product_id="cast_part_0042", quality="scrap",
        spatial={"camera_id": "vision_cell_02.cam_side", "boxes_xyxy": [[612, 318, 704, 379]], "mask_uri": "iwm://golden/0004/labels/crack.png"},
    ),
    make_episode(
        5, "robot_assembly_cell", "robot_cell_01", 15.0, "automatic", 55.0,
        [("joint_state", "robot_joint_state", "rad", 250.0, "telemetry"), ("force_torque", "force_torque", "N_Nm", 1000.0, "force_torque"), ("gripper", "gripper_width", "mm", 250.0, "telemetry")],
        "FAULT.MECHANICAL.MISALIGNMENT", 2, "ACTION.ROBOT_CORRECTION", "plc", "recovered",
        {"insertion_depth_mm": 22.0, "assembly_complete": True}, recovery=5.0,
    ),
    make_episode(
        6, "pneumatic_valve", "valve_18", 10.0, "open", 76.0,
        [("upstream_pressure", "pressure", "bar", 100.0, "telemetry"), ("downstream_pressure", "pressure", "bar", 100.0, "telemetry"), ("flow", "flow_rate", "L/min", 100.0, "telemetry")],
        "FAULT.FLUID.LEAKAGE", 2, "ACTION.SETPOINT_CHANGE", "operator", "degraded_continuing",
        {"valve_open_pct": 45.0, "leak_suspected": True},
    ),
    make_episode(
        7, "electric_motor", "motor_21", 20.0, "running", 94.0,
        [("temperature", "winding_temperature", "degC", 10.0, "telemetry"), ("current", "motor_current", "A", 1000.0, "telemetry"), ("speed", "rotational_speed", "rpm", 100.0, "telemetry")],
        "FAULT.THERMAL.OVERHEAT", 3, "ACTION.STOP", "safety", "stopped",
        {"motor_running": False, "temperature_deg_c": 91.0}, downtime=180.0,
        alarms=[{"code": "MOTOR_TEMP_HH", "start_offset_s": 3.0, "end_offset_s": 20.0}],
    ),
    make_episode(
        8, "air_compressor", "compressor_04", 14.0, "loaded", 88.0,
        [("pressure", "discharge_pressure", "bar", 100.0, "telemetry"), ("flow", "air_flow", "m3/min", 100.0, "telemetry"), ("power", "electrical_power", "kW", 1000.0, "telemetry")],
        "FAULT.PROCESS.PRESSURE_DROP", 2, "ACTION.MAINTENANCE_INSPECTION", "maintenance", "degraded_continuing",
        {"pressure_bar": 5.8, "inspection_required": True},
    ),
    make_episode(
        9, "forklift_zone", "forklift_zone_02", 6.0, "active", 40.0,
        [("forklift_speed", "vehicle_speed", "m/s", 50.0, "telemetry"), ("zone_occupancy", "occupancy", "count", 30.0, "telemetry"), ("distance", "minimum_distance", "m", 30.0, "telemetry")],
        "SAFETY.NEAR_MISS", 3, "ACTION.EMERGENCY_STOP", "safety", "stopped",
        {"minimum_distance_m": 0.7, "collision": False}, downtime=12.0,
        spatial={"camera_id": "forklift_zone_02.cam_side", "boxes_xyxy": [[180, 190, 520, 700], [690, 160, 1040, 720]], "mask_uri": None},
    ),
    make_episode(
        10, "centrifugal_pump", "pump_11", 10.0, "running", 60.0,
        [("pressure", "pressure", "bar", 100.0, "telemetry"), ("current", "motor_current", "A", 1000.0, "telemetry"), ("vibration", "vibration", "mm/s", 25600.0, "vibration")],
        "STATE.NORMAL_OPERATION", 0, "ACTION.NO_ACTION", "source", "normal_completion",
        {"vibration_rms_mm_s": 1.1, "pressure_bar": 3.5}, ood_tags=["unseen_lighting"], quality="na",
    ),
]

payload = {
    "dataset": "IWM-Episodes v1 golden samples",
    "notice": "Illustrative schema-validation records only; all URIs and factory identities are fictitious.",
    "schema_version": "1.0.0",
    "samples": SAMPLES,
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {len(SAMPLES)} samples -> {OUT}")

