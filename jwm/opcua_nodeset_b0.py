"""Extract provenance-backed B0 records from official OPC UA NodeSets."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(node: ET.Element, local_name: str) -> str | None:
    for child in node.iter():
        if _local(child.tag) == local_name and child.text:
            return child.text.strip()
    return None


def _range(node: ET.Element) -> dict[str, float] | None:
    low, high = _child_text(node, "Low"), _child_text(node, "High")
    if low is None or high is None:
        return None
    try:
        low_f, high_f = float(low), float(high)
    except ValueError:
        return None
    return {"low": low_f, "high": high_f} if low_f < high_f else None


def extract_official_nodeset_items(nodeset_root: Path, start_index: int) -> list[dict[str, Any]]:
    """Return one item per analog variable with an explicit valued range."""
    records: list[dict[str, Any]] = []
    for path in sorted(nodeset_root.rglob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        variables = {
            node.attrib["NodeId"]: node
            for node in root
            if _local(node.tag) == "UAVariable" and "NodeId" in node.attrib
        }
        properties: dict[str, dict[str, ET.Element]] = {}
        for node in variables.values():
            browse = node.attrib.get("BrowseName", "").split(":")[-1]
            parent = node.attrib.get("ParentNodeId")
            if parent and browse in {"EURange", "InstrumentRange", "EngineeringUnits"}:
                properties.setdefault(parent, {})[browse] = node
        source_family = path.relative_to(nodeset_root).parts[0]
        provenance = str(path.relative_to(nodeset_root)).replace("\\", "/")
        for parent_id, props in properties.items():
            parent = variables.get(parent_id)
            if parent is None:
                continue
            eu_range = _range(props["EURange"]) if "EURange" in props else None
            instrument = (
                _range(props["InstrumentRange"])
                if "InstrumentRange" in props
                else None
            )
            if eu_range is None and instrument is None:
                continue
            unit = (
                _child_text(props["EngineeringUnits"], "Text")
                if "EngineeringUnits" in props
                else None
            )
            name = parent.attrib.get("BrowseName", parent_id).split(":")[-1]
            description = _child_text(parent, "Description")
            digest = hashlib.sha1(f"{provenance}:{parent_id}".encode()).hexdigest()[:12]
            records.append(
                {
                    "item_id": f"opc_nodeset:{digest}",
                    "source_dataset": f"OPCFoundation-{source_family}",
                    "split": "test",
                    "input": {
                        "tag_name": name,
                        "anonymized_tag_id": f"nodeset_tag_{start_index + len(records):05d}",
                        "declared_data_type": parent.attrib.get("DataType", "unknown"),
                        "representative_samples": [],
                        "documentation": description,
                    },
                    "ground_truth": {
                        "data_type": parent.attrib.get("DataType", "unknown"),
                        "engineering_unit": unit,
                        "instrument_range": instrument,
                        "eu_range": eu_range,
                        "role": "sensor_feedback",
                        "relationships": [
                            {
                                "relation": "component_of",
                                "target_tag_id": parent.attrib.get("ParentNodeId", "unknown"),
                            }
                        ],
                    },
                    "provenance": {
                        "schema_and_type": provenance,
                        "role": "official OPC UA Companion Specification analog variable",
                        "unit": provenance if unit else None,
                        "range": provenance,
                        "relationship": provenance,
                    },
                }
            )
    return records
