"""Collect B0 metadata and sample values from an OPC UA server."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from asyncua import Client, ua


PROPERTY_NAMES = {"EngineeringUnits", "InstrumentRange", "EURange"}


def plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {key: plain(item) for key, item in asdict(value).items()}
    if hasattr(value, "__dict__"):
        return {
            key: plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


async def collect(args: argparse.Namespace) -> dict[str, Any]:
    client = Client(args.endpoint)
    client.secure_channel_timeout = args.timeout * 1000
    await client.connect()
    try:
        roots = (
            [client.get_node(node_id) for node_id in args.root]
            if args.root
            else [client.nodes.objects]
        )
        queue = [(root, 0) for root in roots]
        visited: set[str] = set()
        variables = []
        while queue and len(visited) < args.max_nodes:
            node, depth = queue.pop(0)
            node_id = str(node.nodeid)
            if node_id in visited:
                continue
            visited.add(node_id)
            try:
                node_class = await node.read_node_class()
                browse_name = await node.read_browse_name()
            except Exception:
                continue
            if node_class == ua.NodeClass.Variable:
                properties = {}
                for prop in await node.get_properties():
                    try:
                        name = (await prop.read_browse_name()).Name
                        if name in PROPERTY_NAMES:
                            properties[name] = plain(await prop.read_value())
                    except Exception:
                        continue
                samples = []
                for _ in range(args.samples):
                    try:
                        samples.append(plain(await node.read_value()))
                    except Exception:
                        samples.append(None)
                    if args.interval:
                        await asyncio.sleep(args.interval)
                try:
                    data_type = (await node.read_data_type_as_variant_type()).name
                except Exception:
                    data_type = "Unknown"
                try:
                    description = plain(await node.read_description())
                except Exception:
                    description = None
                variables.append(
                    {
                        "node_id": node_id,
                        "browse_name": browse_name.Name,
                        "data_type": data_type,
                        "description": description,
                        "engineering_unit": properties.get("EngineeringUnits"),
                        "instrument_range": properties.get("InstrumentRange"),
                        "eu_range": properties.get("EURange"),
                        "representative_samples": samples,
                    }
                )
            if depth < args.max_depth:
                try:
                    queue.extend((child, depth + 1) for child in await node.get_children())
                except Exception:
                    pass
        return {
            "schema_version": "factorytraj-b0-opcua-probe-0.1.0",
            "endpoint": args.endpoint,
            "roots": args.root,
            "variables": variables,
            "coverage": {
                "engineering_unit": sum(
                    value["engineering_unit"] is not None for value in variables
                )
                / max(len(variables), 1),
                "instrument_or_eu_range": sum(
                    value["instrument_range"] is not None
                    or value["eu_range"] is not None
                    for value in variables
                )
                / max(len(variables), 1),
            },
        }
    finally:
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()
    payload = asyncio.run(asyncio.wait_for(collect(args), args.timeout))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"variables={len(payload['variables'])} coverage={payload['coverage']} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
