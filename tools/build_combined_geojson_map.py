#!/usr/bin/env python3
# ruff: noqa: E501
"""Build a compact all-set GeoJSON and an interactive MapLibre overview."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>NC 2018 D11 — Combined Rut Severity Map</title>
  <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@^5.12.0/dist/maplibre-gl.css">
  <script src="https://unpkg.com/maplibre-gl@^5.12.0/dist/maplibre-gl.js"></script>
  <style>
    html, body, #map { height: 100%; margin: 0; }
    body { font-family: Inter, Aptos, Arial, sans-serif; color: #152535; }
    #panel {
      position: absolute; z-index: 5; left: 16px; top: 16px; width: 290px;
      background: rgba(255,255,255,.96); border-radius: 12px; padding: 16px 17px;
      box-shadow: 0 5px 22px rgba(15,31,48,.22); backdrop-filter: blur(5px);
    }
    h1 { margin: 0 0 4px; font-size: 18px; line-height: 1.25; }
    .subtitle { color: #627483; font-size: 12px; margin-bottom: 13px; }
    .metric { display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; }
    .metric b { font-variant-numeric: tabular-nums; }
    .divider { height: 1px; background: #dde5ea; margin: 11px 0; }
    .layer { display: flex; align-items: center; gap: 8px; font-size: 12px; margin: 7px 0; }
    .layer input { margin: 0; }
    .swatch { width: 12px; height: 12px; border-radius: 50%; flex: 0 0 12px; }
    .boundary-swatch { width: 14px; height: 10px; border: 2px solid #0f747a; background: rgba(27,166,166,.08); flex: 0 0 14px; }
    .count { color: #647582; margin-left: auto; font-variant-numeric: tabular-nums; }
    .hint { font-size: 10.5px; color: #647582; line-height: 1.4; margin-top: 10px; }
    #status { font-size: 11px; color: #0f747a; font-weight: 700; margin-top: 9px; }
    #file-warning {
      display: none; position: absolute; z-index: 10; inset: 0; background: rgba(15,31,48,.92);
      color: white; align-items: center; justify-content: center; text-align: center; padding: 30px;
    }
    #file-warning code { color: #8fe0dc; }
    .maplibregl-popup-content { border-radius: 9px; padding: 12px 14px; min-width: 235px; }
    .popup-title { font-weight: 800; margin-bottom: 7px; }
    .popup-grid { display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; font-size: 12px; }
    .popup-grid span:nth-child(odd) { color: #687a88; }
    .popup-grid span:nth-child(even) { text-align: right; font-variant-numeric: tabular-nums; }
    .popup-link { display: inline-block; margin-top: 9px; color: #0f747a; font-size: 12px; font-weight: 700; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="panel">
    <h1>NC 2018 D11 Rut Map</h1>
    <div class="subtitle">All survey sets • severity-colored 3DC records</div>
    <div class="metric"><span>Survey sets</span><b>__SETS__</b></div>
    <div class="metric"><span>Total records</span><b>__TOTAL__</b></div>
    <div class="metric"><span>Overview display points</span><b>__DISPLAYED__</b></div>
    <div class="divider"></div>
    <label class="layer"><input id="boundary-toggle" type="checkbox" checked><span class="boundary-swatch"></span><span>North Carolina boundary</span></label>
    <div class="divider"></div>
    <label class="layer"><input type="checkbox" data-layer="severity-0" checked><span class="swatch" style="background:#718096"></span><span>Severity 0</span><span class="count">__S0__</span></label>
    <label class="layer"><input type="checkbox" data-layer="severity-1" checked><span class="swatch" style="background:#f1c84b"></span><span>Severity 1</span><span class="count">__S1__</span></label>
    <label class="layer"><input type="checkbox" data-layer="severity-2" checked><span class="swatch" style="background:#ef8b36"></span><span>Severity 2 · moderate</span><span class="count">__S2__</span></label>
    <label class="layer"><input type="checkbox" data-layer="severity-3" checked><span class="swatch" style="background:#cd4747"></span><span>Severity 3 · high</span><span class="count">__S3__</span></label>
    <div id="status">Loading combined GeoJSON…</div>
    <div class="hint">Click a point for file-level rut results. The overview contains every Severity 2/3 point, every fifth Severity 1 point, and every tenth Severity 0 point. The separate combined GeoJSON retains all records. Boundary: U.S. Census Bureau TIGERweb, 2025 vintage. A local web server and internet connection are required.</div>
  </div>
  <div id="file-warning">
    <div><h2>Open through a local web server</h2><p>From the map folder, run:<br><code>python3 -m http.server 8000</code><br><br>Then open <code>http://localhost:8000/all_sets_rut_map.html</code>.</p></div>
  </div>
  <script>
    if (location.protocol === 'file:') document.getElementById('file-warning').style.display = 'flex';

    const map = new maplibregl.Map({
      container: 'map',
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            minzoom: 0,
            maxzoom: 19,
            attribution: '© OpenStreetMap contributors'
          }
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }]
      },
      bounds: __BOUNDS__,
      fitBoundsOptions: { padding: { top: 35, bottom: 35, left: 335, right: 35 } },
      maxZoom: 19
    });
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'imperial' }), 'bottom-right');

    const styles = {
      0: { color: '#718096', radius: [2.0, 4.0], opacity: 0.42 },
      1: { color: '#f1c84b', radius: [2.3, 5.0], opacity: 0.72 },
      2: { color: '#ef8b36', radius: [3.2, 7.0], opacity: 0.90 },
      3: { color: '#cd4747', radius: [4.0, 8.5], opacity: 0.95 }
    };

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
    }
    function number(value, digits=3) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : 'N/A';
    }
    function popupHtml(p) {
      const link = p.preview ? `<a class="popup-link" href="${escapeHtml(p.preview)}" target="_blank">Open grayscale preview</a>` : '';
      return `<div class="popup-title">${escapeHtml(p.file_name)}</div><div class="popup-grid">
        <span>Set</span><span>${escapeHtml(p.set)}</span>
        <span>Frame</span><span>${escapeHtml(p.frame)}</span>
        <span>Severity</span><span>${escapeHtml(p.severity)}</span>
        <span>Left rut</span><span>${number(p.left_rut)} in</span>
        <span>Right rut</span><span>${number(p.right_rut)} in</span>
        <span>Average rut</span><span>${number(p.average_rut)} in</span>
        <span>Cross slope</span><span>${number(p.cross_slope, 2)}%</span>
      </div>${link}`;
    }

    map.on('load', () => {
      map.addSource('nc-boundary', {
        type: 'geojson',
        data: 'north_carolina_boundary.geojson'
      });
      map.addLayer({
        id: 'nc-boundary-fill',
        type: 'fill',
        source: 'nc-boundary',
        paint: {
          'fill-color': '#1ba6a6',
          'fill-opacity': 0.035
        }
      });
      map.addLayer({
        id: 'nc-boundary-line',
        type: 'line',
        source: 'nc-boundary',
        paint: {
          'line-color': '#0f747a',
          'line-width': ['interpolate', ['linear'], ['zoom'], 5, 2.3, 12, 4.5],
          'line-opacity': 0.95
        }
      });
      map.addSource('rut-points', {
        type: 'geojson',
        data: 'all_sets_rut_map_points.geojson',
        generateId: true
      });
      for (const severity of [0, 1, 2, 3]) {
        const style = styles[severity];
        const id = `severity-${severity}`;
        map.addLayer({
          id,
          type: 'circle',
          source: 'rut-points',
          filter: ['==', ['get', 'severity'], severity],
          paint: {
            'circle-color': style.color,
            'circle-opacity': style.opacity,
            'circle-radius': ['interpolate', ['linear'], ['zoom'], 5, style.radius[0], 15, style.radius[1]],
            'circle-stroke-color': severity >= 2 ? '#ffffff' : style.color,
            'circle-stroke-width': severity >= 2 ? 0.7 : 0
          }
        });
        map.on('click', id, event => {
          const feature = event.features && event.features[0];
          if (!feature) return;
          new maplibregl.Popup({ maxWidth: '330px' })
            .setLngLat(feature.geometry.coordinates)
            .setHTML(popupHtml(feature.properties))
            .addTo(map);
        });
        map.on('mouseenter', id, () => { map.getCanvas().style.cursor = 'pointer'; });
        map.on('mouseleave', id, () => { map.getCanvas().style.cursor = ''; });
      }
      document.querySelectorAll('input[data-layer]').forEach(input => {
        input.addEventListener('change', event => {
          map.setLayoutProperty(event.target.dataset.layer, 'visibility', event.target.checked ? 'visible' : 'none');
        });
      });
      document.getElementById('boundary-toggle').addEventListener('change', event => {
        const visibility = event.target.checked ? 'visible' : 'none';
        map.setLayoutProperty('nc-boundary-fill', 'visibility', visibility);
        map.setLayoutProperty('nc-boundary-line', 'visibility', visibility);
      });
    });
    map.on('sourcedata', event => {
      if (event.sourceId === 'rut-points' && event.isSourceLoaded) {
        document.getElementById('status').textContent = 'All records loaded';
      }
    });
    map.on('error', event => {
      if (event.error) document.getElementById('status').textContent = `Map load warning: ${event.error.message}`;
    });
  </script>
</body>
</html>
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/NC_2018_D11"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/NC_2018_D11/map"))
    return parser


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _coordinate_pairs(value: Any):
    if not isinstance(value, list):
        return
    if len(value) >= 2 and _number(value[0]) is not None and _number(value[1]) is not None:
        yield float(value[0]), float(value[1])
        return
    for item in value:
        yield from _coordinate_pairs(item)


def _preview_link(value: Any, *, input_root: Path, output_dir: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    preview = Path(value)
    try:
        relative = preview.resolve().relative_to(input_root.resolve())
    except (OSError, ValueError):
        return value
    return Path(os.path.relpath(input_root / relative, output_dir)).as_posix()


def _write_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(input_root: Path, output_dir: Path) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    sources = sorted(
        input_root.glob("set_*/set_*_rut_results.geojson"),
        key=lambda path: int(path.parent.name.removeprefix("set_")),
    )
    if not sources:
        raise FileNotFoundError(f"No set GeoJSON files found under {input_root}")

    geojson_path = output_dir / "all_sets_rut_points.geojson"
    map_geojson_path = output_dir / "all_sets_rut_map_points.geojson"
    html_path = output_dir / "all_sets_rut_map.html"
    manifest_path = output_dir / "all_sets_rut_map_manifest.json"
    boundary_path = output_dir / "north_carolina_boundary.geojson"
    if not boundary_path.is_file():
        raise FileNotFoundError(
            f"North Carolina boundary GeoJSON is missing: {boundary_path}. "
            "Download Census TIGERweb state 37 before building the map."
        )
    severity_counts: Counter[int] = Counter()
    records_total = 0
    mapped = 0
    west = south = float("inf")
    east = north = float("-inf")

    def write_geojson(temporary: Path) -> None:
        nonlocal records_total, mapped, west, south, east, north
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write('{"type":"FeatureCollection","features":[')
            first = True
            for source in sources:
                payload = json.loads(source.read_text(encoding="utf-8"))
                for feature in payload.get("features", []):
                    records_total += 1
                    properties = feature.get("properties") or {}
                    severity = int(properties.get("severity", -1))
                    severity_counts[severity] += 1
                    geometry = feature.get("geometry")
                    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
                    valid_point = (
                        isinstance(coordinates, list)
                        and len(coordinates) >= 2
                        and _number(coordinates[0]) is not None
                        and _number(coordinates[1]) is not None
                    )
                    if valid_point:
                        longitude = float(coordinates[0])
                        latitude = float(coordinates[1])
                        west = min(west, longitude)
                        east = max(east, longitude)
                        south = min(south, latitude)
                        north = max(north, latitude)
                        mapped += 1
                        compact_geometry: dict[str, Any] | None = {
                            "type": "Point",
                            "coordinates": [longitude, latitude],
                        }
                    else:
                        compact_geometry = None
                    compact = {
                        "type": "Feature",
                        "geometry": compact_geometry,
                        "properties": {
                            "set": str(properties.get("set", source.parent.name.removeprefix("set_"))),
                            "file_name": properties.get("file_name"),
                            "frame": properties.get("starting_frame_number"),
                            "left_rut": _number(properties.get("averaged_left_rut")),
                            "right_rut": _number(properties.get("averaged_right_rut")),
                            "average_rut": _number(properties.get("averaged_rut")),
                            "cross_slope": _number(properties.get("cross_slope_average_percent")),
                            "severity": severity,
                            "preview": _preview_link(
                                properties.get("preview_png"),
                                input_root=input_root,
                                output_dir=output_dir,
                            ),
                        },
                    }
                    if not first:
                        stream.write(",")
                    json.dump(compact, stream, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
                    first = False
            stream.write("]}\n")

    _write_atomic(geojson_path, write_geojson)
    if mapped == 0:
        raise ValueError("No valid point coordinates were found")

    display_counts: Counter[int] = Counter()

    def write_map_geojson(temporary: Path) -> None:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        seen: Counter[int] = Counter()
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write('{"type":"FeatureCollection","features":[')
            first = True
            for feature in payload["features"]:
                severity = int(feature["properties"]["severity"])
                seen[severity] += 1
                keep = (
                    severity >= 2
                    or (severity == 1 and seen[severity] % 5 == 1)
                    or (severity == 0 and seen[severity] % 10 == 1)
                )
                if not keep:
                    continue
                display_counts[severity] += 1
                if not first:
                    stream.write(",")
                json.dump(feature, stream, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
                first = False
            stream.write("]}\n")

    _write_atomic(map_geojson_path, write_map_geojson)
    displayed = sum(display_counts.values())
    boundary_payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary_pairs = [
        pair
        for feature in boundary_payload.get("features", [])
        for pair in _coordinate_pairs((feature.get("geometry") or {}).get("coordinates"))
    ]
    if not boundary_pairs:
        raise ValueError(f"North Carolina boundary has no valid coordinates: {boundary_path}")
    boundary_west = min(pair[0] for pair in boundary_pairs)
    boundary_east = max(pair[0] for pair in boundary_pairs)
    boundary_south = min(pair[1] for pair in boundary_pairs)
    boundary_north = max(pair[1] for pair in boundary_pairs)
    map_bounds = [
        [min(west, boundary_west), min(south, boundary_south)],
        [max(east, boundary_east), max(north, boundary_north)],
    ]
    replacements = {
        "__SETS__": f"{len(sources):,}",
        "__TOTAL__": f"{records_total:,}",
        "__DISPLAYED__": f"{displayed:,}",
        "__S0__": f"{severity_counts[0]:,}",
        "__S1__": f"{severity_counts[1]:,}",
        "__S2__": f"{severity_counts[2]:,}",
        "__S3__": f"{severity_counts[3]:,}",
        "__BOUNDS__": json.dumps(map_bounds, separators=(",", ":")),
    }
    html = HTML_TEMPLATE
    for old, new in replacements.items():
        html = html.replace(old, new)
    _write_atomic(html_path, lambda temporary: temporary.write_text(html, encoding="utf-8"))

    manifest = {
        "schema_version": 1,
        "source_files": [str(path) for path in sources],
        "sets": len(sources),
        "records_total": records_total,
        "mapped_coordinates": mapped,
        "records_without_coordinates": records_total - mapped,
        "overview_display_points": displayed,
        "overview_sampling": {
            "severity_0": "every 10th record",
            "severity_1": "every 5th record",
            "severity_2": "all records",
            "severity_3": "all records",
        },
        "overview_severity_counts": {str(key): display_counts[key] for key in (-1, 0, 1, 2, 3)},
        "severity_counts": {str(key): severity_counts[key] for key in (-1, 0, 1, 2, 3)},
        "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "north_carolina_bounds_wgs84": {
            "west": boundary_west,
            "south": boundary_south,
            "east": boundary_east,
            "north": boundary_north,
        },
        "map_bounds_wgs84": {
            "west": map_bounds[0][0],
            "south": map_bounds[0][1],
            "east": map_bounds[1][0],
            "north": map_bounds[1][1],
        },
        "outputs": {
            "geojson": str(geojson_path),
            "overview_geojson": str(map_geojson_path),
            "north_carolina_boundary": str(boundary_path),
            "html": str(html_path),
        },
    }
    _write_atomic(
        manifest_path,
        lambda temporary: temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8"),
    )
    return manifest


def main() -> None:
    args = _parser().parse_args()
    manifest = build(args.input_root, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
