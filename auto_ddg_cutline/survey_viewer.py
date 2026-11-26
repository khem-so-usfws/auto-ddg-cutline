"""Render an offline survey overview using browser-native Canvas."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .benchmark import replace_manifest
from .survey_photo_view import PHOTO_PANEL, PHOTO_SCRIPT, PHOTO_STYLE

SURVEY_OVERVIEW_NAME: str = "survey-overview.html"

VIEWER: str = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Survey Overview</title>
<style>
:root { color-scheme: dark; font: 16px system-ui, sans-serif; }
body { margin: 0; background: #121c22; color: #eef4f6; }
main { max-width: 78rem; margin: auto; padding: 1.25rem; }
h1, h2 { margin: 0 0 .55rem; } h1 { font-size: 1.6rem; } h2 { font-size: 1.1rem; }
p { margin: .4rem 0 .8rem; } a { color: #9bdaff; }
section { margin-top: 1.2rem; } [hidden] { display: none !important; }
ul, ol { padding-left: 1.45rem; }
li { padding: .3rem 0; overflow-wrap: anywhere; }
.controls { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; margin-bottom: .55rem; }
.controls select { max-width: min(19rem, 72vw); }
button, select { font: inherit; }
canvas { width: 100%; height: clamp(20rem, 57vh, 44rem); display: block;
  background: #0c141a; border: 1px solid #516773; box-sizing: border-box;
  border-radius: .4rem; cursor: grab; touch-action: none; }
canvas:active { cursor: grabbing; }
canvas:focus-visible { outline: 2px solid #9bdaff; outline-offset: 2px; }
.columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.25rem; }
.muted { color: #bdccd6; }
.draft { color: #79d5ff; } .refined { color: #ffd365; }
.foot-a { color: #57ded1; } .foot-b { color: #ff96c6; } .shared { color: #efd285; }
.current { background: #2b3f4b; }
@media (max-width: 45rem) { .columns { grid-template-columns: 1fr; gap: 0; } }
__PHOTO_STYLE__
</style>
</head>
<body><main>
<h1>Survey Overview</h1>
<p id="summary"></p>
<p id="draft-link" hidden></p>
<div class="controls">
  <label>View <select id="overview-mode">
    <option value="photos">Photos around the island</option>
    <option value="terrain">Terrain (3D)</option>
  </select></label>
  <label>Photos <select id="filter">
    <option value="all">All photos</option>
    <option value="checked">Checked photos</option>
  </select></label>
  <label id="block-control">Reconstruction block <select id="block"></select></label>
</div>
__PHOTO_PANEL__
<section id="scene" hidden>
  <div class="controls">
    <label><input type="checkbox" id="terrain" checked> Terrain</label>
    <label>Terrain layer <select id="terrain-layer">
      <option value="sparse">Sparse terrain</option>
      <option value="dense" id="dense-option">Dense terrain</option>
    </select></label>
    <button type="button" id="zoom-in" aria-label="Zoom in">+</button>
    <button type="button" id="zoom-out" aria-label="Zoom out">−</button>
    <button type="button" id="reset">Reset view</button>
    <label><input type="checkbox" id="cameras" checked> Cameras</label>
    <label><input type="checkbox" id="frustums"> Camera frustums</label>
  </div>
  <p id="dense-context" class="muted" hidden></p>
  <div class="controls" aria-label="Compare photo footprints">
    <label class="foot-a">Photo A <select id="photo-a"></select></label>
    <label class="foot-b">Photo B <select id="photo-b"></select></label>
  </div>
  <p id="footprints" class="muted" aria-live="polite"></p>
  <canvas id="view" tabindex="0" aria-label="Rotate the 3D survey model"></canvas>
  <p class="muted">Drag or use arrow keys to rotate. Scroll or use the buttons to zoom.
    Click a camera to set Photo A or a line to open its paired review.
    Frustums show the directions and fields of view of registered cameras.</p>
  <p><span class="draft">Draft cutline</span> · <span class="refined">Refined cutline</span>
    · White markers show checked photos. Gray markers show other photos. Breaks mark
    stretches without reliable depth.</p>
</section>
<p id="empty" hidden>No 3D block is available. The handoffs remain available below.</p>
<div class="columns">
  <section><h2 id="photo-heading">Checked photos</h2><ol id="photos"></ol></section>
  <section><h2>Cutlines and gaps</h2>
    <p id="connections"></p>
    <ul id="cuts"></ul>
    <p id="missing" class="muted"></p>
  </section>
</div>
</main>
<script type="application/json" id="overview-data">__OVERVIEW_DATA__</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("overview-data").textContent);
const byId = id => document.getElementById(id);
const photoNumber = new Map(data.photos.map((photo, index) => [photo.name, index + 1]));
const scene = byId("scene");
const canvas = byId("view");
const ctx = canvas.getContext("2d");
const blockSelect = byId("block");
const overviewMode = byId("overview-mode");
const photoFilter = byId("filter");
const terrainToggle = byId("terrain");
const terrainLayer = byId("terrain-layer");
const frustumToggle = byId("frustums");
const pickA = byId("photo-a");
const pickB = byId("photo-b");
const checked = new Set(data.photos.filter(photo => photo.checked).map(photo => photo.name));
let blockIndex = 0, yaw = 0.25, tilt = 0.32, zoom = 1;
let hitCameras = [], hitLines = [];
let drag = null, pending = false;

function linkFor(label, href) {
    const item = document.createElement(href ? "a" : "span");
    item.textContent = label;
    if (href) {
        item.href = href;
        item.target = "_blank";
        item.rel = "noopener";
    }
    return item;
}

const registered = new Set(data.blocks.flatMap(block => block.cameras.map(cam => cam.name)));
const photoRows = new Map();
for (const photo of data.photos) {
    const row = document.createElement("li");
    row.value = photoNumber.get(photo.name);
    row.append(linkFor(photo.name, photo.link));
    if (checked.has(photo.name)) row.append(" · Checked");
    if (!registered.has(photo.name)) row.append(" · Unregistered");
    byId("photos").append(row);
    photoRows.set(photo.name, row);
}
byId("summary").textContent = checked.size + " of " + data.photos.length +
    " checked · " + registered.size + " registered · " + data.cuts.length + " cutlines";
if (data.draft) {
    byId("draft-link").append(linkFor("Open draft cutline review", data.draft));
    byId("draft-link").hidden = false;
}
byId("connections").textContent = !checked.size ?
    "Switch to All photos to inspect the colony." :
    data.sections.length > 1 ?
        "The current cutlines form " + data.sections.length + " separate groups. " +
        data.sections.map(group => group.join(", ")).join(" / ") + "." :
        "The current set forms one connected group.";
if (data.missing.length) {
    const unresolved = data.missing.map(pair => pair.map(name => photoNumber.get(name)).join(" ↔ "));
    byId("missing").textContent = "Unresolved handoffs: " + unresolved.join(", ") + ".";
}
for (const block of data.blocks) {
    const option = document.createElement("option");
    option.textContent = "Block " + (blockSelect.options.length + 1) +
        " · " + block.cameras.filter(cam => checked.has(cam.name)).length + " checked" +
        " / " + block.cameras.length + " registered";
    blockSelect.append(option);
    block.pointColors = (block.colors || []).map(([r, g, b]) => "rgb(" + [
        Math.round(r * 0.7 + 44), Math.round(g * 0.7 + 50), Math.round(b * 0.7 + 51)
    ].join(",") + ")");
}

function shownCameras(block) {
    return !block ? [] : block.cameras.filter(cam =>
        photoFilter.value === "all" || checked.has(cam.name));
}

function selectedTerrain(block) {
    return terrainLayer.value === "dense" && block?.dense ? block.dense : block?.surface;
}

function updatePhotoControls(clearPins = false) {
    const block = data.blocks[blockIndex];
    const current = shownCameras(block);
    for (const select of [pickA, pickB]) {
        const previous = clearPins ? "" : select.value;
        select.replaceChildren();
        const empty = document.createElement("option");
        empty.textContent = "Choose a photo";
        empty.value = "";
        select.append(empty);
        for (const cam of current) {
            const option = document.createElement("option");
            option.value = cam.name;
            option.textContent = photoNumber.get(cam.name) + ". " + cam.name;
            select.append(option);
        }
        select.value = current.some(cam => cam.name === previous) ? previous : "";
        select.disabled = !current.length;
    }
    byId("dense-option").disabled = !block?.dense;
    if (!block?.dense) terrainLayer.value = "sparse";
    updateTerrainContext();
    byId("photo-heading").textContent = photoFilter.value === "all" ? "All photos" : "Checked photos";
    for (const [name, row] of photoRows) {
        row.hidden = photoFilter.value !== "all" && !checked.has(name);
    }
    describeFootprints();
}

function updateTerrainContext() {
    const block = data.blocks[blockIndex];
    const terrain = selectedTerrain(block);
    terrainToggle.disabled = !terrain?.faces.length;
    const context = byId("dense-context");
    context.hidden = !terrain?.focus;
    if (terrain?.focus) {
        context.textContent = "Dense pilot around " + terrain.focus + " · " +
            terrain.photos.length + " registered views. Footprints and cutlines are limited to this section.";
    }
}

function describeFootprints() {
    const label = byId("footprints");
    const block = data.blocks[blockIndex];
    for (const [name, row] of photoRows) {
        row.classList.toggle("current", name === pickA.value || name === pickB.value);
    }
    if (!selectedTerrain(block)?.faces.length) {
        label.textContent = "This block has insufficient terrain support for footprints. Showing sparse landmarks.";
        return;
    }
    if (!terrainToggle.checked) {
        label.textContent = "Turn on Terrain to see photo footprints.";
        return;
    }
    label.replaceChildren(
        "Choose up to two photos. ",
        linkLegend("Photo A", "foot-a"), " · ",
        linkLegend("Photo B", "foot-b"), " · ",
        linkLegend("Shared support", "shared"), ". Holes show where support is missing."
    );
    const pair = data.cuts.find(cut => cut.photos.includes(pickA.value) &&
        cut.photos.includes(pickB.value) && pickA.value !== pickB.value);
    if (pair?.link) label.append(" ", linkFor("Open paired review", pair.link));
}

function linkLegend(text, style) {
    const span = document.createElement("span");
    span.textContent = text;
    span.className = style;
    return span;
}

function setPin(select, other, name) {
    select.value = name;
    if (name && other.value === name) other.value = "";
    describeFootprints();
    redraw();
}

function updateCutList() {
    const list = byId("cuts");
    list.replaceChildren();
    const block = data.blocks[blockIndex];
    const lines = selectedTerrain(block)?.lines || block?.lines || [];
    const placements = new Map(lines.map(line => [line.pair, line]));
    if (!data.cuts.length) {
        const row = document.createElement("li");
        row.textContent = "No cutline joins the checked photos yet. Use Refine cutlines to look for new handoffs.";
        list.append(row);
    }
    data.cuts.forEach((cut, index) => {
        const row = document.createElement("li");
        const caption = cut.photos.map(name => photoNumber.get(name)).join(" ↔ ");
        row.append(linkFor(caption + " · " + (cut.refined ? "Refined" : "Draft"), cut.link));
        const placed = placements.get(index);
        row.append(placed ?
            " · " + Math.round(100 * placed.supported / placed.total) + "% of sampled line placed" :
            " · Photo review only on this layer");
        if (!cut.link) row.append(" · Paired HTML report unavailable");
        list.append(row);
    });
}

function paintTerrain(surface, project, scale, a, b) {
    if (!terrainToggle.checked || !surface?.faces.length) return false;
    const vertices = surface.vertices.map(project);
    const faces = surface.faces.map(([i, j, k, views], index) => {
        const p = vertices[i], q = vertices[j], r = vertices[k];
        return {p, q, r, depth: (p[2] + q[2] + r[2]) / 3,
            color: surface.colors?.[index],
            inA: a !== undefined && views.includes(a),
            inB: b !== undefined && views.includes(b)};
    }).sort((left, right) => left.depth - right.depth);
    // Paint distant faces first so hidden patches do not bleed through the surface.
    for (const {p, q, r, depth, color, inA, inB} of faces) {
        // Combine directional lighting and depth so adjacent slopes read as terrain.
        const ux = q[0] - p[0], uy = q[1] - p[1], uz = (q[2] - p[2]) * scale;
        const vx = r[0] - p[0], vy = r[1] - p[1], vz = (r[2] - p[2]) * scale;
        const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const facing = Math.abs(0.32 * nx - 0.45 * ny + 0.83 * nz) /
            (Math.hypot(nx, ny, nz) || 1);
        const tone = Math.round(23 * facing + Math.max(-7, Math.min(9, depth * 7)));
        const shade = color ? color.map(value => Math.min(255, value * (0.8 + 0.2 * facing) + 12)) :
            [48 + tone, 61 + tone, 65 + tone];
        ctx.beginPath();
        ctx.moveTo(p[0], p[1]);
        ctx.lineTo(q[0], q[1]);
        ctx.lineTo(r[0], r[1]);
        ctx.closePath();
        ctx.fillStyle = "rgb(" + shade.join(",") + ")";
        ctx.fill();
        if (inA || inB) {
            ctx.fillStyle = inA && inB ? "rgba(239,210,133,.65)" :
                inA ? "rgba(87,222,209,.53)" : "rgba(255,150,198,.53)";
            ctx.fill();
        }
    }
    return true;
}

function draw() {
    pending = false;
    if (scene.hidden || !ctx || !data.blocks.length) return;
    const block = data.blocks[blockIndex];
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const scale = Math.min(width, height) * 0.39 * zoom;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt);
    function project(p) {
        const x = p[0] * cy - p[1] * sy;
        const y = p[0] * sy + p[1] * cy;
        return [width / 2 + scale * x, height / 2 - scale * (p[2] * ct - y * st), y * ct + p[2] * st];
    }
    const a = block.cameras.find(cam => cam.name === pickA.value);
    const b = block.cameras.find(cam => cam.name === pickB.value);
    const surface = selectedTerrain(block);
    if (!paintTerrain(surface, project, scale, a?.view, b?.view)) {
        for (let index = 0; index < block.points.length; index++) {
            const [x, y, depth] = project(block.points[index]);
            ctx.fillStyle = block.pointColors[index] || "#8ba7a9";
            ctx.globalAlpha = Math.max(0.24, Math.min(0.85, 0.57 + depth * 0.19));
            ctx.fillRect(x, y, 1.7, 1.7);
        }
        ctx.globalAlpha = 1;
        for (const [camera, color] of [[a, "#57ded1"], [b, "#ff96c6"]]) {
            if (!camera) continue;
            ctx.fillStyle = color;
            for (const index of camera.observed) {
                const [x, y] = project(block.points[index]);
                ctx.fillRect(x - 1, y - 1, 3, 3);
            }
        }
    }
    hitLines = [];
    ctx.lineWidth = 2.6;
    ctx.lineCap = "round";
    for (const line of surface?.lines || block.lines) {
        const cut = data.cuts[line.pair];
        ctx.strokeStyle = cut.refined ? "#ffd365" : "#79d5ff";
        for (const span of line.spans) {
            const path = span.map(project);
            ctx.beginPath();
            ctx.moveTo(path[0][0], path[0][1]);
            for (let i = 1; i < path.length; i++) ctx.lineTo(path[i][0], path[i][1]);
            ctx.stroke();
            if (cut.link) hitLines.push({link: cut.link, path});
        }
    }
    hitCameras = [];
    const showMarkers = byId("cameras").checked;
    const showFrustums = frustumToggle.checked;
    if (!showMarkers && !showFrustums) return;
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const cam of shownCameras(block)) {
        const [x, y] = project(cam.position);
        const color = cam === a ? "#57ded1" : cam === b ? "#ff96c6" :
            checked.has(cam.name) ? "#eef4f6" : "#8798a3";
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        if (showFrustums && cam.frustum?.length === 4) {
            const corners = cam.frustum.map(project);
            ctx.globalAlpha = cam === a || cam === b ? 1 : 0.38;
            ctx.beginPath();
            ctx.moveTo(corners[0][0], corners[0][1]);
            for (const corner of corners.slice(1)) ctx.lineTo(corner[0], corner[1]);
            ctx.closePath();
            for (const corner of corners) {
                ctx.moveTo(x, y);
                ctx.lineTo(corner[0], corner[1]);
            }
            ctx.stroke();
            ctx.globalAlpha = 1;
        } else if (showMarkers) {
            const [dx, dy] = project(cam.direction);
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(dx, dy);
            ctx.stroke();
        }
        if (!showMarkers || x < 0 || x > width || y < 0 || y > height) continue;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#101c22";
        ctx.fillText(photoNumber.get(cam.name), x, y);
        hitCameras.push({x, y, name: cam.name});
    }
}

function redraw() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(draw);
}
function changeZoom(factor) {
    zoom = Math.max(0.06, Math.min(8, zoom * factor));
    redraw();
}
function rotate(x, y) {
    yaw += x;
    tilt = Math.max(-1.55, Math.min(1.55, tilt + y));
    redraw();
}
function reset() {
    yaw = 0.25;
    tilt = 0.32;
    const positions = shownCameras(data.blocks[blockIndex]);
    const radius = Math.max(1.3, ...positions.map(cam => Math.hypot(...cam.position)));
    zoom = 1.3 / radius;
    redraw();
}
function selectAt(x, y) {
    const marker = hitCameras.find(cam => Math.hypot(x - cam.x, y - cam.y) <= 11);
    if (marker) {
        setPin(pickA, pickB, marker.name);
        return;
    }
    let nearest = 9, target = null;
    for (const item of hitLines) {
        for (let i = 1; i < item.path.length; i++) {
            const [ax, ay] = item.path[i - 1], [bx, by] = item.path[i];
            const dx = bx - ax, dy = by - ay;
            const fraction = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)));
            const distance = Math.hypot(x - ax - fraction * dx, y - ay - fraction * dy);
            if (distance < nearest) {
                nearest = distance;
                target = item.link;
            }
        }
    }
    if (target) window.open(target, "_blank", "noopener");
}

__PHOTO_SCRIPT__

function updateView() {
    const terrain = overviewMode.value === "terrain";
    islandPhotos.hidden = terrain;
    scene.hidden = !terrain || !data.blocks.length || !ctx;
    byId("empty").hidden = !terrain || !scene.hidden;
    if (terrain) reset();
}

if (!ctx) byId("empty").textContent = "Canvas is unavailable in this browser. Use the paired-photo links below.";
byId("block-control").hidden = data.blocks.length <= 1;
overviewMode.addEventListener("change", updateView);
photoFilter.addEventListener("change", () => {
    updatePhotoControls();
    updateIslandPhotos();
    reset();
});
blockSelect.addEventListener("change", () => {
    blockIndex = blockSelect.selectedIndex;
    focusPicker.value = "";
    updatePhotoControls(true);
    updateIslandPhotos();
    updateCutList();
    reset();
});
if (ctx && data.blocks.length) {
    pickA.addEventListener("change", () => setPin(pickA, pickB, pickA.value));
    pickB.addEventListener("change", () => setPin(pickB, pickA, pickB.value));
    terrainToggle.addEventListener("change", () => {
        describeFootprints();
        redraw();
    });
    terrainLayer.addEventListener("change", () => {
        updateTerrainContext();
        describeFootprints();
        updateCutList();
        redraw();
    });
    frustumToggle.addEventListener("change", redraw);
    byId("reset").addEventListener("click", reset);
    byId("zoom-in").addEventListener("click", () => changeZoom(1.2));
    byId("zoom-out").addEventListener("click", () => changeZoom(1 / 1.2));
    byId("cameras").addEventListener("change", redraw);
    canvas.addEventListener("pointerdown", event => {
        canvas.setPointerCapture(event.pointerId);
        drag = {x: event.clientX, y: event.clientY, distance: 0};
    });
    canvas.addEventListener("pointermove", event => {
        if (!drag) return;
        const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
        drag = {x: event.clientX, y: event.clientY, distance: drag.distance + Math.hypot(dx, dy)};
        rotate(dx * 0.008, dy * 0.008);
    });
    canvas.addEventListener("pointerup", event => {
        if (drag?.distance < 5) {
            const bounds = canvas.getBoundingClientRect();
            selectAt(event.clientX - bounds.left, event.clientY - bounds.top);
        }
        drag = null;
    });
    canvas.addEventListener("pointercancel", () => { drag = null; });
    canvas.addEventListener("wheel", event => {
        event.preventDefault();
        changeZoom(event.deltaY < 0 ? 1.1 : 1 / 1.1);
    }, {passive: false});
    canvas.addEventListener("keydown", event => {
        const step = {ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0],
            ArrowUp: [0, -0.1], ArrowDown: [0, 0.1]}[event.key];
        if (!step) return;
        event.preventDefault();
        rotate(...step);
    });
    new ResizeObserver(redraw).observe(canvas);
}
const denseBlock = data.denseFocus ?
    data.blocks.findIndex(block => block.dense?.focus === data.denseFocus) : -1;
if (denseBlock >= 0) {
    blockIndex = denseBlock;
    blockSelect.selectedIndex = denseBlock;
    overviewMode.value = "terrain";
    byId("dense-option").disabled = false;
    terrainLayer.value = "dense";
}
updatePhotoControls();
updateCutList();
updateIslandPhotos();
updateView();
</script>
</body></html>
"""


def write_overview(path: Path, data: dict[str, Any]) -> None:
    """Write one self-contained HTML viewer with links to the local photo reviews.

    Args:
        path: Destination in the overview job folder.
        data: Reconstruction blocks and the current checkbox snapshot.
    """
    # Escape opening tags in JSON so a filename cannot terminate the script.
    payload: str = json.dumps(data, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    page: str = (
        VIEWER.replace("__PHOTO_STYLE__", PHOTO_STYLE)
        .replace("__PHOTO_PANEL__", PHOTO_PANEL)
        .replace("__PHOTO_SCRIPT__", PHOTO_SCRIPT)
        .replace("__OVERVIEW_DATA__", payload)
    )
    write_overview_file(path, page)


def write_overview_file(path: Path, text: str) -> None:
    """Atomically publish a refreshed overview or a cached terrain file."""
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary: Path = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            replace_manifest(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def ensure_overview_entry(folder: Path) -> None:
    """Provide a working review link before the first overview is built."""
    path: Path = folder / SURVEY_OVERVIEW_NAME
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(
                '<!doctype html><html lang="en"><meta charset="utf-8">'
                "<title>Survey Overview</title><h1>Survey Overview</h1>"
                "<p>Use <strong>Update Survey Overview</strong> in the launcher to "
                "build the overview from your checked photos.</p>"
                '<p><a href="selection-plan.html">Back to draft cutlines</a></p></html>'
            )
    except FileExistsError:
        # A generated report supersedes this entry page.
        pass
