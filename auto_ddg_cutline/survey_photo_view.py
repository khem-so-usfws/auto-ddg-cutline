"""Markup and interactions for the offline photo view of a survey."""

from __future__ import annotations

PHOTO_STYLE: str = """
.photo-ring { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .55rem;
  max-width: 61rem; margin: .75rem auto 1.2rem; }
.ring-center { display: grid; place-content: center; text-align: center; padding: .5rem;
  font-size: .86rem; color: #bdccd6; }
.ring-center strong { color: #eef4f6; font-size: .95rem; }
.photo-card, .photo-slot { min-width: 0; border: 1px solid #415563; border-radius: .35rem;
  overflow: hidden; background: #1d2a32; }
.photo-slot { display: grid; place-items: center; min-height: 2.7rem;
  border-style: dashed; color: #a9bdc9; font-size: .82rem; text-align: center; }
.photo-card button { display: block; width: 100%; padding: 0; border: 0;
  color: inherit; background: none; text-align: left; cursor: pointer; }
.photo-card button:focus-visible { outline: 2px solid #ebf8ff; outline-offset: -2px; }
.photo-card img, .photo-placeholder { width: 100%; aspect-ratio: 16 / 9;
  object-fit: contain; display: block; background: #0a131a; }
.photo-placeholder { box-sizing: border-box; display: grid; place-items: center;
  color: #a9bdc9; text-align: center; }
.photo-card.active { border-color: #57ded1; box-shadow: 0 0 0 1px #57ded1 inset; }
.photo-card.neighbor { border-color: #efd285; }
.photo-card .title, .photo-card small { display: block; padding: .15rem .45rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.photo-card small { color: #bdccd6; font-size: .78rem; }
.photo-card.neighbor small { color: #efd285; }
.photo-card.active small { color: #57ded1; }
.photo-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .65rem; max-width: 61rem; margin: .6rem auto; }
.photo-pair figure { margin: 0; padding: .4rem; min-width: 0; }
.photo-pair figcaption { margin-top: .35rem; overflow-wrap: anywhere; font-size: .85rem; }
.photo-pair img, .photo-pair .photo-placeholder { aspect-ratio: 4 / 3; }
.photo-pair .photo-placeholder { padding: .55rem; }
@media (max-width: 34rem) {
  .photo-ring { gap: .3rem; }
  .ring-center { font-size: .73rem; padding: .3rem; }
  .ring-center strong { font-size: .78rem; }
  .photo-card .title { font-size: .82rem; }
}
"""

PHOTO_PANEL: str = """
<section id="island-photos" aria-label="Photos around the island">
  <h2>Photos around the island</h2>
  <p id="photo-layout-note" class="muted"></p>
  <div class="controls">
    <label>Focus photo <select id="focus-photo"></select></label>
    <label>Compare with <select id="neighbor-photo"></select></label>
  </div>
  <div id="photo-ring" class="photo-ring" aria-label="Island photo groups"></div>
  <p id="photo-pair-status" aria-live="polite"></p>
  <div id="photo-pair" class="photo-pair"></div>
</section>
"""

PHOTO_SCRIPT: str = """
const islandPhotos = byId("island-photos");
const focusPicker = byId("focus-photo");
const neighborPicker = byId("neighbor-photo");
const photoByName = new Map(data.photos.map(photo => [photo.name, photo]));
const photoNeighbors = new Map(data.photos.map(photo => [photo.name, new Map()]));

for (const edge of data.overlaps || []) {
    const [a, b] = edge.photos;
    photoNeighbors.get(a)?.set(b, edge);
    photoNeighbors.get(b)?.set(a, edge);
}
// Give recovered or refined cutlines priority over the draft preview link.
for (const cut of data.cuts) {
    const [a, b] = cut.photos;
    const edge = {...photoNeighbors.get(a)?.get(b), ...cut, cutline: true};
    photoNeighbors.get(a)?.set(b, edge);
    photoNeighbors.get(b)?.set(a, edge);
}

function surveyNames() {
    return data.photos.filter(photo => photoFilter.value === "all" || photo.checked)
        .map(photo => photo.name);
}

function photoOptions(picker, names, previous, placeholder) {
    picker.replaceChildren();
    if (placeholder) {
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = placeholder;
        picker.append(empty);
    }
    for (const name of names) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = photoNumber.get(name) + ". " + name;
        picker.append(option);
    }
    picker.value = names.includes(previous) ? previous : placeholder ? "" : (names[0] || "");
    picker.disabled = !names.length;
}

function photoImage(photo, placeholder = "Preview unavailable") {
    const item = document.createElement(photo?.thumbnail ? "img" : "span");
    if (photo?.thumbnail) {
        item.src = photo.thumbnail;
        item.alt = photo.name;
    } else {
        item.className = "photo-placeholder";
        item.textContent = placeholder;
    }
    return item;
}

function showPhotoPair() {
    const left = photoByName.get(focusPicker.value);
    const right = photoByName.get(neighborPicker.value);
    const status = byId("photo-pair-status");
    const pair = byId("photo-pair");
    status.replaceChildren();
    pair.replaceChildren();
    if (!left) return;
    const edge = photoNeighbors.get(left.name)?.get(right?.name);
    if (edge && right) {
        const text = edge.cutline ? (edge.refined ? "Refined cutline" : "Draft cutline") :
            "Verified overlap";
        status.append(text + (edge.link ? " · " : ""));
        if (edge.link) status.append(linkFor("Open paired review", edge.link));
    } else {
        status.textContent = "Select a verified neighbor to compare the original views.";
    }
    for (const photo of [left, right]) {
        if (!photo) continue;
        const frame = document.createElement("figure");
        frame.className = "photo-card";
        frame.append(photoImage(photo));
        const label = document.createElement("figcaption");
        label.append(linkFor(photoNumber.get(photo.name) + ". " + photo.name, photo.link));
        frame.append(label);
        pair.append(frame);
    }
}

function updatePhotoNeighbors(previous = neighborPicker.value) {
    const edges = photoNeighbors.get(focusPicker.value) || new Map();
    const names = new Set(surveyNames());
    const neighbors = Array.from(edges.keys()).filter(name => names.has(name));
    neighbors.sort((a, b) => Number(!!edges.get(b).cutline) - Number(!!edges.get(a).cutline) ||
        (edges.get(b).inliers || 0) - (edges.get(a).inliers || 0));
    photoOptions(neighborPicker, neighbors, previous || neighbors[0],
        neighbors.length ? "Choose a verified neighbor" : "No verified neighbor in this view");
    showPhotoPair();
}

function photoCard(name, count, edges) {
    const photo = photoByName.get(name);
    const card = document.createElement("div");
    const focused = name === focusPicker.value;
    const overlap = edges.has(name);
    card.className = "photo-card" + (focused ? " active" : overlap ? " neighbor" : "");
    const button = document.createElement("button");
    button.type = "button";
    button.title = photo.name;
    button.append(photoImage(photo));
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = photoNumber.get(name) + ". " + photo.name;
    const note = document.createElement("small");
    note.textContent = [
        focused ? "Focus photo" : edges.get(name)?.cutline ? "Cutline" : overlap ? "Verified overlap" : "",
        photo.checked ? "Checked" : "",
        count > 1 ? count + " views in this direction" : "",
    ].filter(Boolean).join(" · ");
    button.append(title);
    if (note.textContent) button.append(note);
    button.addEventListener("click", () => {
        if (overlap) {
            neighborPicker.value = name;
            showPhotoPair();
        } else if (!focused) {
            focusPicker.value = name;
            photoFocusChanged();
        }
    });
    card.append(button);
    return card;
}

function drawPhotoRing(names) {
    const ring = byId("photo-ring");
    const note = byId("photo-layout-note");
    const block = data.blocks[blockIndex];
    const edges = photoNeighbors.get(focusPicker.value) || new Map();
    ring.replaceChildren();
    if (!block || !shownCameras(block).length) {
        note.textContent = "Photos in filename order. Switch reconstruction blocks or show All photos to see camera positions.";
        for (const name of names) ring.append(photoCard(name, 1, edges));
        return;
    }
    note.textContent = "Camera positions are grouped around the island. Select a focus photo and use the highlighted overlaps to compare views.";
    const sectors = Array.from({length: 8}, () => []);
    for (const camera of shownCameras(block)) {
        const [x, y] = camera.position;
        const direction = (Math.round(Math.atan2(x, y) / (Math.PI / 4)) + 8) % 8;
        sectors[direction].push(camera.name);
    }
    // Clockwise sectors occupy the eight cells around the center of the grid.
    for (const sector of [7, 0, 1, 6, -1, 2, 5, 4, 3]) {
        if (sector === -1) {
            const center = document.createElement("div");
            center.className = "ring-center";
            const label = document.createElement("strong");
            label.textContent = "Island center";
            center.append(label, "Approximate camera directions");
            ring.append(center);
            continue;
        }
        const candidates = sectors[sector];
        if (!candidates.length) {
            const slot = document.createElement("div");
            slot.className = "photo-slot";
            slot.textContent = "No photo in this view";
            ring.append(slot);
            continue;
        }
        // Prefer the focus, then a verified neighbor so the ring exposes links.
        candidates.sort((a, b) => Number(b === focusPicker.value) - Number(a === focusPicker.value) ||
            Number(edges.has(b)) - Number(edges.has(a)) ||
            Number(checked.has(b)) - Number(checked.has(a)) ||
            photoNumber.get(a) - photoNumber.get(b));
        ring.append(photoCard(candidates[0], candidates.length, edges));
    }
}

function photoFocusChanged() {
    if (!data.blocks[blockIndex]?.cameras.some(cam => cam.name === focusPicker.value)) {
        const containing = data.blocks.findIndex(block =>
            block.cameras.some(cam => cam.name === focusPicker.value));
        if (containing >= 0) {
            blockIndex = containing;
            blockSelect.selectedIndex = containing;
            updatePhotoControls(true);
            updateCutList();
        }
    }
    const names = surveyNames();
    drawPhotoRing(names);
    updatePhotoNeighbors("");
}

function updateIslandPhotos() {
    const names = surveyNames();
    const cameras = shownCameras(data.blocks[blockIndex]).map(cam => cam.name);
    const priority = cameras.find(name => checked.has(name)) || cameras[0] || names[0];
    const previous = focusPicker.value;
    photoOptions(focusPicker, names, previous || priority,
        names.length ? "" : "No checked photos");
    drawPhotoRing(names);
    updatePhotoNeighbors();
}
focusPicker.addEventListener("change", photoFocusChanged);
neighborPicker.addEventListener("change", showPhotoPair);
"""
