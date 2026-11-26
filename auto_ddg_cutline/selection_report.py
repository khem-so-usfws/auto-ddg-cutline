"""Write a portable colony-selection report and its machine-readable evidence."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import Frame
from .paired import IndexPair
from .selection_evidence import GRID_COLS, GRID_ROWS, SHARPER_ALTERNATE_RATIO
from .survey_viewer import SURVEY_OVERVIEW_NAME, ensure_overview_entry

STYLE: str = """
body { margin: 0 auto; padding: 1.5rem; max-width: 85rem; color: #172c33;
font: 16px/1.5 system-ui, sans-serif; background: #f6f8f8; }
h1, h2, h3 { line-height: 1.15; }
h2 { margin-top: 1.8rem; } h3 { overflow-wrap: anywhere; }
nav { display: flex; flex-wrap: wrap; gap: .6em 1.4em; }
a { color: #005a69; } ul { padding-left: 1.35rem; }
.card { padding: .85rem 1rem; margin: .7rem 0; border: 1px solid #c8d6d7; border-radius: .45rem; background: white; }
.status { color: #385761; } .warning { border-left: .35rem solid #9a5a0d; }
.photos, .pair { display: grid; gap: .8rem; }
.photos { grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); }
.pair { grid-template-columns: repeat(2, minmax(0, 1fr)); }
figure { margin: 0; min-width: 0; } figcaption { overflow-wrap: anywhere; }
img { width: 100%; display: block; border-radius: .25rem; }
.pair img { max-width: 36rem; } details { margin-top: .65rem; }
summary { cursor: pointer; overflow-wrap: anywhere; }
table { border-collapse: collapse; }
th, td { border-bottom: 1px solid #c8d6d7; padding: .35rem .65rem; text-align: left; }
.pair-controls { align-items: center; flex-wrap: wrap; gap: .5rem; margin-bottom: .6rem; }
.pair-controls:not([hidden]) { display: flex; }
.pair-controls select { max-width: min(100%, 42rem); }
.native-scroll { height: 26rem; overflow: auto; background: #172c33; }
.native-photo { position: relative; }
.pair .native-photo img, .native-photo svg { position: absolute; inset: 0; max-width: none; }
.native-photo img { border-radius: 0; }
:target { outline: 2px solid #005a69; outline-offset: .15rem; }
@media (max-width: 40rem) { .pair { grid-template-columns: 1fr; } }
"""

REVIEW_NAVIGATION: str = """
const drafts = Array.from(document.querySelectorAll("[data-draft]"));
const pairPicker = document.getElementById("draft-picker");
const previous = document.getElementById("previous-draft");
const next = document.getElementById("next-draft");
function showDraft(index) {
    if (index < 0 || index >= drafts.length) return;
    drafts.forEach((draft, position) => {
        draft.hidden = position !== index;
        draft.open = position === index;
    });
    pairPicker.selectedIndex = index;
    previous.disabled = index === 0;
    next.disabled = index === drafts.length - 1;
}
if (drafts.length) {
    for (const draft of drafts) {
        const option = document.createElement("option");
        option.textContent = draft.querySelector("summary").textContent;
        pairPicker.append(option);
    }
    pairPicker.addEventListener("change", () => showDraft(pairPicker.selectedIndex));
    previous.addEventListener("click", () => showDraft(pairPicker.selectedIndex - 1));
    next.addEventListener("click", () => showDraft(pairPicker.selectedIndex + 1));
    document.getElementById("draft-controls").hidden = false;
    showDraft(0);
}
function showLinkedItem() {
    const target = document.getElementById(location.hash.slice(1));
    if (!target) return;
    const draft = target.closest("[data-draft]");
    if (draft) showDraft(drafts.indexOf(draft));
    for (let item = target; item; item = item.parentElement) {
        if (item.tagName === "DETAILS") item.open = true;
    }
    target.scrollIntoView({block: "start"});
}
addEventListener("hashchange", showLinkedItem);
showLinkedItem();
for (const detail of document.querySelectorAll(".native-detail")) {
    detail.addEventListener("toggle", () => {
        if (!detail.open) return;
        for (const pane of detail.querySelectorAll(".native-scroll")) {
            pane.scrollLeft = Number(pane.dataset.x) - pane.clientWidth / 2;
            pane.scrollTop = Number(pane.dataset.y) - pane.clientHeight / 2;
        }
    });
}
"""


def image_data(pixels: NDArray[np.uint8]) -> str:
    """Encode a small preview for a standalone offline HTML report."""
    success: bool
    encoded: NDArray[np.uint8]
    success, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise OSError("Could not encode an overlap preview.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def thumbnail(frame: Frame) -> NDArray[np.uint8]:
    """Keep stored orientation and convert the working raster to a small preview."""
    height: int
    width: int
    height, width = frame.gray.shape
    scale: float = min(1.0, 420 / max(height, width))
    pixels: NDArray[np.uint8] = cv2.resize(
        frame.gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)


def native_points(
    points: list[list[float]], row: dict[str, Any], pixels: NDArray[np.uint8]
) -> NDArray[np.int32]:
    """Map source pixel centers to preview coordinates."""
    height: int
    width: int
    height, width = pixels.shape[:2]
    scale: NDArray[np.float64] = np.array(
        [width / row["width"], height / row["height"]]
    )
    return np.rint((np.asarray(points) + 0.5) * scale - 0.5).astype(np.int32)


def mark_cells(
    pixels: NDArray[np.uint8], cells: list[int], color: tuple[int, int, int]
) -> None:
    """Draw a cell outline to show where supporting landmarks were sampled."""
    height: int
    width: int
    height, width = pixels.shape[:2]
    for cell in cells:
        grid_row: int
        column: int
        grid_row, column = divmod(cell, GRID_COLS)
        left: int
        top: int
        left, top = column * width // GRID_COLS, grid_row * height // GRID_ROWS
        right: int = (column + 1) * width // GRID_COLS - 1
        bottom: int = (grid_row + 1) * height // GRID_ROWS - 1
        cv2.rectangle(pixels, (left, top), (right, bottom), color, 1, cv2.LINE_AA)


def caption(text: str, uri: str, alt: str) -> str:
    """Render a tagged preview with an accessible caption."""
    return f'<figure><img loading="lazy" src="{uri}" alt="{escape(alt)}"><figcaption>{text}</figcaption></figure>'


def native_comparison(
    pair: dict[str, Any], rows: list[dict[str, Any]], frames: list[Frame]
) -> str:
    """Overlay the two draft versions on scrollable stored-orientation pixels."""
    earlier: dict[str, Any] | None = pair.get("initial_handoff") or pair.get(
        "working_handoff"
    )
    current: dict[str, Any] | None = pair.get("handoff")
    if not current and not earlier:
        return ""
    panes: list[str] = []
    for offset, index in enumerate(pair["photos"]):
        row: dict[str, Any] = rows[index]
        width: int = row["width"]
        height: int = row["height"]
        photo: str = escape(row["name"])
        image_name: str = escape(row["id"])
        lines: list[str] = []
        for outline in frames[index].protected_polygons:
            vertices_text: str = " ".join(f"{x:.2f},{y:.2f}" for x, y in outline)
            lines.append(
                f'<polygon points="{vertices_text}" fill="none" stroke="#e879f9" stroke-width="2"/>'
            )
        x: float = width / 2
        y: float = height / 2
        for handoff, color in ((earlier, "#72e7ff"), (current, "#ffdb35")):
            if handoff is None:
                continue
            vertices: list[list[float]] = handoff["views"][offset]["line"]
            x, y = np.mean(vertices, axis=0).tolist()
            points: str = " ".join(f"{px:.2f},{py:.2f}" for px, py in vertices)
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        panes.append(
            f"<figure><figcaption>{photo}</figcaption>"
            f'<div class="native-scroll" tabindex="0" data-x="{x:.2f}" data-y="{y:.2f}">'
            f'<div class="native-photo" style="width:{width}px;height:{height}px">'
            f'<img loading="lazy" src="images/{image_name}" alt="{photo}">'
            f'<svg width="{width}" height="{height}" viewBox="-.5 -.5 {width} {height}" aria-hidden="true">'
            + "".join(lines)
            + "</svg></div></div></figure>"
        )
    prior: str = "Initial cut" if "initial_handoff" in pair else "Working draft"
    key: str = f"{prior} in blue. " if earlier else ""
    if current:
        key += "Refined cut in yellow. "
    if any(frames[index].protected_polygons for index in pair["photos"]):
        key += "Protected groups in magenta. "
    return (
        '<details class="native-detail"><summary>Inspect original pixels at 100%</summary>'
        f'<p>{key}Scroll each photo to follow the line.</p><div class="pair">'
        + "".join(panes)
        + "</div></details>"
    )


def pair_preview(
    pair: dict[str, Any],
    rows: list[dict[str, Any]],
    thumbnails: list[NDArray[np.uint8]],
    frames: list[Frame],
) -> str:
    """Show draft seams and keep sides, or landmarks for pairs without drafts."""
    figures: list[str] = []
    palette: tuple[tuple[int, int, int], ...] = (
        (50, 200, 255),
        (230, 255, 55),
        (255, 125, 230),
        (130, 240, 120),
    )
    for position, index in enumerate(pair["photos"]):
        pixels: NDArray[np.uint8] = thumbnails[index].copy()
        row: dict[str, Any] = rows[index]
        label: str = escape(row["name"])
        for polygon in frames[index].protected_polygons:
            outline: NDArray[np.int32] = native_points(polygon.tolist(), row, pixels)
            cv2.polylines(pixels, [outline], True, (249, 121, 232), 2, cv2.LINE_AA)
        if "handoff" not in pair and "documented_cells_by_photo" in pair:
            mark_cells(
                pixels, pair["documented_cells_by_photo"][position], (190, 164, 44)
            )
        if "handoff" not in pair and "matched_landmarks_native" in pair:
            points: NDArray[np.int32] = native_points(
                pair["matched_landmarks_native"][position], row, pixels
            )
            for serial, point in enumerate(points, start=1):
                center: tuple[int, int] = (int(point[0]), int(point[1]))
                cv2.circle(
                    pixels,
                    center,
                    5,
                    palette[(serial - 1) % len(palette)],
                    -1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    pixels,
                    str(serial),
                    (center[0] + 6, center[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        if "handoff" in pair:
            view: dict[str, Any] = pair["handoff"]["views"][position]
            line: NDArray[np.int32] = native_points(view["line"], row, pixels)
            cv2.polylines(pixels, [line], False, (15, 15, 15), 5, cv2.LINE_AA)
            cv2.polylines(pixels, [line], False, (35, 240, 255), 2, cv2.LINE_AA)
            keep: NDArray[np.int32] = native_points([view["keep_point"]], row, pixels)[
                0
            ]
            marker: tuple[int, int] = (int(keep[0]), int(keep[1]))
            cv2.circle(pixels, marker, 9, (15, 15, 15), 4, cv2.LINE_AA)
            cv2.circle(pixels, marker, 9, (87, 255, 105), 2, cv2.LINE_AA)
            label += (
                f" · Count toward the green ring ({escape(view['keep_direction'])})"
            )
        original: str = escape(frames[index].path.resolve().as_uri())
        label += f' · <a href="{original}">Open original</a>'
        figures.append(
            caption(
                label,
                image_data(pixels),
                (
                    "Proposed counting boundary and keep-side marker"
                    if "handoff" in pair
                    else "Matching landmarks in the source photo"
                ),
            )
        )
    return '<div class="pair">' + "".join(figures) + "</div>"


def photo_card(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    frames: list[Frame],
    thumbnails: list[NDArray[np.uint8]],
    available: dict[IndexPair, str],
    *,
    compact: bool = False,
) -> str:
    """Show a photo's selection status, unassessed area and best available replacement."""
    index: int = row["index"]
    pixels: NDArray[np.uint8] = thumbnails[index].copy()
    mark_cells(pixels, row["unassessed_cells"], (60, 179, 235))
    gaps: list[int] = (
        row["cells_without_selected_replacement"]
        if (compact and row["state"] == "coverage review")
        else []
    )
    mark_cells(pixels, gaps, (60, 65, 232))
    name: str = escape(row["name"])
    alternatives: list[dict[str, Any]] = row["selected_replacements"]
    adjacent: str = ", ".join(
        escape(rows[other]["name"]) for other in row["supported_neighbors"]
    )
    explanation: str = (
        f"<p>{len(row['documented_cells'])} of {GRID_COLS * GRID_ROWS} cells have local evidence. "
        f"{len(row['unassessed_cells'])} cells need inspection (orange outlines).</p>"
    )
    if gaps:
        explanation += (
            f"<p>{len(gaps)} documented cells need a counting view (red outlines).</p>"
        )
    sharper: int = len(row.get("cells_with_sharper_alternative", []))
    if sharper:
        explanation += (
            f"<p>A comparable view is sharper in {sharper} matched cells.</p>"
        )
    if adjacent:
        explanation += f"<p>Supported overlap with {adjacent}.</p>"
    if alternatives:
        options: list[str] = []
        for alternative in alternatives:
            target: int = alternative["photo"]
            link: str = escape(rows[target]["name"])
            key: IndexPair = (min(index, target), max(index, target))
            if key in available:
                link = f'<a href="#{available[key]}">{link}</a>'
            improved: int = sum(
                (cell.get("relative_sharpness") or 0) >= SHARPER_ALTERNATE_RATIO
                for cell in alternative["cells"]
            )
            sharpness: str = f", sharper in {improved}" if improved else ""
            options.append(
                f"<li>{link} ({len(alternative['cells'])} directly supported cells{sharpness})</li>"
            )
        explanation += (
            "<details><summary>Selected replacement views</summary><ul>"
            + "".join(options)
            + "</ul></details>"
        )
    original: str = escape(frames[index].path.resolve().as_uri())
    return (
        f'<article class="card" id="photo-{index}"><h3>{name}</h3>'
        f'<p class="status"><strong>{escape(row["state"].capitalize())}</strong> · {escape(row["reason"])}</p>'
        f'<img loading="lazy" src="{image_data(pixels)}" alt="Photo thumbnail with unassessed cells outlined">'
        f'{explanation}<p><a href="{original}">Open original</a></p></article>'
    )


def write_selection_report(
    job: Path, plan: dict[str, Any], frames: list[Frame]
) -> None:
    """Write the review pages into the run workspace using native-pixel evidence."""
    target: Path = job / "selection-plan.json"
    target.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = plan["photos"]
    thumbnails: list[NDArray[np.uint8]] = [thumbnail(frame) for frame in frames]
    refined: bool = bool(plan.get("refinement"))
    compact: dict[str, Any] | None = plan.get("compact_selection")

    def link(index: int) -> str:
        """Link a review item to its original photo card."""
        return f'<a href="#photo-{index}">{escape(rows[index]["name"])}</a>'

    def links(indexes: list[int], separator: str = " / ") -> str:
        """Join photo links into a pair, route, or review group."""
        return separator.join(link(index) for index in indexes)

    def photo_gallery(label: str, indexes: list[int]) -> str:
        """Put one group of photo cards behind a single disclosure."""
        return (
            f'<details class="card"><summary>{label} ({len(indexes)})</summary>'
            + '<div class="photos">'
            + "".join(
                photo_card(
                    rows[index],
                    rows,
                    frames,
                    thumbnails,
                    previews,
                    compact=compact is not None,
                )
                for index in indexes
            )
            + "</div></details>"
        )

    previews: dict[IndexPair, str] = {}
    preview_cards: dict[str, list[str]] = defaultdict(list)
    draft_cards: list[str] = []
    for pair in plan["overlap_pairs"]:
        a: int
        b: int
        a, b = pair["photos"]
        if not any(
            key in pair
            for key in ("handoff", "handoff_error", "matched_landmarks_native")
        ):
            continue
        anchor: str = f"pair-{a}-{b}"
        previews[(a, b)] = anchor
        label: str = f'{pair["inliers"]:,} verified matches.'
        if "held_out_total" in pair:
            label += f' {pair["held_out_supported"]} of {pair["held_out_total"]} held-out checks agree locally.'
        group: str = "Replacement evidence"
        if pair.get("review_count_assignment"):
            label += " Inspect assignment between these overlapping counting views."
            group = "Overlapping counting views"
        if "handoff_error" in pair:
            label += " Handoff needs review. " + escape(pair["handoff_error"])
            group = "Failed handoffs"
        has_draft: bool = "handoff" in pair
        if has_draft:
            if pair["handoff"].get("group_rerouted"):
                label += " Rerouted around a protected bird group."
            warnings: list[str] = pair["handoff"]["warnings"]
            if warnings:
                label += " " + escape(" ".join(warnings))
        title: str = " / ".join(escape(rows[index]["name"]) for index in pair["photos"])
        draft_attribute: str = " data-draft" if has_draft else ""
        opened: str = " open" if has_draft and not draft_cards else ""
        card: str = (
            f'<details class="card" id="{anchor}"{draft_attribute}{opened}>'
            f'<summary>{title}</summary><p>{links(pair["photos"])} · {label}</p>'
            + pair_preview(pair, rows, thumbnails, frames)
            + (native_comparison(pair, rows, frames) if refined else "")
            + "</details>"
        )
        if has_draft:
            draft_cards.append(card)
        else:
            preview_cards[group].append(card)

    summary: dict[str, int] = plan["summary"]
    manual: bool = plan.get("counting_set_source") == "analyst"
    source: str = "selected" if manual else "suggested"
    export_pairs: set[tuple[int, ...]] = {
        tuple(pair) for pair in plan.get("export_pairs", [])
    }
    heading: str = "Refined cutline review" if refined else "Draft cutline review"
    proposed: set[int] = set(plan["counting_set"])
    excluded: list[int] = plan["rules"]["exclude"]
    needs_review: list[int] = [
        row["index"]
        for row in rows
        if row["index"] not in proposed and row["state"] == "coverage review"
    ]
    remaining: list[int] = sorted(
        set(range(len(rows))) - proposed - set(excluded) - set(needs_review)
    )
    local: dict[str, Any] | None = plan.get("local_sfm")
    local_pairs: set[tuple[int, ...]] = (
        {tuple(pair["photos"]) for pair in local["drafts"]} if local else set()
    )
    primary: set[tuple[int, ...]] = {
        tuple(pair["photos"]) for pair in plan["overlap_pairs"] if "handoff" in pair
    }
    local_comparisons: list[str] = []
    if local is not None:
        for pair in local["drafts"]:
            a, b = pair["photos"]
            exportable: bool = (a, b) in export_pairs - primary
            title = " / ".join(escape(rows[index]["name"]) for index in (a, b))
            label = f'Draft from local SfM block {pair["block"]}. {pair["three_view_tracks"]:,} landmarks are supported by at least three views.'
            warning: str = " ".join(pair["handoff"]["warnings"])
            if warning:
                label += " " + escape(warning)
            draft_attribute = " data-draft" if exportable else ""
            opened = " open" if exportable and not draft_cards else ""
            card = (
                f'<details class="card" id="local-pair-{a}-{b}"{draft_attribute}{opened}>'
                f'<summary>{title}</summary><p>{links(pair["photos"])} · {label}</p>'
                + pair_preview(pair, rows, thumbnails, frames)
                + (
                    native_comparison(pair, rows, frames)
                    if refined and exportable
                    else ""
                )
                + "</details>"
            )
            if exportable:
                draft_cards.append(card)
            else:
                local_comparisons.append(card)
    pilot: dict[str, Any] | None = plan.get("shared_patch_pilot")
    pilot_link: str = (
        '<a href="#shared-patches">Shared 3D patch test</a>'
        if pilot is not None
        else ""
    )
    compact_summary: str = ""
    if compact is not None:
        target: int = compact["target_photos"]
        protected: int = len(compact["protected_photos"])
        compact_summary = (
            f"<p><strong>Counting photo target: {target}</strong> · "
            f"{protected} views kept for distinct terrain, section coverage or analyst pins. "
            f'<a href="#compact-evidence">See what the compact set covers</a>.</p>'
        )
    overview_link: str = ""
    if not refined:
        ensure_overview_entry(job)
        overview_link = f'<a href="{SURVEY_OVERVIEW_NAME}">Survey Overview</a>'
    body: list[str] = [
        f"<h1>{heading}</h1>",
        f"<p><strong>{len(proposed)} {source} of {len(rows)} photos</strong> "
        f"· {len(needs_review)} omitted photos to check "
        f'· {summary["unresolved_handoffs"]} joins to check</p>',
        compact_summary,
        '<nav><a href="#overlaps">Paired cutlines</a><a href="#review">Review flags</a>'
        '<a href="#sections">Counting set</a><a href="#omissions">Omitted photos</a>'
        '<a href="#local">Local SfM</a><a href="selection-plan.json">Plan JSON</a>'
        f"{overview_link}{pilot_link}</nav>",
        '<h2 id="overlaps">Paired cutlines</h2>',
    ]
    if draft_cards:
        routing_note: str = (
            " Draft routing favors crossing fewer strong color edges."
            if plan.get("draft_color_edge_weight")
            else ""
        )
        body.extend(
            [
                "<p>Yellow lines mark cutlines. Count toward the green ring in each photo."
                f"{routing_note}</p>",
                '<div id="draft-controls" class="pair-controls" hidden>'
                '<button type="button" id="previous-draft">Previous</button>'
                '<label for="draft-picker">Photo pair</label><select id="draft-picker"></select>'
                '<button type="button" id="next-draft">Next</button></div>',
                *draft_cards,
            ]
        )
    else:
        body.append(
            "<p>No paired cutlines were generated. Check the flagged joins below.</p>"
        )

    body.append('<h2 id="review">Review flags</h2>')
    if "photos_with_sharper_alternatives" in summary:
        body.append(
            f'<p>{summary["photos_with_sharper_alternatives"]} photos have sharper alternatives '
            "for some matched terrain. Check the photo cards for those comparisons.</p>"
        )
    if plan.get("protected_groups"):
        body.append(
            f'<p>Used {summary["protected_group_polygons"]} protected DDG polygons. '
            f'Rerouted {summary["group_rerouted_handoffs"]} handoffs. '
            "Magenta outlines mark groups drawn in that photo. Check the matching birds and count side in both photos.</p>"
        )
    routes: dict[str, Any] | None = plan.get("refinement_routes")
    if routes is not None:
        body.append(
            f'<p>Refined {summary["refined_reviewed_handoffs"]} of '
            f'{summary["reviewed_handoffs"]} reviewed photo pairs and planned '
            f'{summary["replanned_handoffs"]} new pairs.</p>'
        )
        if routes["unlinked_added_photos"]:
            body.append(
                '<p class="card warning">Check these added photos separately. '
                "They have no verified cutline to another checked photo: "
                + links(routes["unlinked_added_photos"], ", ")
                + ".</p>"
            )
    restores: dict[IndexPair, dict[str, Any]] = {}
    if compact is not None:
        body.extend(
            [
                '<h3 id="compact-evidence">Compact recommendation</h3>',
                f'<p>Checked {compact["additional_pairs"]:,} additional verified overlaps. '
                f'{compact["additional_supported_pairs"]:,} added usable local evidence.</p>',
                f'<p>The full-coverage proposals selected {compact["shortlist_coverage_photos"]} '
                f'photos for {compact["shortlist_documented_cells"]:,} documented cells with the shortlist '
                f'and {compact["expanded_coverage_photos"]} photos for {summary["documented_cells"]:,} cells '
                "with the expanded search, before adding bridges.</p>",
                f'<p>The recommendation covers {summary["covered_documented_cells"]:,} of '
                f'{summary["documented_cells"]:,} documented image cells. '
                "These samples can include the same terrain in several photos. Check gaps before confirming the selection.</p>",
            ]
        )
        if len(proposed) > compact["target_photos"]:
            body.append("<p>Protected views exceed the photo target in this run.</p>")
        for section in compact["unrepresented_sections"]:
            photo: str = link(section["review_photo"])
            detail: str = (
                f'{section["documented_cells"]} documented cells.'
                if section["documented_cells"]
                else "No local coverage evidence."
            )
            body.append(
                f'<p class="card warning">Section {section["section"]} has no counting photo. '
                f"{detail} Inspect {photo} before omitting it.</p>"
            )
        restores = {tuple(item["photos"]): item for item in compact["handoff_restores"]}
    if not refined:
        body.append(
            '<p class="card warning">Check for omitted terrain and clearer views, then '
            "adjust the <strong>Use for refinement</strong> checkboxes in the launcher. "
            "Matched cells show sampled overlap evidence. Full colony coverage requires visual review.</p>"
        )
    if needs_review:
        body.append(
            f'<p><a href="#omissions">Check {len(needs_review)} omitted photos</a> for '
            "unassessed areas or terrain without a comparable selected view.</p>"
        )
    issues: list[str] = []
    for gap in plan["coverage_gaps"]:
        text: str = (
            f'Possible coverage gap · {link(gap["photo"])} has {len(gap["cells"])} '
            "sampled cells without an eligible selected view."
        )
        if "restore_photo" in gap:
            text += (
                f' Check {link(gap["restore_photo"])} to restore '
                f'{gap["restored_cells"]} of these cells.'
            )
        issues.append(text)
    for pair in plan["unresolved_handoffs"]:
        first: int
        second: int
        first, second = pair
        pair_title: str = links(pair)
        if tuple(pair) in local_pairs and tuple(pair) in export_pairs:
            issues.append(
                f'Check recovered handoff · {pair_title} · <a href="#local-pair-{first}-{second}">Review local SfM draft</a>'
            )
        else:
            preview: str = previews.get((first, second), "")
            detail: str = f' · <a href="#{preview}">Inspect join</a>' if preview else ""
            restore: dict[str, Any] | None = restores.get((first, second))
            if restore is not None and restore["restore_photos"]:
                suggestions: str = links(restore["restore_photos"], ", ")
                if restore["validated"]:
                    detail += f" · Restore {suggestions} for a validated draft route."
                else:
                    detail += f" · Inspect {suggestions} as a possible bridge."
            elif restore is not None:
                detail += " · No bridge identified."
            issues.append(f"Unresolved handoff · {pair_title}{detail}")
    if issues:
        body.append("<ul><li>" + "</li><li>".join(issues) + "</li></ul>")
    else:
        body.append("<p>No sampled coverage gaps or unresolved joins were flagged.</p>")
    if summary["triplets_needing_review"] or summary["cross_links_needing_review"]:
        body.append(
            '<p><a href="#ownership-checks">Review multi-photo ownership flags</a></p>'
        )

    if pilot is not None:
        body.append('<h2 id="shared-patches">Shared 3D patch test</h2>')
        if "model_count" in pilot:
            body.append(
                f'<p>{len(pilot["registered_photos"])} of {len(rows)} photos registered '
                f'across {pilot["model_count"]} sparse models.</p>'
            )
            if pilot["unregistered_photos"]:
                body.append(
                    '<details class="card"><summary>Unregistered photos to check</summary><p>'
                    + links(pilot["unregistered_photos"], ", ")
                    + "</p></details>"
                )
        if "block_photos" in pilot:
            body.append(
                f'<p>{escape(pilot["block_reason"])} Photos in test block · '
                + links(pilot["block_photos"], ", ")
                + "</p>"
            )
        if pilot["status"] != "complete":
            body.append(f'<p>{escape(pilot["reason"])}</p>')
        else:
            body.extend(
                [
                    f'<p>Coverage below refers to {pilot["supported_patches"]:,} supported patches '
                    "in this block. Check omitted views for terrain outside those patches. "
                    "Use the launcher checkboxes to choose photos for refinement.</p>",
                    f"<p>An eligible view must observe the patch and provide at least "
                    f'{pilot["min_relative_linear_detail"]:.0%} of its best projected detail.</p>',
                    "<table><thead><tr><th>Method</th><th>Photos</th><th>Patches covered</th>"
                    "<th>Draft joins</th><th>Unresolved joins</th></tr></thead><tbody>",
                ]
            )
            methods: tuple[tuple[str, str], ...] = (
                ("photo_local", "Photo-local recommendation"),
                ("shared_patch", "Shared 3D patches"),
            )
            for key, title in methods:
                result: dict[str, Any] = pilot[key]
                body.append(
                    f'<tr><th>{title}</th><td>{len(result["photos"])}</td>'
                    f'<td>{result["supported_patches_covered"]:,}</td>'
                    f'<td>{result["draft_handoffs"]}</td><td>{result["unresolved_handoffs"]}</td></tr>'
                )
            body.append("</tbody></table>")
            if pilot["photos_without_eligible_patch_evidence"]:
                body.append(
                    "<p>Check photos without eligible patch evidence · "
                    + links(pilot["photos_without_eligible_patch_evidence"], ", ")
                    + "</p>"
                )
            for key, title in methods:
                result = pilot[key]
                body.append(
                    f'<details class="card"><summary>{title} · photos and joins</summary>'
                    f'<p>{links(result["photos"], ", ") or "No counting photos in this block."}</p>'
                )
                for handoff in result["handoffs"]:
                    body.append(f'<h3>{links(handoff["photos"])}</h3>')
                    if "handoff" in handoff:
                        body.append(pair_preview(handoff, rows, thumbnails, frames))
                    else:
                        body.append(f'<p>{escape(handoff["handoff_error"])}</p>')
                body.append("</details>")

    body.append('<h2 id="sections">Counting set</h2>')
    displayed: list[int] = []
    for section in plan["sections"]:
        order: list[int] = section["order"]
        if not order:
            continue
        displayed.extend(order)
        body.append(f'<h3>Section {section["id"]}</h3><p>{links(order, " → ")}</p>')
        if section["handoff_walks"] != [order]:
            body.append(
                "<p>Connected handoff routes · "
                + " | ".join(links(walk, " → ") for walk in section["handoff_walks"])
                + "</p>"
            )
    if proposed:
        displayed.extend(
            index for index in plan["counting_set"] if index not in displayed
        )
        body.append(
            photo_gallery("See counting photos", list(dict.fromkeys(displayed)))
        )
    else:
        body.append("<p>Review the omitted photos before choosing a counting set.</p>")

    body.append('<h2 id="omissions">Omitted photos</h2>')
    for label, indexes in (
        ("Check for unique coverage or clearer detail", needs_review),
        ("Other candidate omissions", remaining),
        ("Analyst exclusions", excluded),
    ):
        if indexes:
            body.append(photo_gallery(label, indexes))
    if not (needs_review or remaining or excluded):
        body.append("<p>All photos are in the counting set.</p>")

    body.append(
        '<details class="card" id="ownership-checks"><summary>Additional overlap checks</summary>'
    )
    if preview_cards:
        body.append(
            "<p>Numbered landmarks and blue cells show available overlap evidence.</p>"
        )
        for group in (
            "Failed handoffs",
            "Overlapping counting views",
            "Replacement evidence",
        ):
            if preview_cards[group]:
                body.append(f"<h3>{group}</h3>")
                body.extend(preview_cards[group])
    body.append(
        f'<p>{summary["covered_documented_cells"]} of {summary["documented_cells"]} documented photo-local cells have a selected view.</p>'
    )
    items: list[str] = []
    for item in plan["cross_link_checks"]:
        a, b = item["photos"]
        pair_link: str = links(item["photos"])
        if (a, b) in previews:
            title = " / ".join(escape(rows[index]["name"]) for index in (a, b))
            pair_link = f'<a href="#{previews[(a, b)]}">{title}</a>'
        items.append(
            f'Overlapping counting views · {pair_link} · {item["duplicate_samples"]} of {item["compared_samples"]} compared landmarks have multiple proposed owners.'
        )
    for item in plan["triplet_checks"]:
        if item["needs_review"]:
            detail: str = item.get(
                "reason",
                f'{item.get("compared_samples", 0)} compared landmarks, {item.get("duplicate_samples", 0)} with multiple owners, {item.get("unassigned_samples", 0)} unassigned.',
            )
            items.append(
                "Three-view review · " + links(item["photos"]) + " · " + escape(detail)
            )
    if items:
        body.append("<ul><li>" + "</li><li>".join(items) + "</li></ul>")
    else:
        body.append(
            "<p>No cross-link or three-view ownership conflicts were flagged.</p>"
        )
    body.append("</details>")
    body.extend(
        [
            '<h2 id="local">Local SfM suggestions</h2>',
            "<p>Suggested subsets reuse overlap evidence to focus reconstruction on unresolved counting decisions.</p>",
        ]
    )
    for suggestion in plan["suggested_local_sfm"]:
        body.append(
            f'<article class="card"><h3>{escape(suggestion["recommended_action"].capitalize())}</h3>'
            f'<p>{links(suggestion["focus"])} · {escape(suggestion["reason"])}</p>'
            f'<p>Suggested views · {links(suggestion["photos"])} ({suggestion["supported_links"]} supported links)</p></article>'
        )
    if not plan["suggested_local_sfm"]:
        body.append("<p>No local model candidates were flagged.</p>")
    if local is not None:
        body.append("<h3>Local reconstruction results</h3>")
        if local.get("reason"):
            body.append(f'<p>{escape(local["reason"])}</p>')
        for block in local["blocks"]:
            body.append(
                f'<article class="card"><h3>Block {block["block"]} · {escape(block["status"])}</h3><p>{links(block["photos"])}</p>'
            )
            if block.get("reason"):
                body.append(f'<p>{escape(block["reason"])}</p>')
            for model in block.get("models", []):
                registered: str = ", ".join(
                    escape(name) for name in model["registered"]
                )
                body.append(
                    f'<p>{escape(model["name"])} registered {registered}. {model["reconstructed_tracks"]:,} reconstructed landmarks.</p>'
                )
            for check in block.get("checks", []):
                if check["status"] == "unresolved":
                    reason: str = escape(
                        check.get(
                            "reason", "No usable local model registered the pair."
                        )
                    )
                    body.append(f'<p>{links(check["photos"])} · {reason}</p>')
            body.append("</article>")
        body.extend(local_comparisons)
    page: str = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{heading}</title><style>'
        + STYLE
        + "</style></head><body>"
        + "\n".join(body)
        + f"<script>{REVIEW_NAVIGATION}</script>"
        + "</body></html>\n"
    )
    (job / "selection-plan.html").write_text(page, encoding="utf-8")
