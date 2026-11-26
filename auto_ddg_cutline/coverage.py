"""Assign supported sparse surface patches to single counting photos."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .colmap_model import Model, Photo, View
from .geometry import Coordinates, Pixels
from .polygons import fill_polygon, mask_polygons

FaceId: TypeAlias = tuple[int, int, int]
Edge: TypeAlias = tuple[int, int]
Projection: TypeAlias = tuple[Coordinates, float]


@dataclass(frozen=True)
class Face:
    """One canonical triangular patch and its eligible photo projections."""

    id: FaceId
    views: dict[int, Projection]


@dataclass(frozen=True)
class Cutline:
    """One physical boundary represented in both owning photos."""

    first: int
    second: int
    points: tuple[Coordinates, Coordinates]
    track_ids: tuple[int, ...]


@dataclass(frozen=True)
class Coverage:
    """Supported patches, single owners and exportable source-pixel shapes."""

    polygons: dict[int, list[Coordinates]]
    cutlines: list[Cutline]
    owner_counts: Counter[int]
    rejected: Counter[str]
    shared_tracks: Counter[Edge]
    reprojection: dict[int, tuple[int, float]]


def in_bounds(points: Coordinates, view: View) -> NDArray[np.bool_]:
    """Check positions against the closed source pixel-center rectangle."""
    bounds: Coordinates = np.array([view.photo.width - 1, view.photo.height - 1])
    return np.isfinite(points).all(axis=1) & ((points >= 0) & (points <= bounds)).all(
        axis=1
    )


def validate_tracks(
    model: Model,
) -> tuple[dict[int, dict[int, Coordinates]], dict[int, tuple[int, float]]]:
    """Keep well-reprojected tracks supported independently by multiple views."""
    observations: dict[int, dict[int, Coordinates]] = defaultdict(dict)
    diagnostics: dict[int, tuple[int, float]] = {}
    for view_id, view in model.views.items():
        ids: list[int] = [
            track_id
            for track_id in view.observations
            if track_id in model.tracks
            and view_id in model.tracks[track_id].views
            and np.isfinite(model.tracks[track_id].error)
            and model.tracks[track_id].error <= 4
            and np.isfinite(model.tracks[track_id].xyz).all()
        ]
        if not ids:
            continue
        xyz: Coordinates = np.array([model.tracks[track_id].xyz for track_id in ids])
        pixels: Coordinates
        depth: Coordinates
        pixels, depth = view.project(xyz)
        measured: Coordinates = np.array(
            [view.observations[track_id] for track_id in ids]
        )
        error: Coordinates = np.linalg.norm(pixels - measured, axis=1)
        placeable: NDArray[np.bool_] = (
            (depth > 0) & in_bounds(pixels, view) & np.isfinite(error)
        )
        if placeable.any():
            diagnostics[view_id] = (
                int(placeable.sum()),
                float(np.quantile(error[placeable], 0.90)),
            )
        for index in np.flatnonzero(placeable & (error <= 3.0)):
            observations[ids[index]][view_id] = pixels[index]
    return {
        track_id: views for track_id, views in observations.items() if len(views) >= 2
    }, diagnostics


def triangulate(
    view: View, points: dict[int, Coordinates]
) -> tuple[list[FaceId], dict[Edge, float], dict[int, float]]:
    """Triangulate tracks in one image and measure local feature spacing."""
    if len(points) < 3:
        return [], {}, {}
    subdiv: cv2.Subdiv2D = cv2.Subdiv2D((0, 0, view.photo.width, view.photo.height))
    ids: dict[tuple[float, ...], int] = {}
    for track_id, point in sorted(points.items()):
        key: tuple[float, ...] = tuple(point.astype(np.float32).tolist())
        if key not in ids:
            subdiv.insert(key)
            ids[key] = track_id
    faces: set[FaceId] = set()
    lengths: dict[Edge, float] = {}
    neighbors: dict[int, list[float]] = defaultdict(list)
    for vertices in subdiv.getTriangleList():
        keys: list[tuple[float, ...]] = [
            tuple(p) for p in vertices.reshape(3, 2).tolist()
        ]
        if any(key not in ids for key in keys):
            continue
        a: int
        b: int
        c: int
        a, b, c = sorted(ids[key] for key in keys)
        faces.add((a, b, c))
        for edge in ((a, b), (a, c), (b, c)):
            if edge not in lengths:
                length: float = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
                lengths[edge] = length
                neighbors[edge[0]].append(length)
                neighbors[edge[1]].append(length)
    # Restrict the spacing estimate to short incident edges.
    spacing: dict[int, float] = {
        track: float(np.median(sorted(distances)[:3]))
        for track, distances in neighbors.items()
    }
    return sorted(faces), lengths, spacing


def edge_samples(vertices: Coordinates, closed: bool) -> Coordinates:
    """Insert shared samples to follow radial distortion along projected edges."""
    ends: Coordinates = np.roll(vertices, -1, axis=0) if closed else vertices[1:]
    starts: Coordinates = vertices if closed else vertices[:-1]
    tiled: Coordinates = (
        starts[:, None]
        + (ends - starts)[:, None] * np.array([0, 1 / 3, 2 / 3])[None, :, None]
    ).reshape(-1, 3)
    return tiled if closed else np.vstack((tiled, vertices[-1:]))


def candidate_face(
    model: Model,
    face: FaceId,
    observations: dict[int, dict[int, Coordinates]],
    reference: int,
) -> Face | None:
    """Check depth continuity and project one face into its observing photos."""
    xyz: Coordinates = np.array([model.tracks[track].xyz for track in face])
    normal: Coordinates = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
    area: float = float(np.linalg.norm(normal))
    if not np.isfinite(area) or area <= np.finfo(float).tiny:
        return None
    normal /= area
    center: Coordinates = xyz.mean(axis=0)
    samples: Coordinates = edge_samples(xyz, closed=True)
    common: set[int] = set.intersection(*(set(observations[track]) for track in face))
    views: dict[int, Projection] = {}
    for view_id in sorted(common):
        view: View = model.views[view_id]
        pixels: Coordinates
        depth: Coordinates
        pixels, depth = view.project(samples)
        if not in_bounds(pixels, view).all() or depth.min() <= 0:
            continue
        if depth.max() / depth.min() > 1.18:
            continue
        ray: Coordinates = view.center - center
        distance: float = float(np.linalg.norm(ray))
        if distance == 0:
            continue
        incidence: float = float(abs(np.dot(normal, ray)) / distance)
        if incidence >= 0.16 and cv2.contourArea(pixels.astype(np.float32)) >= 3:
            views[view_id] = (pixels, incidence)
    if reference not in views or len(views) < 2:
        return None
    return Face(face, views)


class ScreenIndex:
    """Track accepted projected faces by image and coarse spatial cell."""

    def __init__(self) -> None:
        """Create indexes for testing projected interior conflicts."""
        self.polygons: dict[int, list[NDArray[np.float32]]] = defaultdict(list)
        self.cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    @staticmethod
    def cells_for(view_id: int, pixels: Coordinates) -> list[tuple[int, int, int]]:
        """Return fixed-size screen cells intersecting a projection's bounds."""
        xmin: np.int_
        ymin: np.int_
        xmin, ymin = np.floor(pixels.min(axis=0) / 48).astype(int)
        xmax: np.int_
        ymax: np.int_
        xmax, ymax = np.floor(pixels.max(axis=0) / 48).astype(int)
        return [
            (view_id, x, y)
            for x in range(xmin, xmax + 1)
            for y in range(ymin, ymax + 1)
        ]

    def accept(self, face: Face) -> bool:
        """Retain faces whose projected interiors agree with accepted coverage."""
        additions: list[tuple[int, NDArray[np.float32], list[tuple[int, int, int]]]] = (
            []
        )
        for view_id, (pixels, _) in face.views.items():
            # Convex triangles give an inexpensive projected conflict test.
            triangle: NDArray[np.float32] = pixels[::3].astype(np.float32)
            cells: list[tuple[int, int, int]] = self.cells_for(view_id, pixels)
            checked: set[int] = {index for cell in cells for index in self.cells[cell]}
            for index in checked:
                previous: NDArray[np.float32] = self.polygons[view_id][index]
                intersection: float
                intersection, _ = cv2.intersectConvexConvex(triangle, previous)
                if intersection > 0.65:
                    return False
            additions.append((view_id, triangle, cells))
        for view_id, triangle, cells in additions:
            index: int = len(self.polygons[view_id])
            self.polygons[view_id].append(triangle)
            for cell in cells:
                self.cells[cell].append(index)
        return True


def surface_faces(
    model: Model, observations: dict[int, dict[int, Coordinates]]
) -> tuple[list[Face], Counter[str]]:
    """Build bounded surface patches from supported reference-view charts."""
    by_image: dict[int, dict[int, Coordinates]] = defaultdict(dict)
    for track, views in observations.items():
        for view_id, xy in views.items():
            by_image[view_id][track] = xy

    def chart_score(view_id: int) -> tuple[int, str]:
        """Prefer reference views with a large and distributed track set."""
        points: Coordinates = np.array(list(by_image[view_id].values()))
        view: View = model.views[view_id]
        cells: int = len(
            np.unique(
                (points / [view.photo.width, view.photo.height] * 8).astype(int), axis=0
            )
        )
        return -len(points) * cells, view.photo.path.name.casefold()

    accepted: list[Face] = []
    accepted_ids: set[FaceId] = set()
    represented: set[int] = set()
    rejected: Counter[str] = Counter()
    index: ScreenIndex = ScreenIndex()
    for reference in sorted(by_image, key=chart_score):
        view: View = model.views[reference]
        faces: list[FaceId]
        lengths: dict[Edge, float]
        spacing: dict[int, float]
        faces, lengths, spacing = triangulate(view, by_image[reference])
        prior: set[int] = represented.copy()
        limit: float = 0.065 * max(view.photo.width, view.photo.height)
        for ids in faces:
            if ids in accepted_ids or set(ids) <= prior:
                continue
            if any(
                lengths[a, b] > min(limit, 3.5 * min(spacing[a], spacing[b]))
                for a, b in combinations(ids, 2)
            ):
                rejected["long_edge"] += 1
                continue
            face: Face | None = candidate_face(model, ids, observations, reference)
            if face is None:
                rejected["weak_visibility_or_depth"] += 1
            elif not index.accept(face):
                rejected["projected_conflict"] += 1
            else:
                accepted.append(face)
                accepted_ids.add(ids)
                represented.update(ids)
    return accepted, rejected


def face_edges(faces: list[Face]) -> dict[Edge, list[FaceId]]:
    """Index the shared physical edges of the accepted patches."""
    edges: dict[Edge, list[FaceId]] = defaultdict(list)
    for face in faces:
        for edge in combinations(face.id, 2):
            edges[edge].append(face.id)
    return edges


def assign_owners(
    model: Model, faces: list[Face], edges: dict[Edge, list[FaceId]]
) -> dict[FaceId, int]:
    """Choose one eligible photo per face and favor coherent adjacent patches."""
    scores: dict[FaceId, dict[int, float]] = {}
    for face in faces:
        areas: dict[int, float] = {
            view: cv2.contourArea(pixels.astype(np.float32))
            for view, (pixels, _) in face.views.items()
        }
        best_area: float = max(areas.values())
        scores[face.id] = {}
        for view_id, (pixels, incidence) in face.views.items():
            photo: Photo = model.views[view_id].photo
            border: Coordinates = np.array([photo.width - 1, photo.height - 1]) - pixels
            margin: float = min(float(pixels.min()), float(border.min()))
            edge_score: float = min(
                1.0, margin / (0.12 * min(photo.width, photo.height))
            )
            scores[face.id][view_id] = (
                0.56 * np.sqrt(areas[view_id] / best_area)
                + 0.24 * incidence
                + 0.20 * edge_score
            )

    neighbors: dict[FaceId, list[tuple[FaceId, float]]] = defaultdict(list)
    for (a, b), joined in edges.items():
        if len(joined) == 2:
            left: FaceId
            right: FaceId
            left, right = joined
            length: float = float(
                np.linalg.norm(model.tracks[b].xyz - model.tracks[a].xyz)
            )
            neighbors[left].append((right, length))
            neighbors[right].append((left, length))
    owners: dict[FaceId, int] = {
        face_id: min(
            options,
            key=lambda view: (
                -options[view],
                model.views[view].photo.path.name.casefold(),
            ),
        )
        for face_id, options in scores.items()
    }
    for _ in range(4):
        # In-place updates reduce alternating ownership on adjacent faces.
        for face_id, options in scores.items():
            nearby: list[tuple[FaceId, float]] = neighbors[face_id]
            total: float = sum(length for _, length in nearby) or 1.0

            def rank(view_id: int) -> tuple[float, str]:
                """Rank by view quality and adjacent ownership, then filename."""
                support: float = sum(
                    length for neighbor, length in nearby if owners[neighbor] == view_id
                )
                return (
                    -(options[view_id] + 0.36 * support / total),
                    model.views[view_id].photo.path.name.casefold(),
                )

            owners[face_id] = min(options, key=rank)
    return owners


def merge_owner(view: View, faces: list[Face]) -> tuple[list[Coordinates], set[FaceId]]:
    """Merge projected faces and recover the patch IDs represented at export."""
    mask: Pixels = np.zeros((view.photo.height, view.photo.width), dtype=np.uint8)
    for face in faces:
        fill_polygon(mask, face.views[view.id][0], 1)
    minimum: float = max(24.0, 0.00020 * view.photo.height * view.photo.width)
    polygons: list[Coordinates] = mask_polygons(mask, minimum)
    mask.fill(0)
    for polygon in polygons:
        fill_polygon(mask, polygon, 1)
    retained: set[FaceId] = set()
    for face in faces:
        x: np.int_
        y: np.int_
        x, y = np.rint(face.views[view.id][0].mean(axis=0)).astype(int)
        if mask[y, x]:
            retained.add(face.id)
    return polygons, retained


def edge_paths(edges: set[Edge]) -> list[tuple[int, ...]]:
    """Chain shared edges without branching across junctions."""
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    pending: set[Edge] = edges.copy()
    paths: list[tuple[int, ...]] = []
    starts: list[int] = sorted(
        adjacency, key=lambda vertex: (len(adjacency[vertex]) == 2, vertex)
    )
    for first in starts:
        for neighbor in sorted(adjacency[first]):
            path: list[int] = [first]
            vertex: int = neighbor
            while True:
                prev: int = path[-1]
                edge: Edge = (min(prev, vertex), max(prev, vertex))
                if edge not in pending:
                    break
                pending.remove(edge)
                path.append(vertex)
                if len(adjacency[vertex]) != 2 or vertex == first:
                    break
                vertex = next(item for item in adjacency[vertex] if item != prev)
            if len(path) > 1:
                paths.append(tuple(path))
    return paths


def shared_cutlines(
    model: Model, edges: dict[Edge, list[FaceId]], owners: dict[FaceId, int]
) -> list[Cutline]:
    """Project a common physical handoff into both count-area owners."""
    by_pair: dict[Edge, set[Edge]] = defaultdict(set)
    for edge, joined in edges.items():
        if len(joined) != 2 or any(face_id not in owners for face_id in joined):
            continue
        first: int
        second: int
        first, second = sorted(owners[face_id] for face_id in joined)
        if first != second:
            by_pair[first, second].add(edge)
    cutlines: list[Cutline] = []
    for (first, second), boundaries in sorted(by_pair.items()):
        for path in edge_paths(boundaries):
            xyz: Coordinates = np.array([model.tracks[track].xyz for track in path])
            xyz = edge_samples(xyz, closed=False)
            left: Coordinates
            left_depth: Coordinates
            left, left_depth = model.views[first].project(xyz)
            right: Coordinates
            right_depth: Coordinates
            right, right_depth = model.views[second].project(xyz)
            if (
                not in_bounds(left, model.views[first]).all()
                or not in_bounds(right, model.views[second]).all()
                or min(left_depth.min(), right_depth.min()) <= 0
            ):
                continue
            minimum: float = max(
                10.0,
                0.02
                * min(
                    model.views[first].photo.height, model.views[second].photo.height
                ),
            )
            if (
                min(
                    cv2.arcLength(poly.astype(np.float32), False)
                    for poly in (left, right)
                )
                < minimum
            ):
                continue
            cutlines.append(Cutline(first, second, (left, right), path))
    return cutlines


def build_coverage(model: Model) -> Coverage:
    """Turn a reconstructed block into partial, editable counting assignments."""
    observations: dict[int, dict[int, Coordinates]]
    reprojection: dict[int, tuple[int, float]]
    observations, reprojection = validate_tracks(model)
    pairs: Counter[Edge] = Counter(
        pair
        for views in observations.values()
        for pair in combinations(sorted(views), 2)
    )
    faces: list[Face]
    rejected: Counter[str]
    faces, rejected = surface_faces(model, observations)
    edges: dict[Edge, list[FaceId]] = face_edges(faces)
    owners: dict[FaceId, int] = assign_owners(model, faces, edges)
    by_owner: dict[int, list[Face]] = defaultdict(list)
    for face in faces:
        by_owner[owners[face.id]].append(face)
    polygons: dict[int, list[Coordinates]] = {}
    for owner, patches in sorted(by_owner.items()):
        regions: list[Coordinates]
        represented: set[FaceId]
        regions, represented = merge_owner(model.views[owner], patches)
        if regions:
            polygons[owner] = regions
        for face in patches:
            if face.id not in represented:
                del owners[face.id]
                rejected["unexportable_component"] += 1
    return Coverage(
        polygons,
        shared_cutlines(model, edges, owners),
        Counter(owners.values()),
        rejected,
        pairs,
        reprojection,
    )
