#!/usr/bin/env python3
"""
Graph-like flowchart DSL with automatic layout.

You define:
- nodes
- edges
- optional callouts / badges / text
- row assignments (main, id, ood, etc.)

The renderer computes default positions.
You can still override any node position with pos=(x, y).

Run:
    python pnc_graph_layout.py

Outputs:
    pnc_schematic.png
    pnc_schematic.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D


# -------------------------------------------------------------------
# Theme
# -------------------------------------------------------------------

THEME = {
    "colors": {
        "text": "#222222",
        "muted": "#666666",
        "arrow": "#444444",
        "input": "#d9edf7",
        "hidden": "#e8f5e9",
        "perturb": "#ffe0b2",
        "correct": "#f3e5f5",
        "output": "#fce4ec",
        "id": "#2e7d32",
        "ood": "#c62828",
        "orange": "#ef6c00",
        "purple": "#8e24aa",
        "soft_id": "#eef7ee",
        "soft_ood": "#fdeeee",
    },
    "fonts": {
        "title": 14,
        "subtitle": 9,
        "box": 9,
        "small": 8,
        "body": 9,
        "section": 10,
    },
    "line_width": 1.8,
    "rounding": 0.02,
    "figsize": (8, 2.6),
    "xlim": (0, 1),
    "ylim": (0, 1),
}


# -------------------------------------------------------------------
# Tiny DSL data structures
# -------------------------------------------------------------------

@dataclass
class Node:
    id: str
    text: str
    row: str = "main"
    style: str = "default"
    width: float = 0.14
    height: float = 0.12
    pos: tuple[float, float] | None = None
    align_x_to: str | None = None
    align_y_to: str | None = None
    dx: float = 0.0
    dy: float = 0.0
    fc: str | None = None
    ec: str = "#333333"
    lw: float | None = None
    fontsize: int | None = None
    rounding: float | None = None

@dataclass
class Edge:
    src: str
    dst: str
    color: str | None = None
    lw: float | None = None
    style: str = "-|>"
    mutation_scale: int = 14
    kind: str = "box_to_box"  # or "point_to_point"
    src_anchor: str = "right"
    dst_anchor: str = "left"
    src_offset: tuple[float, float] = (0.0, 0.0)
    dst_offset: tuple[float, float] = (0.0, 0.0)


@dataclass
class Badge:
    target: str | None = None
    xy: tuple[float, float] | None = None
    text: str = "+"
    radius: float = 0.03
    fc: str = "#ef6c00"
    ec: str = "none"
    fontsize: int = 11
    color: str = "white"
    weight: str = "bold"
    dx: float = 0.0
    dy: float = 0.0
    align_x_to: str | None = None
    align_y_to: str | None = None


@dataclass
class TextCallout:
    text: str
    xy: tuple[float, float] | None = None
    target: str | None = None
    dx: float = 0.0
    dy: float = 0.0
    fontsize: int = 11
    color: str = "#222222"
    weight: str | None = None
    ha: str = "center"
    va: str = "center"


@dataclass
class GuideLine:
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    start_node: str | None = None
    end_node: str | None = None
    start_anchor: str = "bottom"
    end_anchor: str = "top"
    color: str = "#666666"
    linestyle: str = "--"
    lw: float = 1.2
    alpha: float = 0.8
    start_offset: tuple[float, float] = (0.0, 0.0)
    end_offset: tuple[float, float] = (0.0, 0.0)

@dataclass
class BrokenEdge:
    src: str
    dst: str

    src_anchor: str = "right"
    dst_anchor: str = "left"

    src_offset: tuple[float, float] = (0.0, 0.0)
    dst_offset: tuple[float, float] = (0.0, 0.0)

    color: str | None = None
    lw: float | None = None
    style: str = "-|>"
    mutation_scale: int = 14

    # gap center can be absolute or inferred from a node
    gap_xy: tuple[float, float] | None = None
    gap_target: str | None = None
    gap_anchor: str = "top"
    gap_dx: float = 0.0
    gap_dy: float = 0.0

    # half-width of the gap along x; good default for horizontal arrows
    gap_halfwidth: float = 0.035

# -------------------------------------------------------------------
# Layout helpers
# -------------------------------------------------------------------

DEFAULT_ROW_Y = {
    "orig": 0.70,
    "pnc": 0.22,
    "id": 0.12,
    "ood": 0.12,
}
DEFAULT_ROW_Y['common'] = (DEFAULT_ROW_Y['orig'] + DEFAULT_ROW_Y['pnc']) / 2

DEFAULT_ROW_XRANGE = {
    "common": (0.05, 0.05),
    "orig": (0.22, 0.72),
    "pnc": (0.22, 0.72),
    "id": (0.18, 0.47),
    "ood": (0.72, 0.91),
}

STYLE_DEFAULTS = {
    "input": {"fc": THEME["colors"]["input"]},
    "hidden": {"fc": THEME["colors"]["hidden"]},
    "perturb": {"fc": THEME["colors"]["perturb"]},
    "correct": {"fc": THEME["colors"]["correct"]},
    "output": {"fc": THEME["colors"]["output"]},
    "id_box": {"fc": THEME["colors"]["soft_id"], "ec": THEME["colors"]["id"]},
    "ood_box": {"fc": THEME["colors"]["soft_ood"], "ec": THEME["colors"]["ood"]},
    "default": {},
}

def resolve_gap_xy(edge: BrokenEdge, node_map: dict[str, Node]) -> tuple[float, float]:
    if edge.gap_xy is not None:
        x, y = edge.gap_xy
    elif edge.gap_target is not None:
        base = node_anchor(node_map[edge.gap_target], edge.gap_anchor)
        x, y = base
    else:
        raise ValueError("BrokenEdge needs either gap_xy or gap_target")

    return (x + edge.gap_dx, y + edge.gap_dy)

def apply_style_defaults(node: Node) -> None:
    style = STYLE_DEFAULTS.get(node.style, {})
    if node.fc is None:
        node.fc = style.get("fc", "#f7f7f7")
    if node.ec == "#333333":
        node.ec = style.get("ec", "#333333")
    if node.lw is None:
        node.lw = THEME["line_width"]
    if node.fontsize is None:
        node.fontsize = THEME["fonts"]["box"]
    if node.rounding is None:
        node.rounding = THEME["rounding"]

def apply_alignment_overrides(nodes: list[Node], node_map: dict[str, Node]) -> None:
    changed = True
    for _ in range(5):  # a few passes is enough for simple dependencies
        if not changed:
            break
        changed = False
        for n in nodes:
            if n.pos is None:
                continue
            x, y = n.pos

            if n.align_x_to is not None:
                ref = node_map[n.align_x_to]
                if ref.pos is None:
                    continue
                new_x = ref.pos[0] + n.dx
                if abs(new_x - x) > 1e-9:
                    x = new_x
                    changed = True

            if n.align_y_to is not None:
                ref = node_map[n.align_y_to]
                if ref.pos is None:
                    continue
                new_y = ref.pos[1] + n.dy
                if abs(new_y - y) > 1e-9:
                    y = new_y
                    changed = True

            n.pos = (x, y)

def resolve_badge_xy(badge: Badge, node_map: dict[str, Node]) -> tuple[float, float]:
    # start from explicit xy if provided
    if badge.xy is not None:
        x, y = badge.xy

    # else attach to target node's top anchor by default
    elif badge.target is not None:
        base = node_anchor(node_map[badge.target], "top")
        x, y = base

    else:
        raise ValueError("Badge needs either xy or target")

    # then alignment overrides
    if badge.align_x_to is not None:
        ref = node_map[badge.align_x_to]
        x = ref.pos[0]

    if badge.align_y_to is not None:
        ref = node_map[badge.align_y_to]
        y = ref.pos[1]

    # finally offsets
    x += badge.dx
    y += badge.dy
    return x, y

def auto_layout(nodes: list[Node]) -> dict[str, Node]:
    node_map = {n.id: n for n in nodes}
    for n in nodes:
        apply_style_defaults(n)

    # Group by row
    rows: dict[str, list[Node]] = {}
    for n in nodes:
        rows.setdefault(n.row, []).append(n)

    # Layout each row independently unless pos already provided
    for row, row_nodes in rows.items():
        auto_nodes = [n for n in row_nodes if n.pos is None]
        if not auto_nodes:
            continue

        xmin, xmax = DEFAULT_ROW_XRANGE.get(row, (0.1, 0.9))
        y = DEFAULT_ROW_Y.get(row, 0.5)
        count = len(auto_nodes)

        if count == 1:
            xs = [(xmin + xmax) / 2]
        else:
            step = (xmax - xmin) / (count - 1)
            xs = [xmin + i * step for i in range(count)]

        for n, x in zip(auto_nodes, xs):
            n.pos = (x, y)

    apply_alignment_overrides(nodes, node_map)
    return node_map


# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------

def node_anchor(node: Node, anchor: str) -> tuple[float, float]:
    assert node.pos is not None
    x, y = node.pos
    w, h = node.width, node.height

    if anchor == "center":
        return (x, y)
    if anchor == "left":
        return (x - w / 2, y)
    if anchor == "right":
        return (x + w / 2, y)
    if anchor == "top":
        return (x, y + h / 2)
    if anchor == "bottom":
        return (x, y - h / 2)
    if anchor == "top_left":
        return (x - w / 2, y + h / 2)
    if anchor == "top_right":
        return (x + w / 2, y + h / 2)
    if anchor == "bottom_left":
        return (x - w / 2, y - h / 2)
    if anchor == "bottom_right":
        return (x + w / 2, y - h / 2)

    raise ValueError(f"Unknown anchor: {anchor}")


def offset_point(pt: tuple[float, float], dxy: tuple[float, float]) -> tuple[float, float]:
    return (pt[0] + dxy[0], pt[1] + dxy[1])


# -------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------

def draw_node(ax, node: Node) -> None:
    assert node.pos is not None
    x, y = node.pos
    patch = FancyBboxPatch(
        (x - node.width / 2, y - node.height / 2),
        node.width,
        node.height,
        boxstyle=f"round,pad=0.012,rounding_size={node.rounding}",
        facecolor=node.fc,
        edgecolor=node.ec,
        linewidth=node.lw,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        node.text,
        ha="center",
        va="center",
        fontsize=node.fontsize,
        color=THEME["colors"]["text"],
    )


def draw_edge(ax, edge: Edge, node_map: dict[str, Node]) -> None:
    src = node_map[edge.src]
    dst = node_map[edge.dst]

    start = offset_point(node_anchor(src, edge.src_anchor), edge.src_offset)
    end = offset_point(node_anchor(dst, edge.dst_anchor), edge.dst_offset)

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=edge.style,
        mutation_scale=edge.mutation_scale,
        linewidth=edge.lw or THEME["line_width"],
        color=edge.color or THEME["colors"]["arrow"],
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def draw_badge(ax, badge: Badge, node_map: dict[str, Node]) -> None:
    x, y = resolve_badge_xy(badge, node_map)
    if badge.xy is not None:
        x, y = badge.xy
    elif badge.target is not None:
        base = node_anchor(node_map[badge.target], "top")
        x, y = offset_point(base, (badge.dx, badge.dy))
    else:
        raise ValueError("Badge needs either xy or target")

    circ = Circle((x, y), radius=badge.radius, facecolor=badge.fc, edgecolor=badge.ec)
    ax.add_patch(circ)
    ax.text(
        x, y, badge.text,
        ha="center", va="center",
        fontsize=badge.fontsize,
        color=badge.color,
        fontweight=badge.weight,
    )


def draw_text_callout(ax, item: TextCallout, node_map: dict[str, Node]) -> None:
    if item.xy is not None:
        x, y = item.xy
    elif item.target is not None:
        base = node_anchor(node_map[item.target], "top")
        x, y = offset_point(base, (item.dx, item.dy))
    else:
        raise ValueError("TextCallout needs either xy or target")

    ax.text(
        x, y, item.text,
        ha=item.ha, va=item.va,
        fontsize=item.fontsize,
        color=item.color,
        fontweight=item.weight,
    )


def draw_guideline(ax, item: GuideLine, node_map: dict[str, Node]) -> None:
    if item.start is None:
        assert item.start_node is not None
        item.start = offset_point(node_anchor(node_map[item.start_node], item.start_anchor), item.start_offset)
    if item.end is None:
        assert item.end_node is not None
        item.end = offset_point(node_anchor(node_map[item.end_node], item.end_anchor), item.end_offset)

    line = Line2D(
        [item.start[0], item.end[0]],
        [item.start[1], item.end[1]],
        linestyle=item.linestyle,
        linewidth=item.lw,
        color=item.color,
        alpha=item.alpha,
    )
    ax.add_line(line)

import math
from matplotlib.patches import FancyArrowPatch

def draw_broken_edge(ax, edge: BrokenEdge, node_map: dict[str, Node]) -> None:
    src = node_map[edge.src]
    dst = node_map[edge.dst]

    start = offset_point(node_anchor(src, edge.src_anchor), edge.src_offset)
    end = offset_point(node_anchor(dst, edge.dst_anchor), edge.dst_offset)

    gx, gy = resolve_gap_xy(edge, node_map)
    hw = edge.gap_halfwidth

    # Direction of the full edge
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        raise ValueError(f"BrokenEdge {edge.src}->{edge.dst} has zero length")

    ux = dx / norm
    uy = dy / norm

    # Gap endpoints placed along the edge direction
    gap_start = (gx - hw * ux, gy - hw * uy)
    gap_end = (gx + hw * ux, gy + hw * uy)

    common = dict(
        linewidth=edge.lw or THEME["line_width"],
        color=edge.color or THEME["colors"]["arrow"],
        mutation_scale=edge.mutation_scale,
    )

    # segment 1: plain line into the gap
    seg1 = FancyArrowPatch(
        start,
        gap_start,
        arrowstyle="-",
        shrinkA=0,
        shrinkB=0,
        **common,
    )
    ax.add_patch(seg1)

    # segment 2: arrow out of the gap
    seg2 = FancyArrowPatch(
        gap_end,
        end,
        arrowstyle=edge.style,
        shrinkA=0,
        shrinkB=0,
        **common,
    )
    ax.add_patch(seg2)


def draw_delta_y_cartoon(ax, node_map):
    """Two stacked bar-charts: top shows y_hat (equal bars),
    bottom shows ||Dy||/||Dh|| (ID small, OOD large)."""
    col = THEME["colors"]

    bar_w = 0.035
    x_id, x_ood = 0.84, 0.90

    title_dy = 0.215  # fixed distance from baseline to title for both charts

    def _draw_bars(base_y, id_h, ood_h, title, show_labels=False):
        for x, h, color, label in [
            (x_id,  id_h,  col["id"],  "ID"),
            (x_ood, ood_h, col["ood"], "OOD"),
        ]:
            bar = FancyBboxPatch(
                (x - bar_w / 2, base_y), bar_w, h,
                boxstyle="round,pad=0,rounding_size=0.004",
                facecolor=color, edgecolor=color, linewidth=0.8,
                alpha=0.85,
            )
            ax.add_patch(bar)
            if show_labels:
                ax.text(x, base_y - 0.035, label,
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=color)

        bx_lo = x_id  - bar_w / 2 - 0.01
        bx_hi = x_ood + bar_w / 2 + 0.01
        ax.plot([bx_lo, bx_hi], [base_y, base_y],
                color=col["muted"], lw=0.8)

        cx = (x_id + x_ood) / 2
        ax.text(cx, base_y + title_dy, title,
                ha="center", va="center", fontsize=9, color=col["text"])

    # Top chart: equal bars
    _draw_bars(0.65, id_h=0.15, ood_h=0.15,
               title=r"$\|\Delta y\| / \|\Delta h\|$", show_labels=True)

    # Bottom chart: unequal bars
    _draw_bars(0.18, id_h=0.04, ood_h=0.15,
               title=r"$\|\Delta y\| / \|\Delta h\|$",
               show_labels=True)

def render(
    *,
    nodes: list[Node],
    edges: list[Edge] | None = None,
    badges: list[Badge] | None = None,
    texts: list[TextCallout] | None = None,
    guides: list[GuideLine] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    footer: str | None = None,
    post_draw: Any = None,
    png: str = "flowchart.png",
    pdf: str = "flowchart.pdf",
) -> None:
    node_map = auto_layout(nodes)

    fig, ax = plt.subplots(figsize=THEME["figsize"])
    ax.set_xlim(*THEME["xlim"])
    ax.set_ylim(*THEME["ylim"])
    ax.axis("off")

    if title:
        ax.text(
            0.5, 0.95, title,
            ha="center", va="center",
            fontsize=THEME["fonts"]["title"],
            fontweight="bold",
            color=THEME["colors"]["text"],
        )

    if subtitle:
        ax.text(
            0.5, 0.90, subtitle,
            ha="center", va="center",
            fontsize=THEME["fonts"]["subtitle"],
            color=THEME["colors"]["muted"],
        )

    for n in nodes:
        draw_node(ax, n)

    for e in (edges or []):
        if isinstance(e, BrokenEdge):
            draw_broken_edge(ax, e, node_map)
        else:
            draw_edge(ax, e, node_map)

    for g in (guides or []):
        draw_guideline(ax, g, node_map)

    for b in (badges or []):
        draw_badge(ax, b, node_map)

    for t in (texts or []):
        draw_text_callout(ax, t, node_map)

    if footer:
        ax.text(
            0.5, 0.05, footer,
            ha="center", va="center",
            fontsize=THEME["fonts"]["body"],
            color=THEME["colors"]["text"],
        )

    if post_draw is not None:
        post_draw(ax, node_map)

    fig.tight_layout()
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png} and {pdf}")


# -------------------------------------------------------------------
# Example: PnC schematic using the DSL
# -------------------------------------------------------------------

C = THEME["colors"]

nodes = [
    Node("x", "Input\n$x$", row="common", style="input", width=0.07, height=0.15),
    Node("h", "Original layer\n$\\theta_\\ell$", row="orig", style="hidden", width=0.17, height=0.15),
    Node("hp", "Perturbed layer\n$\\theta_\\ell + \\Delta_{perturb}$", row="pnc", style="perturb", width=0.19, height=0.15, align_x_to="h"),
    Node("h+1", "Following affine layer\n$\\theta_{\\ell+1}$", row="orig", style="hidden", width=0.22, height=0.15, fontsize=8),
    Node("corr", "Corrected affine layer\n$\\theta_{\\ell+1} + \\Delta_{correct}$", row="pnc", style="correct", width=0.22, height=0.15, align_x_to="h+1", fontsize=8),
    Node("y_orig", "Prediction\n$\\hat y$", row="orig", style="output", width=0.12, height=0.15),
    Node("y_perturb", "Prediction\n$\\hat y_{new}$", row="pnc", style="output", width=0.12, height=0.15),

    # Node(
    #     "id_absorb",
    #     "Perturbation is largely absorbed\nby the corrected affine layer",
    #     row="id",
    #     style="id_box",
    #     width=0.26,
    #     height=0.10,
    # ),
    # Node(
    #     "id_small",
    #     "Small predictive change\n$\\|\\Delta y\\|$",
    #     row="id",
    #     style="id_box",
    #     width=0.20,
    #     height=0.10,
    # ),

    # # These two are manually placed to make the right-side branch layout nicer.
    # Node(
    #     "ood_fail",
    #     "Suppression no longer\nfully transfers",
    #     row="ood",
    #     style="ood_box",
    #     width=0.22,
    #     height=0.10,
    #     pos=(0.71, 0.21),
    # ),
    # Node(
    #     "ood_large",
    #     "Larger\n$\\|\\Delta y\\|$",
    #     row="ood",
    #     style="ood_box",
    #     width=0.10,
    #     height=0.10,
    #     pos=(0.91, 0.21),
    # ),
]

edges = [
    Edge("x", "h"),
    Edge("x", "hp"),
    Edge("h", "h+1"),
    Edge("h+1", "y_orig"),
    Edge("hp", "corr"),
    Edge("corr", "y_perturb"),

    # Edge("id_absorb", "id_small", color=C["id"]),
    # Edge("ood_fail", "ood_large", color=C["ood"]),
    BrokenEdge("h", "hp", src_anchor="bottom", dst_anchor="top",
               gap_target="hp", gap_anchor="top", gap_dy=0.18,
               gap_halfwidth=0.08),
    BrokenEdge("h+1", "corr", src_anchor="bottom", dst_anchor="top",
               gap_target="corr", gap_anchor="top", gap_dy=0.18,
               gap_halfwidth=0.08)
]

badges = [
    Badge(target="hp", text="+", fc=C["orange"], radius=0.028, fontsize=13, dy=0.15, align_x_to='h'),
    Badge(target="corr", text="LS", fc=C["purple"], radius=0.034, fontsize=10, dy=0.15, align_x_to='h+1'),
]

texts = [
    TextCallout(
        target="hp",
        #dx=-0.03,
        dy=0.21,
        text="parameter perturbation",
        fontsize=10,
        color=C["orange"],
    ),
    TextCallout(
        target="corr",
        #dx=-0.03,
        dy=0.21,
        text="least-squares correction",
        fontsize=10,
        color=C["purple"],
    ),
    # TextCallout(
    #     xy=(0.18, 0.33),
    #     text="Calibration / ID inputs",
    #     fontsize=THEME["fonts"]["section"],
    #     weight="bold",
    #     color=C["id"],
    # ),
    # TextCallout(
    #     xy=(0.78, 0.33),
    #     text="Shifted / OOD inputs",
    #     fontsize=THEME["fonts"]["section"],
    #     weight="bold",
    #     color=C["ood"],
    # ),
]

guides = [
    # GuideLine(
    #     start_node="corr",
    #     end_node="id_absorb",
    #     start_anchor="bottom",
    #     end_anchor="top",
    #     color=C["id"],
    #     start_offset=(-0.01, 0.0),
    #     end_offset=(0.06, 0.0),
    # ),
    # GuideLine(
    #     start_node="corr",
    #     end_node="ood_fail",
    #     start_anchor="bottom",
    #     end_anchor="top",
    #     color=C["ood"],
    #     start_offset=(0.02, 0.0),
    #     end_offset=(0.05, 0.0),
    # ),
]

if __name__ == "__main__":
    render(
        nodes=nodes,
        edges=edges,
        badges=badges,
        texts=texts,
        guides=guides,
        title="Perturb-and-Correct (PnC)",
        # subtitle="Perturb a hidden layer, then correct in the following affine layer.",
        # footer="PnC suppresses the perturbation effect on ID data, while allowing the output to change on OOD data.",
        post_draw=draw_delta_y_cartoon,
        png="pnc_schematic.png",
        pdf="pnc_schematic.pdf",
    )