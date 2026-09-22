"""Render a compact, data-driven animation of the two V5 experiments.

The animation uses representative successful frozen missions:
Part I (individual construction, seed 1000) and Part II (collective assembly,
seed 2002). The drone drawings are schematic; all plotted gaps, trajectories,
posterior scores, proposals, and constitutive events come from the simulator.

Author: Luca M. Possati
"""

from __future__ import annotations

import argparse
from pathlib import Path
from math import cos, pi, sin

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from active_drone.collective import make_collective_system, run_collective_mission
from active_drone.simulation import make_system, run_mission
from active_drone.structure import ACTIVE, EXTERNAL, INTERNAL, SENSORY


W, H = 1000, 560
BG = "#F7F9FC"
WHITE = "#FFFFFF"
NAVY = "#102A43"
BLUE = "#1769E0"
CYAN = "#00A6A6"
GREEN = "#168B5B"
ORANGE = "#F28E2B"
RED = "#D64545"
PURPLE = "#7A5AF8"
GREY = "#66788A"
LIGHT = "#E7EDF3"
ROLE_COLORS = [PURPLE, CYAN, GREEN, ORANGE]
ROLE_NAMES = ["I", "S", "A", "E"]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)


F12, F13, F14, F16, F18, F22 = (font(s) for s in (12, 13, 14, 16, 18, 22))
B12, B14, B16, B18, B22 = (font(s, True) for s in (12, 14, 16, 18, 22))


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
             fill: str, fnt: ImageFont.FreeTypeFont) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
              text, fill=fill, font=fnt)


def rounded(draw: ImageDraw.ImageDraw, box, radius=16, fill=WHITE,
            outline=LIGHT, width=2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def drone(draw: ImageDraw.ImageDraw, x: float, y: float, color: str,
          scale: float = 1.0, label: str | None = None) -> None:
    arm = 34 * scale
    rotor = 11 * scale
    draw.line((x-arm, y-13*scale, x+arm, y+13*scale), fill=NAVY, width=max(2, int(4*scale)))
    draw.line((x-arm, y+13*scale, x+arm, y-13*scale), fill=NAVY, width=max(2, int(4*scale)))
    for rx, ry in ((x-arm, y-13*scale), (x+arm, y+13*scale),
                   (x-arm, y+13*scale), (x+arm, y-13*scale)):
        draw.ellipse((rx-rotor, ry-rotor/3, rx+rotor, ry+rotor/3), fill=GREY)
    draw.rounded_rectangle((x-20*scale, y-12*scale, x+20*scale, y+12*scale),
                           radius=int(7*scale), fill=color, outline=NAVY,
                           width=max(1, int(2*scale)))
    draw.ellipse((x-5*scale, y-5*scale, x+5*scale, y+5*scale), fill=WHITE)
    if label:
        centered(draw, (x, y+30*scale), label, NAVY, B12)


def polyline(draw: ImageDraw.ImageDraw, values: np.ndarray, box, color: str,
             upto: int, fixed_range: tuple[float, float] | None = None,
             width: int = 3) -> None:
    x0, y0, x1, y1 = box
    upto = max(1, min(upto, len(values)-1))
    vals = values[:upto+1]
    if fixed_range is None:
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi-lo < 1e-8: hi = lo+1
    else:
        lo, hi = fixed_range
    pts = []
    for i, v in enumerate(vals):
        px = x0 + (x1-x0) * i / max(len(values)-1, 1)
        py = y1 - (y1-y0) * (float(v)-lo) / max(hi-lo, 1e-9)
        pts.append((px, py))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def ring_nodes(draw: ImageDraw.ImageDraw, center: tuple[float, float],
               radii: tuple[float, float], roles: np.ndarray, matrix: np.ndarray,
               progress: float, collective: bool = False) -> None:
    cx, cy = center; rx, ry = radii
    positions = []
    for k in range(8):
        angle = -pi/2 + 2*pi*k/8
        positions.append((cx + rx*cos(angle), cy + ry*sin(angle)))

    # Direct I-E couplings are the unscreened paths that violate the blanket.
    for i in range(8):
        for j in range(i+1, 8):
            if {int(roles[i]), int(roles[j])} == {INTERNAL, EXTERNAL}:
                strength = max(abs(float(matrix[i, j])), abs(float(matrix[j, i])))
                if strength > .05:
                    draw.line((*positions[i], *positions[j]), fill=RED,
                              width=max(2, int(2+5*strength)))

    outline = GREEN if progress > .92 else (CYAN if progress > .45 else GREY)
    draw.ellipse((cx-rx-15, cy-ry-15, cx+rx+15, cy+ry+15),
                 outline=outline, width=5)
    if collective:
        draw.arc((cx-rx-22, cy-ry-22, cx+rx+22, cy+ry+22),
                 195, 345, fill=BLUE, width=3)
        draw.arc((cx-rx-22, cy-ry-22, cx+rx+22, cy+ry+22),
                 15, 165, fill=PURPLE, width=3)
    for k, (px, py) in enumerate(positions):
        color = ROLE_COLORS[int(roles[k])]
        draw.ellipse((px-11, py-11, px+11, py+11), fill=color, outline=WHITE, width=2)
        centered(draw, (px, py), ROLE_NAMES[int(roles[k])], WHITE, B12)


def legend(draw: ImageDraw.ImageDraw) -> None:
    x = 371
    for k, (name, color) in enumerate(zip(("internal", "sensory", "active", "external"), ROLE_COLORS)):
        px = x + k*67
        draw.ellipse((px, 48, px+10, 58), fill=color)
        draw.text((px+14, 46), name[0].upper(), fill=GREY, font=F12)


def individual_panel(draw: ImageDraw.ImageDraw, rec: list[dict], idx: int) -> None:
    x0, y0, x1, y1 = 18, 70, 492, 542
    rounded(draw, (x0, y0, x1, y1), radius=20)
    draw.text((36, 86), "PART I", fill=BLUE, font=B14)
    draw.text((36, 106), "Individual boundary design", fill=NAVY, font=B18)
    draw.text((36, 130), "Infer roles - select by EFE - change the causal interface",
              fill=GREY, font=F12)
    r = rec[idx]
    gaps = np.asarray([z["factorization_gap"] for z in rec], float)
    initial = max(float(gaps[0]), 1e-9)
    progress = float(np.clip(1-gaps[idx]/initial, 0, 1))
    y = float(r["y"])
    cy = 272 + np.clip(y, -1.5, 1.5)*13
    ring_nodes(draw, (250, cy), (92, 82), np.asarray(r["roles"]),
               np.asarray(r["matrix"]), progress)
    drone(draw, 250, cy, BLUE, .86, "one drone")

    changed = bool(r["design_changed_causal_matrix"])
    if changed:
        status, color = f"CONSTITUTIVE ACTION: {r['design_action']}", GREEN
    elif gaps[idx] <= .1*initial:
        status, color = "BOUNDARY CONSTRUCTED", GREEN
    elif idx < 15:
        status, color = "LEARNING PATH DYNAMICS", PURPLE
    else:
        status, color = "JOINT EFE POLICY SEARCH", BLUE
    rounded(draw, (78, 374, 422, 406), radius=12, fill="#F0F7FF", outline=color, width=2)
    centered(draw, (250, 390), status, color, B12)

    draw.text((38, 424), "Path-space gap", fill=NAVY, font=B12)
    draw.line((145, 439, 456, 439), fill=LIGHT, width=1)
    draw.line((145, 483, 456, 483), fill=LIGHT, width=1)
    polyline(draw, gaps, (145, 432, 456, 486), RED, idx, (0, max(gaps)*1.05))
    draw.text((38, 449), f"{gaps[idx]:.2f}", fill=RED, font=B16)
    draw.text((38, 485), f"role acc. {100*r['role_accuracy']:.0f}%", fill=GREY, font=F12)
    draw.text((374, 511), "seed 1000", fill=GREY, font=F12)


def collective_panel(draw: ImageDraw.ImageDraw, rec: list[dict], idx: int) -> None:
    x0, y0, x1, y1 = 508, 70, 982, 542
    rounded(draw, (x0, y0, x1, y1), radius=20)
    draw.text((526, 86), "PART II", fill=PURPLE, font=B14)
    draw.text((526, 106), "Collective boundary design", fill=NAVY, font=B18)
    draw.text((526, 130), "Pool partial evidence - agree - enact one group boundary",
              fill=GREY, font=F12)
    r = rec[idx]
    gaps = np.asarray([z["factorization_gap"] for z in rec], float)
    initial = max(float(gaps[0]), 1e-9)
    progress = float(np.clip(1-gaps[idx]/initial, 0, 1))

    # Local posteriors feed a pooled group model.
    rounded(draw, (665, 154, 825, 188), radius=12, fill="#F4F0FF", outline=PURPLE, width=2)
    centered(draw, (745, 171), "POOLED  q( M )", PURPLE, B12)
    draw.line((625, 188, 695, 205), fill=BLUE, width=3)
    draw.line((865, 188, 795, 205), fill=PURPLE, width=3)
    draw.text((536, 159), "local A", fill=BLUE, font=B12)
    draw.text((904, 159), "local B", fill=PURPLE, font=B12)

    center_y = 290 + np.clip(float(r["payload"]), -1.4, 1.4)*11
    ring_nodes(draw, (745, center_y), (163, 84), np.asarray(r["roles"]),
               np.asarray(r["matrix"]), progress, collective=True)
    y_pair = np.asarray(r["y"], float)
    drone(draw, 665, center_y+18*np.clip(y_pair[0], -1, 1), BLUE, .65, "A")
    drone(draw, 825, center_y+18*np.clip(y_pair[1], -1, 1), PURPLE, .65, "B")
    draw.line((690, center_y+4, 800, center_y+4), fill=ORANGE, width=5)
    draw.rounded_rectangle((727, center_y-5, 763, center_y+18), radius=5,
                           fill=ORANGE, outline=NAVY, width=1)

    changed = bool(r["design_changed_causal_matrix"])
    if changed:
        status, color = "BILATERAL ACTION CHANGED THE GROUP", GREEN
    elif bool(r["agreement"]):
        status, color = "MATCHING PROPOSALS - HANDSHAKE", BLUE
    elif gaps[idx] <= .1*initial:
        status, color = "GROUP BOUNDARY ACTIVE", GREEN
    elif idx < 15:
        status, color = "COMPLEMENTARY LOCAL INFERENCE", PURPLE
    else:
        status, color = "EVIDENCE POOLING + JOINT EFE", BLUE
    rounded(draw, (573, 374, 917, 406), radius=12, fill="#F8F5FF", outline=color, width=2)
    centered(draw, (745, 390), status, color, B12)

    draw.text((528, 424), "Group gap", fill=NAVY, font=B12)
    draw.line((625, 439, 946, 439), fill=LIGHT, width=1)
    draw.line((625, 483, 946, 483), fill=LIGHT, width=1)
    polyline(draw, gaps, (625, 432, 946, 486), RED, idx, (0, max(gaps)*1.05))
    draw.text((528, 449), f"{gaps[idx]:.2f}", fill=RED, font=B16)
    draw.text((528, 485), f"entropy {r['pooled_entropy']:.2f}", fill=GREY, font=F12)
    draw.text((862, 511), "seed 2002", fill=GREY, font=F12)


def render(output: Path, frames_count: int = 72) -> None:
    world_i, agent_i = make_system(1000, "constitutive", "construct")
    result_i = run_mission(world_i, agent_i)
    world_c, team_c = make_collective_system(2002, "collective_constitutive", "assemble")
    result_c = run_collective_mission(world_c, team_c)
    if not result_i.success or not result_c.success:
        raise RuntimeError("Selected representative missions must succeed")

    frames: list[Image.Image] = []
    for k in range(frames_count):
        phase = k / max(frames_count-1, 1)
        idx_i = int(round(phase*(len(result_i.records)-1)))
        idx_c = int(round(phase*(len(result_c.records)-1)))
        im = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(im)
        centered(draw, (W/2, 22), "DESIGNING THE BOUNDARY", NAVY, B22)
        centered(draw, (W/2, 45), "Actual V5 mission trajectories - constitutive Markov-blanket inversion",
                 GREY, F13)
        legend(draw)
        individual_panel(draw, result_i.records, idx_i)
        collective_panel(draw, result_c.records, idx_c)
        frames.append(im)

    # Give readers time to inspect the completed boundaries.
    frames.extend([frames[-1].copy() for _ in range(12)])
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=95,
                   loop=0, optimize=True, disposal=2)
    print({"output": str(output), "frames": len(frames),
           "individual_success": result_i.success,
           "collective_success": result_c.success,
           "individual_changes": result_i.causal_changes,
           "collective_changes": result_c.causal_changes})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("assets/experiments_overview.gif"))
    parser.add_argument("--frames", type=int, default=72)
    args = parser.parse_args()
    render(args.output, args.frames)


if __name__ == "__main__":
    main()
