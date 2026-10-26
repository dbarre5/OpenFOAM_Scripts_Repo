#!/usr/bin/env python3
"""
Grid Convergence Index (GCI) calculator for CFD.

Implements the three-grid procedure of Celik et al. (2008),
"Procedure for Estimation and Reporting of Uncertainty Due to
Discretization in CFD Applications" (ASME J. Fluids Eng. 130(7)).

Usage:
    python gci.py                      # interactive prompts
    python gci.py --cells 4000000 1000000 250000 --dim 3 \
                  --phi 0.9702 0.9650 0.9511
    python gci.py --h 0.01 0.02 0.04 --phi 0.9702 0.9650 0.9511

Grids are ordered FINE -> MEDIUM -> COARSE (grid 1 = finest).
phi is the solution variable of interest (e.g. drag coefficient,
pressure drop, peak velocity) on each grid.
"""

import argparse
import math
import sys

FS = 1.25  # safety factor for three-grid studies (use 3.0 for two grids)


def representative_size(n_cells, dim, volume=1.0):
    """Representative cell size h = (V/N)^(1/dim)."""
    return (volume / n_cells) ** (1.0 / dim)


def apparent_order(r21, r32, e21, e32, tol=1e-10, max_iter=1000):
    """Solve for the apparent order p by fixed-point iteration."""
    ratio = e32 / e21
    s = math.copysign(1.0, ratio)

    # Constant refinement ratio -> closed form, no iteration needed
    if abs(r21 - r32) < 1e-12:
        return abs(math.log(abs(ratio))) / math.log(r21), s

    p = abs(math.log(abs(ratio))) / math.log(r21)  # initial guess
    for _ in range(max_iter):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(ratio)) + q) / math.log(r21)
        if abs(p_new - p) < tol:
            return p_new, s
        p = p_new
    raise RuntimeError("Apparent order did not converge.")


def sort_grids(h, phi):
    """Order grids fine -> coarse (smallest h first), keeping phi paired with h."""
    pairs = sorted(zip(h, phi), key=lambda t: t[0])
    h2 = [a for a, _ in pairs]
    phi2 = [b for _, b in pairs]
    if list(h) != h2:
        print("Note: grids were not ordered fine -> coarse; sorted by h "
              "(each phi kept paired with its h).")
    return h2, phi2


def gci_study(h, phi):
    h1, h2, h3 = h
    f1, f2, f3 = phi

    if not (h1 < h2 < h3):
        raise ValueError("Grids must be ordered fine -> coarse (h1 < h2 < h3).")

    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2

    if e21 == 0 or e32 == 0:
        raise ValueError("Two grids give identical results; GCI is undefined "
                         "(solution is already grid independent, or the "
                         "variable is insensitive to the mesh).")

    p, s = apparent_order(r21, r32, e21, e32)

    out = dict(r21=r21, r32=r32, e21=e21, e32=e32, p=p, s=s)
    out["oscillatory"] = (e32 / e21) < 0

    # Richardson extrapolated values
    f_ext21 = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    f_ext32 = (r32 ** p * f2 - f3) / (r32 ** p - 1)

    # Errors
    ea21 = abs((f1 - f2) / f1)
    ea32 = abs((f2 - f3) / f2)
    eext21 = abs((f_ext21 - f1) / f_ext21)
    eext32 = abs((f_ext32 - f2) / f_ext32)

    gci21 = FS * ea21 / (r21 ** p - 1)
    gci32 = FS * ea32 / (r32 ** p - 1)

    out.update(f_ext21=f_ext21, f_ext32=f_ext32, ea21=ea21, ea32=ea32,
               eext21=eext21, eext32=eext32, gci21=gci21, gci32=gci32)

    # Asymptotic range check: should be ~1.0
    out["asymptotic"] = gci32 / (r21 ** p * gci21)
    return out


def report(h, phi, res):
    pct = lambda x: f"{100 * x:.4g} %"
    print("\n" + "=" * 58)
    print(" GRID CONVERGENCE INDEX  (Celik et al. 2008)")
    print("=" * 58)
    print(f" {'Grid':<8}{'h':>14}{'phi':>18}")
    for i, (hi, fi) in enumerate(zip(h, phi), 1):
        print(f" {i:<8}{hi:>14.6g}{fi:>18.8g}")
    print("-" * 58)
    print(f" Refinement ratio r21 = {res['r21']:.4f}   r32 = {res['r32']:.4f}")
    if min(res["r21"], res["r32"]) < 1.3:
        print(" ! Warning: refinement ratio < 1.3 - results may be unreliable.")
    print(f" e21 = {res['e21']:.6g}   e32 = {res['e32']:.6g}")

    if res["oscillatory"]:
        print("\n ! Oscillatory convergence (e32/e21 < 0). The GCI below is")
        print("   not very reliable - consider a fourth grid or least-squares")
        print("   fit across more grids.")
    elif abs(res["e21"]) > abs(res["e32"]):
        print("\n ! Monotonic DIVERGENCE (|e21| > |e32|). Fine grids are not")
        print("   improving the answer; check mesh quality / iterative error.")
        print("   GCI is not meaningful for a diverging study.")
        print("=" * 58 + "\n")
        return

    p = res["p"]
    print(f"\n Apparent order p            = {p:.4f}")
    print(f" Extrapolated value (21)     = {res['f_ext21']:.8g}")
    print(f" Extrapolated value (32)     = {res['f_ext32']:.8g}")
    print(f" Approx. rel. error  e_a21   = {pct(res['ea21'])}")
    print(f" Extrap. rel. error  e_ext21 = {pct(res['eext21'])}")
    print(f" GCI fine  (21)              = {pct(res['gci21'])}")
    print(f" GCI coarse(32)              = {pct(res['gci32'])}")
    a = res["asymptotic"]
    ok = "OK" if abs(a - 1) < 0.05 else "not yet in asymptotic range"
    print(f" Asymptotic ratio            = {a:.4f}  ({ok})")
    print("-" * 58)
    print(f" Fine-grid solution: {phi[0]:.8g} +/- {100*res['gci21']:.3g} %"
          f"  (+/- {abs(phi[0]) * res['gci21']:.4g})")
    print("=" * 58 + "\n")


def make_plot(h, phi, res, path="gci_plot.png", label="Result (phi)", show=True):
    """Plot the original GCI graph and show interpretation in a separate window."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping graphic. pip install matplotlib)")
        return None

    h1, h2, h3 = h
    f1, f2, f3 = phi
    mono_ok = (not res["oscillatory"]) and abs(res["e21"]) <= abs(res["e32"])

    # ---------------------------------------------------------------
    # Assessment text for the separate interpretation window
    # ---------------------------------------------------------------
    if res["oscillatory"]:
        assessment = "Oscillatory convergence"
        explanation = (
            "The solution changes direction between successive meshes. "
            "The GCI estimate may be unreliable. Consider adding another "
            "mesh and checking iterative convergence."
        )
    elif not mono_ok:
        assessment = "Convergence not demonstrated"
        explanation = (
            "The change between the two finest meshes is larger than the "
            "change between the medium and coarse meshes. Check mesh quality "
            "and iterative convergence before interpreting the GCI."
        )
    elif abs(res["asymptotic"] - 1) < 0.05:
        assessment = "Consistent with the asymptotic range"
        explanation = (
            "The results show monotonic convergence, and the asymptotic "
            "ratio is close to one. This supports the GCI estimate, provided "
            "the solutions are also adequately converged iteratively."
        )
    else:
        assessment = "Convergence observed; asymptotic range not confirmed"
        explanation = (
            "The results show monotonic convergence, but the asymptotic "
            "ratio differs from one. Treat the GCI as an approximate "
            "uncertainty estimate; additional mesh refinement may be useful."
        )

    # ---------------------------------------------------------------
    # WINDOW 1: ORIGINAL GRAPH STYLE AND ANNOTATIONS
    # ---------------------------------------------------------------
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    BLUE, SURFACE = "#2a78d6", "#fcfcfb"

    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    fig.subplots_adjust(left=0.105, right=0.96, top=0.84, bottom=0.19)

    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK2)

    ax.tick_params(colors=INK2, labelsize=10)

    # Plot limits, including extrapolated value and GCI uncertainty.
    xmax = h3 * 1.30
    ax.set_xlim(-0.05 * xmax, xmax)

    ys = [f1, f2, f3]
    if mono_ok:
        ys += [
            res["f_ext21"],
            f1 * (1 + res["gci21"]),
            f1 * (1 - res["gci21"]),
        ]

    ylo, yhi = min(ys), max(ys)
    pad = (yhi - ylo) * 0.22 or abs(f1) * 0.05 or 1.0
    ax.set_ylim(ylo - pad, yhi + pad)

    # Richardson extrapolation curve and estimated h -> 0 value.
    if mono_ok:
        p = res["p"]
        f_ext = res["f_ext21"]
        a_coef = (f1 - f_ext) / (h1 ** p)

        hh = [i * h3 / 200 for i in range(201)]
        ax.plot(
            hh,
            [f_ext + a_coef * x**p for x in hh],
            linestyle=(0, (5, 4)),
            linewidth=1.5,
            color=INK2,
            zorder=2,
        )

        ax.plot(
            [0], [f_ext],
            marker="D",
            markersize=7,
            markerfacecolor=SURFACE,
            markeredgecolor=INK,
            markeredgewidth=1.5,
            linestyle="none",
            zorder=4,
        )

        ax.annotate(
            "Estimated answer with an\ninfinitely fine mesh = %.5g" % f_ext,
            xy=(0, f_ext),
            xytext=(14, 16),
            textcoords="offset points",
            fontsize=9.5,
            color=INK2,
            va="bottom",
            ha="left",
        )

    # Fine, medium, and coarse mesh results.
    ax.plot(
        [h1, h2, h3],
        [f1, f2, f3],
        color=BLUE,
        linewidth=2,
        zorder=3,
    )
    ax.plot(
        [h1, h2, h3],
        [f1, f2, f3],
        linestyle="none",
        marker="o",
        markersize=10,
        markerfacecolor=BLUE,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=5,
    )

    # GCI uncertainty bar on the fine-grid result.
    if mono_ok:
        ax.errorbar(
            [h1],
            [f1],
            yerr=[abs(f1) * res["gci21"]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.6,
            capsize=7,
            capthick=1.6,
            zorder=4,
        )

    # Keep the original point labels and placement logic.
    names = ["Fine mesh", "Medium mesh", "Coarse mesh"]
    pts = [(h1, f1), (h2, f2), (h3, f3)]

    xr = xmax * 1.05
    yr = (yhi - ylo) + 2 * pad

    def nd(a, b):
        dx = (b[0] - a[0]) / xr * 1.9
        dy = (b[1] - a[1]) / yr
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        return dx / n, dy / n

    candidates = [
        (14, 12, "left", "bottom"),
        (-14, 12, "right", "bottom"),
        (14, -12, "left", "top"),
        (-14, -12, "right", "top"),
    ]

    for i, (name, pt) in enumerate(zip(names, pts)):
        neighbors = [
            pts[j] for j in (i - 1, i + 1) if 0 <= j < 3
        ]

        if i == 0 and mono_ok:
            neighbors.append((0.0, res["f_ext21"]))

        directions = [nd(pt, q) for q in neighbors]

        def score(candidate):
            n = (candidate[0] ** 2 + candidate[1] ** 2) ** 0.5
            return max(
                (candidate[0] * d[0] + candidate[1] * d[1]) / n
                for d in directions
            )

        best = min(candidates, key=score)

        ax.annotate(
            "%s\n%.5g" % (name, pt[1]),
            xy=pt,
            xytext=(best[0], best[1]),
            textcoords="offset points",
            ha=best[2],
            va=best[3],
            fontsize=10,
            color=INK,
            fontweight="bold",
        )

    ax.set_xticks([0, h1, h2, h3])
    ax.set_xticklabels([
        "0\n(ideal)",
        "%.4g" % h1,
        "%.4g" % h2,
        "%.4g" % h3,
    ])

    ax.set_xlabel(
        "Cell size h      ← finer mesh  |  coarser mesh →",
        fontsize=10.5,
        color=INK,
        labelpad=8,
    )
    ax.set_ylabel(label, fontsize=10.5, color=INK, labelpad=8)

    fig.suptitle(
        "Grid convergence study (GCI, Celik et al. 2008)",
        x=0.105,
        y=0.94,
        ha="left",
        fontsize=12,
        color=INK,
    )

    # Save the graph only, suitable for inserting into a technical report.
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    print("Graph saved to %s" % path)

    # ---------------------------------------------------------------
    # WINDOW 2: EXPLANATION AND NUMERICAL SUMMARY
    # ---------------------------------------------------------------
    info = plt.figure(figsize=(9, 5.5), facecolor="white")
    info.canvas.manager.set_window_title("GCI study - interpretation")

    info.text(
        0.08, 0.92,
        "Grid convergence assessment",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )

    info.text(
        0.08, 0.85,
        assessment,
        fontsize=11,
        fontweight="bold",
        color=INK,
    )

    info.text(
        0.08, 0.79,
        explanation,
        fontsize=10,
        color=INK,
        wrap=True,
        va="top",
    )

    # Summary table.
    rows = [
        ["Fine-grid result", "%.7g" % f1],
        ["Medium-grid result", "%.7g" % f2],
        ["Coarse-grid result", "%.7g" % f3],
        ["Apparent order, p", "%.4g" % res["p"]],
        ["Refinement ratio r21", "%.4g" % res["r21"]],
        ["Refinement ratio r32", "%.4g" % res["r32"]],
        [
            "Fine-grid GCI",
            "%.4g %%" % (100 * res["gci21"])
            if mono_ok else "Not reliable",
        ],
        [
            "Richardson-extrapolated value",
            "%.7g" % res["f_ext21"]
            if mono_ok else "Not reported",
        ],
        ["Asymptotic ratio", "%.4g" % res["asymptotic"]],
    ]

    info_ax = info.add_subplot(111)
    info_ax.axis("off")
    table = info.table(
        cellText=rows,
        colLabels=["Quantity", "Value"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.43, 0.30],
        bbox=[0.08, 0.20, 0.72, 0.43],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#eeeeee")
        else:
            cell.set_facecolor("white")

    info.text(
        0.08, 0.12,
        "GCI estimates discretization uncertainty associated with finite mesh "
        "resolution. The three-grid procedure uses a safety factor of 1.25. "
        "The estimate should be interpreted alongside mesh quality and "
        "iterative convergence.",
        fontsize=8.5,
        color=INK2,
        wrap=True,
        va="top",
    )

    info.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

    if show:
        plt.show()

    return fig


def ask_floats(prompt, n):
    while True:
        raw = input(prompt)
        for ch in ",()[];":
            raw = raw.replace(ch, " ")
        raw = raw.split()
        try:
            vals = [float(x) for x in raw]
            if len(vals) == n:
                return vals
        except ValueError:
            pass
        print(f"  Please enter exactly {n} numbers.")


def interactive():
    print("GCI calculator - enter grids from FINEST to COARSEST.\n")
    mode = ""
    while mode not in ("1", "2"):
        mode = input("Grid size given as (1) cell counts or (2) h directly? [1/2]: ").strip()
    if mode == "1":
        dim = int(input("Dimension (2 or 3) [3]: ").strip() or 3)
        cells = ask_floats("Cell counts (fine medium coarse): ", 3)
        h = [representative_size(n, dim) for n in cells]
    else:
        h = ask_floats("h values (fine medium coarse): ", 3)
    phi = ask_floats("Solution values phi (fine medium coarse): ", 3)
    return h, phi


def main():
    ap = argparse.ArgumentParser(description="CFD Grid Convergence Index calculator")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cells", type=float, nargs=3, metavar=("N1", "N2", "N3"),
                   help="cell counts, fine medium coarse")
    g.add_argument("--h", type=float, nargs=3, metavar=("H1", "H2", "H3"),
                   help="representative grid sizes, fine medium coarse")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the summary graphic")
    ap.add_argument("--out", default="gci_plot.png",
                    help="PNG file for the graphic (default gci_plot.png)")
    ap.add_argument("--label", default="Result (phi)",
                    help="y-axis label, e.g. 'Drag coefficient Cd'")
    ap.add_argument("--dim", type=int, default=3, choices=(2, 3),
                    help="dimension, used with --cells (default 3)")
    ap.add_argument("--phi", type=float, nargs=3, metavar=("F1", "F2", "F3"),
                    help="solution values, fine medium coarse")
    # parse_known_args so Jupyter/IPython's own "-f kernel.json" is ignored
    args, _ = ap.parse_known_args()

    try:
        if args.phi and (args.cells or args.h):
            phi = args.phi
            h = args.h or [representative_size(n, args.dim) for n in args.cells]
        else:
            h, phi = interactive()
        h, phi = sort_grids(h, phi)
        res = gci_study(h, phi)
        report(h, phi, res)
        if not args.no_plot:
            make_plot(h, phi, res, path=args.out, label=args.label)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")   # print instead of sys.exit so Jupyter stays clean
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")


if __name__ == "__main__":
    main()
