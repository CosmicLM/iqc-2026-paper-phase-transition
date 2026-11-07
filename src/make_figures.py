"""Make every figure for the write-up and slides.

Run with: python src/make_figures.py
Writes PNGs to results/figures/. Everything is recomputed from the model,
except the scaling chart, which is redrawn from results/scaling_results.csv.

Figures
  01 tipping_map         flip year for 20 industries x 12 U.S. regions
  02 utah_pull           Utah: flip year alone vs with partner ties
  03 region_pull         20 industries in one region: alone vs with ties
  04 phase_transition    share digital vs time, for weaker and stronger ties
  05 scaling             solve time vs network size (classical methods)
  06 energy_gap          gap between the best and second-best state, per year
  07 quantum_layers      quantum P(exact answer) by year and circuit depth
  08 gap_vs_quantum      does a small energy gap mean more quantum errors?
  09 density_matrix      full 5-qubit density matrix of the 2037 quantum state
  10 reduced_density     2-qubit density matrices: tied pair vs untied pair
  11 entanglement_ramp   ground-state probability and entanglement during the ramp
  12 shot_histogram      most common measured answers, 2037
  13 regional_lockstep   neighbour ties pull regions onto the same flip year
  14 classical_vs_quantum same networks through every method: time and accuracy
  15 two_industry_levels  Healthcare-Insurance energy levels (whiteboard table)
  16 quantum_probabilities P(digital) per company per year, one qubit each (20 qubits)
  17 quantum_energy       final <H> of the quantum state vs the exact minimum
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, concurrence, partial_trace

import quantum_qaoa as qaoa
import scaling_comparison as model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

# Reference palette (light mode).
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red",
                                              ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", [BLUE_RAMP[0], BLUE_RAMP[-1]])
NEVER = "#d9d8d4"

YEARS = list(model.YEARS)
UTAH_YEAR = 2037   # Healthcare and Government flip together this year


# ---------------------------------------------------------------- helpers

def style(ax, title, xlabel=None, ylabel=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=13, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def new_figure(width=9, height=5.2, **kwargs):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE, **kwargs)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}.png")


def text_colour(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


def pressure(year):
    return model.PRESSURE_PER_YEAR * (year - YEARS[0])


def exact_flip_years(resistances, ties):
    states = {y: model.solve_min_cut(resistances, ties, pressure(y)) for y in YEARS}
    return model.flip_years(states)


def plain_region():
    """README values for one region, no regional shift or noise."""
    ties = [(a, b, model.TIE_SCALE * s) for a, b, s in model.PARTNER_STRENGTHS]
    return dict(model.RESISTANCES), ties


def all_energies(resistances, ties, A):
    """Energy of every arrangement (only for small networks)."""
    nodes = list(resistances)
    index = {n: i for i, n in enumerate(nodes)}
    count = len(nodes)
    numbers = np.arange(2 ** count, dtype=np.uint32)
    states = np.empty((2 ** count, count), dtype=np.int8)
    for i in range(count):
        states[:, i] = 2 * ((numbers >> i) & 1).astype(np.int8) - 1
    field = A - np.array([resistances[n] for n in nodes])
    energies = -(states @ field)
    for a, b, j in ties:
        energies -= j * states[:, index[a]] * states[:, index[b]]
    return energies


def energy_gap(resistances, ties, A):
    levels = np.unique(np.round(all_energies(resistances, ties, A), 9))
    return levels[1] - levels[0]


def final_state(resistances, ties, A, layers, ramp=1.0):
    start, steps, nodes = qaoa.ramp_layers(resistances, ties, A, layers, ramp)
    state = Statevector.from_label("0" * len(nodes)).evolve(start)
    for step in steps:
        state = state.evolve(step)
    return state, nodes


def p_exact(state, nodes, resistances, ties, A):
    """Probability that one shot returns a lowest-energy arrangement."""
    e_min = model.energy(model.solve_min_cut(resistances, ties, A), resistances, ties, A)
    total = 0.0
    for index, prob in enumerate(state.probabilities()):
        if prob < 1e-12:
            continue
        arrangement = {n: (1 if (index >> i) & 1 == 0 else -1) for i, n in enumerate(nodes)}
        if model.energy(arrangement, resistances, ties, A) <= e_min + 1e-9:
            total += prob
    return total


def basis_label(index, nodes, short):
    digital = [short[n] for i, n in enumerate(nodes) if (index >> i) & 1 == 0]
    return "+".join(digital) if digital else "all paper"


UTAH_SHORT = {"Healthcare": "HC", "Education": "Ed", "Government": "Gov",
              "Retail": "Ret", "Manufacturing": "Mfg"}


# ---------------------------------------------------------------- figures

def fig_tipping_map():
    resistances, ties = model.us_network(12)
    flips = exact_flip_years(resistances, ties)
    industries = list(model.RESISTANCES)
    regions = model.REGIONS
    grid = np.full((len(industries), len(regions)), np.nan)
    for (region, industry), year in flips.items():
        grid[industries.index(industry), regions.index(region)] = year
    order = np.argsort(np.nanmean(np.where(np.isnan(grid), 2041, grid), axis=1))
    grid = grid[order]
    industries = [industries[i] for i in order]

    first, last = int(np.nanmin(grid)), int(np.nanmax(grid))
    span = list(range(first, last + 1))
    picks = np.linspace(1, len(BLUE_RAMP) - 1, len(span)).round().astype(int)
    colours = [BLUE_RAMP[i] for i in picks]
    cmap = ListedColormap(colours)
    cmap.set_bad(NEVER)
    norm = BoundaryNorm(np.arange(first - 0.5, last + 1.5), cmap.N)

    fig, ax = new_figure(11, 8)
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = grid[r, c]
            label = "–" if np.isnan(value) else f"'{int(value) % 100:02d}"
            colour = NEVER if np.isnan(value) else colours[int(value) - first]
            ax.text(c, r, label, ha="center", va="center", fontsize=7.5,
                    color=text_colour(colour))
    ax.set_xticks(range(len(regions)), regions, rotation=35, ha="right")
    ax.set_yticks(range(len(industries)), industries)
    ax.set_xticks(np.arange(-0.5, len(regions)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(industries)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    style(ax, "Tipping-point map: year each industry goes digital, by region",
          grid_axis=None)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=ax, ticks=span, fraction=0.03, pad=0.02)
    bar.ax.tick_params(colors=MUTED)
    bar.outline.set_visible(False)
    bar.set_label("Flip year (exact, minimum cut)", color=MUTED)
    save(fig, "01_tipping_map")


def dumbbell(resistances, ties, title, name, short_names=None):
    alone = exact_flip_years(resistances, [])
    tied = exact_flip_years(resistances, ties)
    nodes = sorted(resistances, key=lambda n: (tied.get(n, 2041), alone.get(n, 2041)))
    fig, ax = new_figure(9, max(3.2, 0.36 * len(nodes) + 1.6))
    for row, node in enumerate(nodes):
        a, t = alone.get(node, 2041), tied.get(node, 2041)
        ax.plot([a, t], [row, row], color=GRID, linewidth=2, zorder=1)
        ax.scatter(a, row, s=60, color=SERIES[0], zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter(t, row, s=60, color=SERIES[1], zorder=3, edgecolor=SURFACE, linewidth=2)
        if a != t:
            shift = t - a
            ax.annotate(f"{shift:+d} yr", (max(a, t), row), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=9, color=MUTED)
    ax.scatter([], [], s=60, color=SERIES[0], label="On its own (no ties)")
    ax.scatter([], [], s=60, color=SERIES[1], label="With partner ties")
    ax.set_yticks(range(len(nodes)), [short_names.get(n, n) if short_names else n
                                      for n in nodes])
    ax.invert_yaxis()
    ax.set_xticks(YEARS[::2])
    ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 1.8)
    style(ax, title, "Flip year", grid_axis="x")
    fig.legend(frameon=False, loc="lower center", labelcolor=INK, ncols=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}.png")


def fig_phase_transition():
    multipliers = [(0.0, "No partner ties"), (1.0, "README ties (0.15 × strength)"),
                   (2.0, "Ties × 2"), (4.0, "Ties × 4")]
    base_res, _ = model.us_network(12)
    fine = np.linspace(0, 1.05, 106)
    fig, ax = new_figure()
    for colour, (mult, label) in zip(SERIES, multipliers):
        saved = model.TIE_SCALE
        model.TIE_SCALE = saved * mult
        resistances, ties = model.us_network(12)
        model.TIE_SCALE = saved
        if mult == 0:
            ties = [t for t in ties if t[0][0] != t[1][0]]  # keep neighbour ties only
        share = []
        for A in fine:
            state = model.solve_min_cut(resistances, ties, A)
            share.append(100 * sum(s == 1 for s in state.values()) / len(state))
        years = YEARS[0] + fine / model.PRESSURE_PER_YEAR
        ax.plot(years, share, color=colour, linewidth=2, label=label)
    ax.set_xlim(YEARS[0], YEARS[-1])
    ax.set_ylim(-2, 102)
    style(ax, "Stronger ties make the switch more sudden (240 nodes, 12 regions)",
          "Year (AI pressure rises 0.07 per year)", "Nodes digital (%)", grid_axis="both")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    save(fig, "04_phase_transition")


def fig_scaling():
    rows = []
    with (ROOT / "results" / "scaling_results.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k.endswith("seconds") or k == "nodes" else v)
                         for k, v in row.items() if v != ""})
    model.plot(rows, OUT / "05_scaling.png")
    print("  05_scaling.png")


def fig_energy_gap():
    utah_res, utah_ties = model.utah_network()
    reg_res, reg_ties = plain_region()
    fig, ax = new_figure()
    for colour, (label, res, ties) in zip(SERIES, [("Utah, 5 nodes", utah_res, utah_ties),
                                                   ("One region, 20 nodes", reg_res, reg_ties)]):
        gaps = [energy_gap(res, ties, pressure(y)) for y in YEARS]
        ax.plot(YEARS, gaps, color=colour, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.set_xticks(YEARS[::2])
    ax.set_ylim(bottom=0)
    style(ax, "Energy gap between the best and second-best answer",
          "Year", "Energy gap (smaller = easier to get wrong)", grid_axis="y")
    ax.legend(frameon=False, loc="upper center", labelcolor=INK)
    save(fig, "06_energy_gap")


def fig_quantum_layers():
    resistances, ties = model.utah_network()
    layer_counts = [1, 2, 3, 5, 10, 30]
    table = np.zeros((len(layer_counts), len(YEARS)))
    for r, layers in enumerate(layer_counts):
        for c, year in enumerate(YEARS):
            A = pressure(year)
            state, nodes = final_state(resistances, ties, A, layers)
            table[r, c] = p_exact(state, nodes, resistances, ties, A)
    fig, ax = new_figure(11, 4.4)
    image = ax.imshow(table, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            shade = SEQUENTIAL(table[r, c])
            hexed = "#%02x%02x%02x" % tuple(int(255 * x) for x in shade[:3])
            ax.text(c, r, f"{table[r, c]:.2f}", ha="center", va="center", fontsize=8,
                    color=text_colour(hexed))
    ax.set_xticks(range(len(YEARS)), YEARS, rotation=45)
    ax.set_yticks(range(len(layer_counts)), [f"{n} layer{'s' if n > 1 else ''}"
                                             for n in layer_counts])
    style(ax, "Quantum: chance one shot gives the exact answer (Utah, 5 qubits; "
              "random guess = 0.03)", "Year", grid_axis=None)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.ax.tick_params(colors=MUTED)
    bar.outline.set_visible(False)
    save(fig, "07_quantum_layers")
    return layer_counts, table


def fig_gap_vs_quantum(layer_counts, table):
    resistances, ties = model.utah_network()
    gaps = [energy_gap(resistances, ties, pressure(y)) for y in YEARS]
    row = table[layer_counts.index(3)]
    fig, ax = new_figure(8, 5.2)
    ax.scatter(gaps, row, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    for gap, p, year in zip(gaps, row, YEARS):
        if year in (2026, 2031, 2034, 2036, 2037, 2040):
            ax.annotate(str(year), (gap, p), xytext=(7, 4), textcoords="offset points",
                        fontsize=9, color=MUTED)
    corr = np.corrcoef(gaps, row)[0, 1]
    style(ax, f"Smaller gap, fewer exact answers? (3 layers, correlation r = {corr:.2f})",
          "Energy gap that year", "Chance one shot is exact", grid_axis="both")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    save(fig, "08_gap_vs_quantum")
    return corr


def annotated_matrix(ax, values, labels, limit, title):
    image = ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            if abs(v) >= 0.005:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(v) > 0.6 * limit else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    style(ax, title, grid_axis=None)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return image


def fig_density_matrix():
    resistances, ties = model.utah_network()
    A = pressure(UTAH_YEAR)
    state, nodes = final_state(resistances, ties, A, 30)
    rho = DensityMatrix(state).data
    top = np.argsort(np.real(np.diag(rho)))[::-1][:8]
    covered = np.real(np.diag(rho))[top].sum()
    sub = rho[np.ix_(top, top)]
    labels = [basis_label(i, nodes, UTAH_SHORT) for i in top]
    limit = max(np.abs(sub.real).max(), np.abs(sub.imag).max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), facecolor=SURFACE)
    image = annotated_matrix(axes[0], sub.real, labels, limit, "Real part")
    annotated_matrix(axes[1], sub.imag, labels, limit, "Imaginary part (same order)")
    axes[1].set_yticklabels([])
    bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    bar.ax.tick_params(colors=MUTED)
    bar.outline.set_visible(False)
    fig.suptitle(f"Density matrix of the quantum state, Utah {UTAH_YEAR}, after a 30-step "
                 f"ramp: the 8 most likely of 32 states ({covered:.0%} of the probability). "
                 f"Diagonal = probability of each answer", color=INK, x=0.02, ha="left",
                 fontsize=11.5)
    fig.savefig(OUT / "09_density_matrix.png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print("  09_density_matrix.png")


def reduced(state, nodes, keep):
    drop = [i for i, n in enumerate(nodes) if n not in keep]
    return partial_trace(state, drop)


def fig_reduced_density():
    resistances, ties = model.utah_network()
    A = pressure(UTAH_YEAR)
    state, nodes = final_state(resistances, ties, A, 3)
    pairs = [("Healthcare", "Government"), ("Retail", "Healthcare")]
    titles = ["Tied: Healthcare–Government", "Not tied: Retail–Healthcare"]
    rhos = [reduced(state, nodes, pair) for pair in pairs]
    limit = max(np.abs(r.data).max() for r in rhos)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), facecolor=SURFACE)
    for ax, pair, title, rho in zip(axes, pairs, titles, rhos):
        kept = [n for n in nodes if n in pair]
        # Reduced basis index: bit k belongs to kept[k]; bit 0 = digital.
        labels = []
        for idx in range(4):
            parts = [f"{UTAH_SHORT[kept[k]]} {'D' if (idx >> k) & 1 == 0 else 'P'}"
                     for k in range(2)]
            labels.append(", ".join(parts))
        image = annotated_matrix(ax, rho.data.real, labels, limit,
                                 f"{title}\nentanglement (concurrence) = {concurrence(rho):.2f}")
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    bar.ax.tick_params(colors=MUTED)
    bar.outline.set_visible(False)
    fig.suptitle(f"Two-industry density matrices (real part), Utah {UTAH_YEAR}, 3-layer "
                 f"circuit.  D = digital, P = paper", color=INK, x=0.02, ha="left",
                 fontsize=12)
    fig.savefig(OUT / "10_reduced_density.png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print("  10_reduced_density.png")


def fig_entanglement_ramp():
    resistances, ties = model.utah_network()
    A = pressure(UTAH_YEAR)
    layers = 30
    start, steps, nodes = qaoa.ramp_layers(resistances, ties, A, layers, 1.0)
    state = Statevector.from_label("0" * len(nodes)).evolve(start)
    pairs = [(("Healthcare", "Government"), "Healthcare–Government (tied)"),
             (("Government", "Education"), "Government–Education (tied)"),
             (("Retail", "Healthcare"), "Retail–Healthcare (not tied)")]
    progress, ground, ent = [0], [p_exact(state, nodes, resistances, ties, A)], \
        {label: [concurrence(reduced(state, nodes, pair))] for pair, label in pairs}
    for k, step in enumerate(steps, start=1):
        state = state.evolve(step)
        progress.append(k)
        ground.append(p_exact(state, nodes, resistances, ties, A))
        for pair, label in pairs:
            ent[label].append(concurrence(reduced(state, nodes, pair)))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                      facecolor=SURFACE)
    top.plot(progress, ground, color=SERIES[0], linewidth=2)
    top.set_ylim(0, 1)
    style(top, f"Inside the quantum ramp, Utah {UTAH_YEAR} (30 steps)", None,
          "P(exact answer)", grid_axis="y")
    for colour, (pair, label) in zip(SERIES[1:], pairs):
        bottom.plot(progress, ent[label], color=colour, linewidth=2, label=label)
    style(bottom, "Entanglement between pairs of industries", "Ramp step",
          "Concurrence", grid_axis="y")
    bottom.set_ylim(bottom=0)
    bottom.legend(frameon=False, labelcolor=INK, loc="upper right")
    save(fig, "11_entanglement_ramp")
    return max(ent[pairs[0][1]]), max(ent[pairs[2][1]])


def fig_shot_histogram():
    resistances, ties = model.utah_network()
    A = pressure(UTAH_YEAR)
    state, nodes = final_state(resistances, ties, A, 3)
    probabilities = state.probabilities()
    exact = model.solve_min_cut(resistances, ties, A)
    exact_index = sum((0 if exact[n] == 1 else 1) << i for i, n in enumerate(nodes))
    top = np.argsort(probabilities)[::-1][:8]
    labels = [basis_label(i, nodes, UTAH_SHORT) for i in top]
    colours = [SERIES[1] if i == exact_index else SERIES[0] for i in top]
    fig, ax = new_figure(10, 5.2)
    bars = ax.bar(range(len(top)), probabilities[top], color=colours, width=0.7)
    for bar, p in zip(bars, probabilities[top]):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.005, f"{p:.2f}", ha="center",
                fontsize=9, color=INK)
    ax.set_xticks(range(len(top)), labels, rotation=25, ha="right")
    ax.axhline(1 / 32, color=MUTED, linewidth=1, linestyle="--")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=SERIES[1], label="Exact answer (minimum cut)"),
               Patch(color=SERIES[0], label="Other answers"),
               Line2D([], [], color=MUTED, linewidth=1, linestyle="--",
                      label="Random guess (1/32)")]
    style(ax, f"Most likely measured answers, Utah {UTAH_YEAR} (3-layer circuit)",
          "Which industries are digital", "Probability per shot")
    ax.legend(handles=handles, frameon=False, labelcolor=INK, loc="upper right")
    save(fig, "12_shot_histogram")


def fig_regional_lockstep(industry="Legal"):
    resistances, ties = model.us_network(12)
    within = [t for t in ties if t[0][0] == t[1][0]]
    with_n, without = exact_flip_years(resistances, ties), exact_flip_years(resistances, within)
    regions = model.REGIONS
    fig, ax = new_figure(9, 5.6)
    for row, region in enumerate(regions):
        a, t = without[(region, industry)], with_n[(region, industry)]
        ax.plot([a, t], [row, row], color=GRID, linewidth=2, zorder=1)
        ax.scatter(a, row, s=60, color=SERIES[0], zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter(t, row, s=60, color=SERIES[1], zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.scatter([], [], s=60, color=SERIES[0], label="No ties between neighbouring regions")
    ax.scatter([], [], s=60, color=SERIES[1], label="Neighbour ties (J = 0.05)")
    ax.set_yticks(range(len(regions)), regions)
    ax.invert_yaxis()
    ax.set_xticks(range(2033, 2040))
    style(ax, f"{industry}: weak ties between neighbouring regions pull them into step",
          "Flip year", grid_axis="x")
    fig.legend(frameon=False, loc="lower center", labelcolor=INK, ncols=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "13_regional_lockstep.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("  13_regional_lockstep.png")


def subnetwork(count):
    """First `count` README industries (in resistance order) in one plain region."""
    resistances, ties = plain_region()
    keep = list(resistances)[:count]
    return ({n: resistances[n] for n in keep},
            [t for t in ties if t[0] in keep and t[1] in keep])


def fig_classical_vs_quantum(layers=30, shots=1000):
    """Same networks through every method: time for 15 years and flip years correct."""
    from time import perf_counter
    import neal
    sampler = neal.SimulatedAnnealingSampler()
    backend = qaoa.LocalBackend()
    sizes = [5, 8, 12, 16, 20]
    rows = []
    for n in sizes:
        resistances, ties = subnetwork(n)
        exact_states, timing, flips = {}, {}, {}

        start = perf_counter()
        exact_states = {y: model.solve_min_cut(resistances, ties, pressure(y)) for y in YEARS}
        timing["Minimum cut"] = perf_counter() - start
        flips["Minimum cut"] = exact_states

        start = perf_counter()
        states = {y: model.solve_brute_force(resistances, ties, pressure(y)) for y in YEARS}
        timing["Brute force"] = perf_counter() - start
        flips["Brute force"] = states

        start = perf_counter()
        states = {y: model.solve_annealing(resistances, ties, pressure(y), sampler)
                  for y in YEARS}
        timing["Simulated annealing"] = perf_counter() - start
        flips["Simulated annealing"] = states

        start = perf_counter()
        circuits, nodes = [], None
        for y in YEARS:
            qc, nodes = qaoa.build_circuit(resistances, ties, pressure(y), layers, 1.0)
            circuits.append(qc)
        all_counts = backend.run(circuits, shots)
        timing["Quantum (QAOA, simulated)"] = perf_counter() - start
        states, hits = {}, []
        for y, counts in zip(YEARS, all_counts):
            A = pressure(y)
            e_min = model.energy(exact_states[y], resistances, ties, A)
            best, best_e, good = None, None, 0
            for bits, c in counts.items():
                state = qaoa.to_state(bits, nodes, True)
                e = model.energy(state, resistances, ties, A)
                good += c if e <= e_min + 1e-9 else 0
                if best_e is None or e < best_e:
                    best, best_e = state, e
            states[y] = best
            hits.append(good / shots)
        flips["Quantum (QAOA, simulated)"] = states

        # Score by energy, not by matching min-cut's flip years: when two arrangements
        # tie for the lowest energy (2034 at 16 nodes), every exact method is right.
        def optimal_years(states):
            return sum(model.energy(states[y], resistances, ties, pressure(y))
                       <= model.energy(exact_states[y], resistances, ties, pressure(y)) + 1e-9
                       for y in YEARS)
        for method in timing:
            correct = optimal_years(flips[method])
            rows.append({"nodes": n, "method": method, "seconds_15_years": timing[method],
                         "years_at_lowest_energy": f"{correct}/{len(YEARS)}",
                         "quantum_mean_p_exact": (sum(hits) / len(hits)
                                                  if method.startswith("Quantum") else "")})
        print(f"    {n} nodes done")

    with (ROOT / "results" / "classical_vs_quantum.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    methods = ["Brute force", "Minimum cut", "Simulated annealing", "Quantum (QAOA, simulated)"]
    fig, ax = new_figure(9.5, 5.4)
    for colour, method in zip(SERIES, methods):
        pts = [(r["nodes"], r["seconds_15_years"]) for r in rows if r["method"] == method]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=method)
        ax.annotate(method, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlim(4, 27)
    style(ax, f"Same model, every method: time for 15 years ({layers}-layer QAOA, "
              f"{shots} shots)", "Nodes (= qubits for the quantum method)",
          "Seconds (log scale)", grid_axis="both")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    save(fig, "14_classical_vs_quantum")
    return rows


def fig_two_industry_levels():
    """The whiteboard table as energy levels: Healthcare-Insurance, all four states."""
    r = {"Healthcare": 0.75, "Insurance": 0.55}
    J = model.TIE_SCALE * 1.0
    fine = np.linspace(0, 1.0, 201)
    labels = {(-1, -1): "both paper", (-1, 1): "Healthcare paper, Insurance digital",
              (1, -1): "Healthcare digital, Insurance paper", (1, 1): "both digital"}
    fig, ax = new_figure(9.5, 5.4)
    years = YEARS[0] + fine / model.PRESSURE_PER_YEAR
    for colour, ((sh, si), label) in zip(SERIES, labels.items()):
        energy = [-(A - r["Healthcare"]) * sh - (A - r["Insurance"]) * si - J * sh * si
                  for A in fine]
        ax.plot(years, energy, color=colour, linewidth=2, label=label)
    ax.axvline(2036, color=MUTED, linewidth=1, linestyle="--")
    ax.text(2036.15, 0.62, "2036 (A = 0.70):\nwhiteboard table", fontsize=9, color=MUTED)
    for (sh, si), value in {(-1, -1): -0.05, (-1, 1): -0.05, (1, -1): 0.35,
                            (1, 1): -0.25}.items():
        ax.scatter(2036, value, s=40, color=INK, zorder=4)
    ax.set_xlim(YEARS[0], YEARS[-1])
    style(ax, "Healthcare–Insurance: energy of each state as AI pressure rises",
          "Year", "Energy (lowest wins)", grid_axis="both")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, 1.0), labelcolor=INK)
    save(fig, "15_two_industry_levels")


def quantum_marginals(resistances, ties, A, layers=30):
    """Final state of the ramp: P(digital) per company and the energy distribution.

    One qubit per company. Qubit i measured as 0 means company i is digital.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    start, steps, nodes = qaoa.ramp_layers(resistances, ties, A, layers, 1.0)
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(len(nodes))
    qc.compose(start, inplace=True)
    for step in steps:
        qc.compose(step, inplace=True)
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    amplitudes = np.asarray(sim.run(transpile(qc, sim)).result().get_statevector())
    probs = np.abs(amplitudes) ** 2
    n = len(nodes)
    # all_energies indexes arrangements with bit 1 = digital; qubits use bit 0 = digital,
    # so basis index k corresponds to arrangement 2^n - 1 - k.
    energies = all_energies(resistances, ties, A)[::-1]
    cube = probs.reshape([2] * n)          # axis 0 is qubit n-1 (most significant)
    p_digital = {}
    for i, node in enumerate(nodes):
        axis = n - 1 - i
        p_digital[node] = float(np.take(cube, 0, axis=axis).sum())
    return p_digital, float(probs @ energies), float(energies.min())


def fig_quantum_probabilities(layers=30):
    """20 industries, 20 qubits: P(digital) per company per year, and the final energy."""
    resistances, ties = plain_region()
    exact = exact_flip_years(resistances, ties)
    industries = sorted(resistances, key=lambda n: (exact.get(n, 2041), resistances[n]))
    grid = np.zeros((len(industries), len(YEARS)))
    mean_energy, ground_energy, rows = [], [], []
    for c, year in enumerate(YEARS):
        p, e_mean, e0 = quantum_marginals(resistances, ties, pressure(year), layers)
        mean_energy.append(e_mean)
        ground_energy.append(e0)
        for r, name in enumerate(industries):
            grid[r, c] = p[name]
            rows.append({"year": year, "industry": name, "p_digital": round(p[name], 6),
                         "exact_state": "digital" if exact.get(name, 9999) <= year else "paper"})
        print(f"    {year} done")
    with (ROOT / "results" / "quantum_probabilities.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = new_figure(12, 8)
    image = ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for r, name in enumerate(industries):
        for c, year in enumerate(YEARS):
            value = grid[r, c]
            shade = SEQUENTIAL(value)
            hexed = "#%02x%02x%02x" % tuple(int(255 * x) for x in shade[:3])
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color=text_colour(hexed))
        flip = exact.get(name)
        if flip:
            col = YEARS.index(flip)
            ax.add_patch(plt.Rectangle((col - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor=SERIES[1], linewidth=2))
    ax.set_xticks(range(len(YEARS)), YEARS, rotation=45)
    ax.set_yticks(range(len(industries)), industries)
    style(ax, f"P(digital) per company from the quantum state (20 qubits, {layers}-step ramp)"
              f"\nOrange box = exact flip year", grid_axis=None)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.ax.tick_params(colors=MUTED)
    bar.outline.set_visible(False)
    bar.set_label("P(measure digital)", color=MUTED)
    save(fig, "16_quantum_probabilities")

    excess = [m - g for m, g in zip(mean_energy, ground_energy)]
    fig, ax = new_figure(9.5, 5.2)
    ax.plot(YEARS, excess, color=SERIES[0], linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    peak = int(np.argmax(excess))
    ax.annotate(f"{YEARS[peak]}: {excess[peak]:.2f}", (YEARS[peak], excess[peak]),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=9,
                color=INK)
    ax.set_xticks(YEARS[::2])
    ax.set_ylim(bottom=0)
    style(ax, r"Energy above the minimum, $\langle\psi|H_C|\psi\rangle - E_0$ "
              r"(20 qubits; 0 = exact)", "Year", "Excess energy", grid_axis="y")
    save(fig, "17_quantum_energy")
    return industries, grid, mean_energy, ground_energy


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {OUT}")
    fig_tipping_map()
    utah_res, utah_ties = model.utah_network()
    dumbbell(utah_res, utah_ties, "Utah: partner ties move flip years", "02_utah_pull")
    reg_res, reg_ties = plain_region()
    dumbbell(reg_res, reg_ties, "One region, 20 industries: who pulls and who holds back",
             "03_region_pull")
    fig_phase_transition()
    fig_scaling()
    fig_energy_gap()
    layer_counts, table = fig_quantum_layers()
    corr = fig_gap_vs_quantum(layer_counts, table)
    fig_density_matrix()
    fig_reduced_density()
    tied_max, untied_max = fig_entanglement_ramp()
    fig_shot_histogram()
    fig_regional_lockstep()
    rows = fig_classical_vs_quantum()
    fig_two_industry_levels()
    fig_quantum_probabilities()
    print("\nClassical vs quantum (15 years each):")
    for r in rows:
        extra = f"  P(exact)/shot {r['quantum_mean_p_exact']:.2f}" if r["quantum_mean_p_exact"] != "" else ""
        print(f"  {r['nodes']:>3} nodes  {r['method']:<27} {r['seconds_15_years']:>9.3f} s  "
              f"lowest energy {r['years_at_lowest_energy']} years{extra}")
    print(f"\nGap vs P(exact) correlation (3 layers): r = {corr:.2f}")
    print(f"Peak concurrence during ramp: tied {tied_max:.3f}, untied {untied_max:.3f}")


if __name__ == "__main__":
    main()
