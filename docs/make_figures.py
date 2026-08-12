"""Generate the README figures for the MLA CUDA kernels.

One figure per algorithm step, each drawn with proportional matrix boxes and the
exact dimensions of the operation, plus an overview and a results chart.

Outputs docs/*.png and docs/mla_figures.pdf.  Run:  python docs/make_figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

# --- palette -----------------------------------------------------------------
# Categorical slots 1-3 of the reference palette; that subset is documented as
# clearing the all-pairs CVD and normal-vision floors in both modes, which is the
# right gate here since any two roles can end up adjacent. Every mark also
# carries a text label, satisfying the relief rule for aqua on a light surface.
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#8a8985"
GRID    = "#e3e2de"

Q_C  = "#2a78d6"   # blue   - activations / query path
W_C  = "#eb6834"   # orange - weights
KV_C = "#1baf7a"   # aqua   - KV cache and things derived from it
NEUT = "#6f6e6a"   # neutral- intermediates

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})

OUT = os.path.dirname(os.path.abspath(__file__))
# Box WIDTHS are always exactly proportional to the column count. Heights follow the
# same scale until they exceed MAXH, at which point the box is drawn truncated with a
# break marker -- some of these tensors are 16,384 rows against 512 columns and cannot
# be drawn to scale on a page. Exact dimensions are printed under every box.
MINT = 1.4          # minimum thickness, so a 1-row strip stays visible
MAXH = 21.0         # height cap; past this the box is drawn with a break
SMAX = 0.030        # cap on units-per-column, so small-dim figures stay sane


# --- primitives --------------------------------------------------------------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100.0 * h / w)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, color, alpha=0.16, lw=1.8, z=2, radius=0.45):
    for fc, a, zz in ((color, alpha, z), ("none", 1.0, z + 1)):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor=color, facecolor=fc, alpha=a, zorder=zz))


def step_header(ax, H, tag, formula, note=None):
    ax.text(3, H - 3.6, tag, fontsize=13, fontweight="bold", va="top", color=INK)
    ax.text(3, H - 9.0, formula, fontsize=10.5, va="top", color=INK)
    if note:
        ax.text(3, H - 13.6, note, fontsize=8.6, va="top", color=INK2)


def box_size(rows, cols, s):
    """Rows and columns share one scale, so a contracted dimension is drawn the same
    length on both operands -- q_lat's 1,536 columns match W_q_b's 1,536 rows. Only a
    box past MAXH breaks that, and it is labelled truncated when it does."""
    w = cols * s
    raw = rows * s
    return w, min(max(raw, MINT), MAXH), raw > MAXH


def matrix(ax, x, ymid, rows, cols, s, color, name, alpha=0.16,
           tiles=None, tile_lbl=None):
    """Width is exactly proportional to `cols`; height too, until it hits MAXH."""
    w, h, cut = box_size(rows, cols, s)
    y = ymid - h / 2
    rbox(ax, x, y, w, h, color, alpha=alpha)

    if tiles:
        tc, tr = tiles
        nx = max(int(round(cols / tc)), 1)
        ny = max(int(round(rows / tr)), 1)
        if 1 < nx <= 24:
            for i in range(1, nx):
                ax.plot([x + i * w / nx] * 2, [y, y + h], color=color, lw=0.8,
                        ls=(0, (2.5, 2)), zorder=5)
        if 1 < ny <= 24:
            for j in range(1, ny):
                ax.plot([x, x + w], [y + j * h / ny] * 2, color=color, lw=0.8,
                        ls=(0, (2.5, 2)), zorder=5)
        rbox(ax, x, y + h - h / min(ny, 24), w / min(nx, 24), h / min(ny, 24),
             color, alpha=0.5, lw=1.6, z=6, radius=0.2)

    if cut:   # rows truncated: break marks on both edges
        for xx in (x + w * 0.02, x + w * 0.98):
            ax.text(xx, ymid, "⋮", ha="center", va="center", fontsize=11,
                    color=color, zorder=8)

    ax.text(x + w / 2, y + h + 1.5, name, ha="center", va="bottom", fontsize=8.8,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y - 1.6, f"{rows:,}  x  {cols:,}", ha="center", va="top",
            fontsize=8.2, color=INK2)
    extra = None
    if cut:
        extra = "(rows truncated to fit)"
    elif h == MINT and rows * s < MINT:
        extra = f"({rows} rows: a thin strip)"
    if extra:
        ax.text(x + w / 2, y - 4.3, extra, ha="center", va="top", fontsize=7.2,
                color=MUTED, style="italic")
    if tile_lbl:
        ax.text(x + w / 2, y - (7.0 if extra else 4.3), tile_lbl, ha="center",
                va="top", fontsize=7.4, color=color)
    return x, x + w, h


def op(ax, x, ymid, sym):
    ax.text(x, ymid, sym, ha="center", va="center", fontsize=15, color=INK2)


HEAD, PAD_T, PAD_B, FOOT = 19.0, 5.0, 10.5, 6.0


def gemm_step(fname, width_in, tag, formula, note, A, B, C, footer, avail_w=88.0):
    """A @ B = C with proportional boxes. A=(name,M,K,color), B=(name,K,N,color),
    C=(name,M,N,color[,tiles,tile_lbl]). The canvas height follows the tallest box,
    so nothing can collide with the header."""
    tot_cols = A[2] + B[2] + C[2]
    gaps = 2 * 7.0                     # two operator gaps
    s = min((avail_w - gaps) / tot_cols, SMAX)
    maxh = max(box_size(M[1], M[2], s)[1] for M in (A, B, C))

    H = (HEAD if note else HEAD - 5.0) + PAD_T + maxh + PAD_B + FOOT
    fig, ax = canvas(width_in, width_in * H / 100.0)
    step_header(ax, H, tag, formula, note)
    ymid = H - HEAD - PAD_T - maxh / 2.0

    group_w = tot_cols * s + gaps
    x = max(5.0, (100.0 - group_w) / 2.0)
    for i, M in enumerate((A, B, C)):
        name, rows, cols, color = M[0], M[1], M[2], M[3]
        tiles = M[4] if len(M) > 4 else None
        tlbl = M[5] if len(M) > 5 else None
        _, xr, _ = matrix(ax, x, ymid, rows, cols, s, color, name, tiles=tiles,
                          tile_lbl=tlbl)
        if i < 2:
            op(ax, xr + 3.5, ymid, "@" if i == 0 else "=")
        x = xr + 7.0

    ax.text(3, 2.0, footer, fontsize=8.4, va="bottom", color=INK2)
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=190, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.14)
    return fig


# =============================================================================
# OVERVIEW 1 — what MLA is
# =============================================================================
def eq_row(fig, ax, ymid, parts, fontsize, color):
    """Lay mathtext `parts` out left-to-right as separate Text objects, then shift the
    whole run to be horizontally centred. Returns one (left, right) x-span per part, in
    data coords, so callouts can be anchored under the term they actually describe
    instead of at a hand-tuned x that drifts whenever the formula is edited."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    texts, spans, x = [], [], 0.0
    for s in parts:
        t = ax.text(x, ymid, s, fontsize=fontsize, va="center", ha="left", color=color)
        bb = t.get_window_extent(renderer=rend)
        (xl, _), (xr, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        texts.append(t)
        spans.append([xl, xr])
        x = xr
    shift = (100.0 - (spans[-1][1] - spans[0][0])) / 2.0 - spans[0][0]
    for t, sp in zip(texts, spans):
        t.set_x(t.get_position()[0] + shift)
        sp[0] += shift
        sp[1] += shift
    return spans


def fig_idea():
    fig, ax = canvas(11, 7.5)
    H = 100 * 7.5 / 11
    ax.text(3, H - 3.6, "Why MLA: one shared latent instead of 128 K/V pairs",
            fontsize=14, fontweight="bold", va="top", color=INK)

    ax.text(3, H - 11.0, "KV cache per token", fontsize=10.2, fontweight="bold",
            va="top", color=INK)
    ax.text(3, H - 14.5,
            "Both bars count the values cached for ONE key position — a single index of Sk, "
            "shared by all 128 heads and by every one of the Sq queries.",
            fontsize=8.4, va="top", color=INK2)
    span = 62.0
    ax.text(6, H - 21.0, "standard MHA   —   128 heads x (192 K + 128 V)  =  40,960 values "
                         "per Sk position", fontsize=8.6, va="bottom", color=INK)
    rbox(ax, 6, H - 24.9, span, 3.4, W_C, radius=0.3)
    mla_w = span * 576 / 40960
    rbox(ax, 6, H - 31.0, mla_w, 3.4, KV_C, alpha=0.9, lw=2.0, radius=0.12)
    ax.add_patch(FancyArrowPatch((6 + mla_w + 0.5, H - 29.3), (13.0, H - 29.3),
                                 arrowstyle="-|>", mutation_scale=7, lw=1.2,
                                 color=KV_C, shrinkA=0, shrinkB=0))
    ax.text(13.8, H - 29.3,
            r"MLA   —   $c_{kv}$ (512)  +  $pe$ (64)  =  576 values per Sk position",
            fontsize=8.6, va="center", color=INK)

    ax.text(3, H - 37.0, "The absorbed path: move the per-head matrix onto the query side",
            fontsize=10.2, fontweight="bold", va="top", color=INK)
    ax.text(50, H - 41.3,
            r"$q_{nope}[h]$ : Sq x 128          $c_{kv}$ : Sk x 512          "
            r"$W_{uk}[h]$ : 128 x 512          so   $K_{nope}[h]=c_{kv}W_{uk}[h]^{\!\top}$ : Sk x 128",
            ha="center", va="top", fontsize=8.4, color=INK2)
    spans = eq_row(fig, ax, H - 49.0, [
        r"$q_{nope}[h]\cdot K_{nope}[h]\;=\;$",
        r"$q_{nope}[h]\cdot\left(c_{kv}W_{uk}[h]^{\!\top}\right)$",
        r"$\;=\;$",
        r"$\left(q_{nope}[h]\,W_{uk}[h]\right)\cdot c_{kv}$",
    ], fontsize=13, color=INK)

    for (xl, xr), color, label in (
        (spans[1], W_C, "reconstruct K for all 128 heads\n"
                        "Sk x 128 x 128 values\ngrows with the cache"),
        (spans[3], Q_C, "absorb into the query, ONCE\n"
                        "Sq x 128 x 512 values\nSq = 1 in decode"),
    ):
        xc = (xl + xr) / 2.0
        ax.plot([xc, xc], [H - 53.0, H - 54.3], color=color, lw=1.4)
        ax.text(xc, H - 54.9, label, ha="center", va="top", fontsize=7.8, color=color)

    ax.text(3, 1.5,
            "c_kv and pe_cache carry no head index: all 128 heads of a query read the SAME cache rows, "
            "so kernel 3a gives one block all 128 heads at once.",
            fontsize=8.4, va="bottom", color=INK2)
    fig.savefig(os.path.join(OUT, "fig0_why_mla.png"), dpi=190, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.14)
    return fig


# =============================================================================
# OVERVIEW 2 — the 9 steps at a glance
# =============================================================================
def fig_pipeline():
    fig, ax = canvas(15.0, 6.0)
    H = 100 * 6.0 / 15.0
    ax.text(3, H - 2.6, "The 9 steps", fontsize=14, fontweight="bold", va="top",
            color=INK)
    ax.text(3, H - 6.8,
            "Kernel 2 builds the 576-wide query; kernel 3 does the attention.   "
            "Shapes for batch-128 decode:  bq = B.Sq = 128 rows,  "
            "flat = B.Sq.H = 16,384 rows,  Sk = 4,989.   "
            "Every operand is listed by name; outputs are marked ->.",
            fontsize=8.6, va="top", color=INK2)

    # tag, the operation on the matrices, operands in, operands out, colour.
    # Every matrix in a product is listed separately, with its own dimensions.
    # A line starting with "=" breaks the dimension above it down into the factors
    # the next step splits it along, so the chain stays readable across reshapes.
    steps = [
        ("2a1",  "x @ W_q_a",
         ["x  128x7168", "W_q_a  7168x1536"], ["q_lat  128x1536"], Q_C),
        ("2a-n", "RMSNorm(q_lat)",
         ["q_lat  128x1536"], ["q_lat  128x1536"], NEUT),
        ("2a2",  "q_lat @ W_q_b",
         ["q_lat  128x1536", "W_q_b  1536x24576"],
         ["q_raw  128x24576", "= 128 x 128h x 192"], Q_C),
        ("2b",   "RoPE on q[128:192]",
         ["q_raw  128x128x192", "= 128 nope + 64 rope"],
         ["q_rope  128x128x64"], NEUT),
        ("2c",   "q_nope @ W_uk[h]",
         ["q_nope  128x128x128", "W_uk[h]  128x512"],
         ["q_absorbed  128x128x512"], Q_C),
        ("3a",   "q @ cache^T, then exp",
         ["q  16384x576", "= (128x128h) x (512+64)", "cache^T  576x4989"],
         ["scores  16384x4989", "m,l  16384x39"], KV_C),
        ("3b",   "max, then exp(m-M)",
         ["m,l  16384x39"], ["alpha  16384x39", "row_sum  16384"], NEUT),
        ("3c",   "p @ c_kv",
         ["p  16384x4989", "c_kv  4989x512"], ["ctx  16384x512"], KV_C),
        ("3d",   "ctx @ W_uv[h]^T",
         ["ctx  16384x512", "W_uv[h]^T  512x128"], ["out  16384x128"], Q_C),
    ]
    w, gap = 9.3, 1.2
    x0 = 3.0
    ybox, hbox = 15.0, 7.6
    for i, (n, o, d_in, d_out, c) in enumerate(steps):
        x = x0 + i * (w + gap)
        xc = x + w / 2.0
        rbox(ax, x, ybox, w, hbox, c, radius=0.6)
        ax.text(xc, ybox + 4.6, n, ha="center", fontsize=9.6, fontweight="bold",
                color=INK)
        ax.text(xc, ybox + 1.7, o, ha="center", fontsize=6.4, color=INK2)
        y = ybox - 1.9
        for lbl in d_in:
            brk = lbl.startswith("=")
            ax.text(xc, y, lbl, ha="center", fontsize=5.7 if brk else 6.0,
                    color=MUTED, style="italic" if brk else "normal")
            y -= 2.0 if brk else 2.3
        for lbl in d_out:
            brk = lbl.startswith("=")
            ax.text(xc, y, lbl if brk else "-> " + lbl, ha="center",
                    fontsize=5.7 if brk else 6.0, color=MUTED if brk else c,
                    style="italic" if brk else "normal")
            y -= 2.0 if brk else 2.3
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.2, ybox + hbox / 2.0),
                                         (x - 0.2, ybox + hbox / 2.0),
                                         arrowstyle="-|>", mutation_scale=7,
                                         lw=1.3, color=MUTED, shrinkA=0, shrinkB=0))
    yr = ybox + hbox + 3.0
    ax.plot([x0, x0 + 5 * (w + gap) - gap], [yr, yr], color=Q_C, lw=2.0)
    ax.text(x0, yr + 1.0, "KERNEL 2  —  Q path", fontsize=8.8, fontweight="bold",
            color=Q_C, va="bottom")
    ax.plot([x0 + 5 * (w + gap), x0 + 9 * (w + gap) - gap], [yr, yr], color=KV_C,
            lw=2.0)
    ax.text(x0 + 5 * (w + gap), yr + 1.0, "KERNEL 3  —  attention", fontsize=8.8,
            fontweight="bold", color=KV_C, va="bottom")

    fig.savefig(os.path.join(OUT, "fig0_pipeline.png"), dpi=190, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.14)
    return fig


# =============================================================================
# THE STEPS
# =============================================================================
def steps():
    figs = []

    # ---- 2a1 -------------------------------------------------------------
    figs.append(gemm_step(
        "step_2a1", 11,
        "Step 2a1   —   project the hidden state into the q_lora latent",
        "q_lat  =  x @ W_q_a",
        "first half of the factorised Q projection.   7,168 = hidden.   1,536 = q_lora_rank.",
        ("x", 128, 7168, Q_C),
        ("W_q_a", 7168, 1536, W_C),
        ("q_lat", 128, 1536, NEUT, (128, 128), "grid.x = 12 col tiles"),
        "grid.x -> 12 output-column tiles of 128   |   grid.y -> 1 row tile (bq = 128)   |   "
        "grid.z -> split-K: the 7,168 reduction is cut into slices\n"
        "Only 12 column tiles means 12 blocks on an 84-SM GPU — split-K is what makes this step use the machine at all."))

    # ---- 2a-norm ---------------------------------------------------------
    fig, ax = canvas(11, 4.0)
    H = 100 * 4.0 / 11
    step_header(ax, H, "Step 2a-norm   —   RMSNorm on the latent",
                "q_lat  :=  q_lat * rsqrt(mean(q_lat^2) + eps) * gain",
                "row-wise; keeps the two matmuls from collapsing into one rank-1536 matrix")
    s = 60.0 / 1536
    matrix(ax, 6, 11.0, 128, 1536, s, NEUT, "q_lat  (in and out)")
    ax.text(6 + 60 + 6, 11.0,
            "one CUDA block per row  ->  128 blocks\n"
            "256 threads cover 1,536 elements (6 each)\n"
            "each warp of 32 shuffles down to one number\n"
            "warp 0 then sums those 8 numbers into the total",
            va="center", fontsize=8.4, color=INK2)
    fig.savefig(os.path.join(OUT, "step_2an.png"), dpi=190, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.14)
    figs.append(fig)

    # ---- 2a2 -------------------------------------------------------------
    figs.append(gemm_step(
        "step_2a2", 11,
        "Step 2a2   —   expand the latent into the per-head query",
        "q_raw  =  q_lat @ W_q_b            (24,576 = 128 heads x 192)",
        None,
        ("q_lat", 128, 1536, NEUT),
        ("W_q_b", 1536, 24576, W_C),
        ("q_raw", 128, 24576, Q_C, (128, 128), "grid.x = 192 col tiles"),
        "grid.x -> 192 output-column tiles of 128   |   grid.y -> 1 row tile   |   grid.z -> split-K over the 1,536 reduction\n"
        "192 blocks against 84 SMs x 2 resident = 168 slots is 1.14 waves: one full wave, then a tail wave leaving 144 slots idle."))

    # ---- 2b --------------------------------------------------------------
    fig, ax = canvas(11, 4.4)
    H = 100 * 4.4 / 11
    step_header(ax, H, "Step 2b   —   RoPE on the rope half of each head",
                "q_rope[b,s,h,:]  =  rotate( q_raw[b,s,h, 128:192], position[s] )",
                "elementwise; rotates adjacent pairs by an angle set by the query's ABSOLUTE position")
    # a single head's 192 values, split 128 | 64 — widths exactly proportional
    s = 66.0 / 192
    x0, ymid, h = 8, 14.5, 7.0
    w = 192 * s
    rbox(ax, x0, ymid - h / 2, 128 * s, h, NEUT, alpha=0.13)
    rbox(ax, x0 + 128 * s, ymid - h / 2, 64 * s, h, KV_C, alpha=0.30)
    ax.text(x0 + 64 * s, ymid, "nope  [0:128]   ->  step 2c", ha="center",
            va="center", fontsize=8.4, color=INK)
    ax.text(x0 + 128 * s + 32 * s, ymid, "rope [128:192]\nrotated here",
            ha="center", va="center", fontsize=7.8, color=INK)
    ax.text(x0 + w / 2, ymid + h / 2 + 1.5, "q_raw, ONE head:  192 values",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=INK)
    ax.text(x0 + w / 2, ymid - h / 2 - 1.6,
            "over all rows and heads:  [128 x 128 x 192]   ->   q_rope [128, 128, 64]",
            ha="center", va="top", fontsize=8.2, color=INK2)
    ax.text(3, 2.0,
            "One thread per (row, head, adjacent pair), grid-stride capped at 32 x SM so the launch never scales with the cache depth. "
            "0.1% of decode.",
            fontsize=8.4, va="bottom", color=INK2)
    fig.savefig(os.path.join(OUT, "step_2b.png"), dpi=190, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.14)
    figs.append(fig)

    # ---- 2c --------------------------------------------------------------
    figs.append(gemm_step(
        "step_2c", 11,
        "Step 2c   —   absorb W_uk into the query  (once per head)",
        "q_absorbed[:,h]  =  q_raw[:,h, 0:128] @ W_uk[h]        repeated for h = 0 .. 127",
        "this is the absorption: it widens the query 128 -> 512 so the cache can be read in its stored latent form",
        ("q_nope[:,h]", 128, 128, Q_C),
        ("W_uk[h]", 128, 512, W_C),
        ("q_absorbed[:,h]", 128, 512, Q_C, (128, 128), "grid.y = 4 rank tiles"),
        "grid.x -> bq row tiles   |   grid.y -> 4 tiles of 128 over kv_rank=512   |   grid.z -> the head index, 0..127\n"
        "One head per block means W_uk[h] (262 KB) is staged into shared memory once and reused by every row in the tile."))

    # ---- 3a --------------------------------------------------------------
    figs.append(gemm_step(
        "step_3a", 11,
        "Step 3a   —   scores, then the local softmax numerator",
        "scores  =  ( q_absorbed . c_kv  +  q_rope . pe_cache ) / sqrt(192)        then store exp(s - m_j)",
        "the two dot products (512 and 64 deep) are run as ONE 576-deep reduction; shown at Sk = 4,989",
        ("query [576 wide]", 16384, 576, Q_C),
        ("cache^T", 576, 4989, KV_C),
        ("scores", 16384, 4989, KV_C, (128, 128), "128 x 128 per block"),
        "grid.x -> flat = B.Sq.H / 128 = 128 tiles   |   grid.y -> Sk / 128 = 39 tiles   |   no grid.z (scores has a real Sk axis)\n"
        "A block's 128 rows are the 128 heads of ONE query, so the cache tile it stages is reused by all 128 of them.",
        ))

    # ---- 3b --------------------------------------------------------------
    Hb = HEAD + PAD_T + MAXH + PAD_B + FOOT
    fig, ax = canvas(11, 11 * Hb / 100.0)
    H = Hb
    step_header(ax, H, "Step 3b   —   reconcile the per-block softmax maxima",
                "M = max_j m_j        alpha_j = exp(m_j - M)        row_sum = sum_j  l_j * alpha_j",
                "3a's grid.y split each row across 39 blocks, none of which could see the global maximum")
    ymid = H - HEAD - PAD_T - MAXH / 2.0
    sb = 0.30
    matrix(ax, 8, ymid, 16384, 39, sb, NEUT,
           "m_part / l_part")
    xa = 8 + 39 * sb
    ax.add_patch(FancyArrowPatch((xa + 2.0, ymid), (xa + 8.0, ymid),
                                 arrowstyle="-|>", mutation_scale=8, lw=1.5,
                                 color=MUTED, shrinkA=0, shrinkB=0))
    matrix(ax, xa + 10.0, ymid, 16384, 1, 2.2, KV_C, "row_sum")
    ax.text(xa + 17.0, ymid,
            "one CUDA block per row  ->  16,384 blocks\n"
            "256 threads stride the 39 partials of that row\n"
            "two block-wide reductions: max, then a weighted sum\n"
            "also writes alpha [16,384 x 39], which step 3c consumes",
            va="center", fontsize=8.6, color=INK2)
    ax.text(3, 2.0,
            "Exact because exp(s - m_j) * exp(m_j - M) = exp(s - M) termwise.  Total storage 7.7 MB against a 327 MB scores tensor; "
            "costs 0.5% of decode.",
            fontsize=8.4, va="bottom", color=INK2)
    fig.savefig(os.path.join(OUT, "step_3b.png"), dpi=190, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.14)
    figs.append(fig)

    # ---- 3c --------------------------------------------------------------
    figs.append(gemm_step(
        "step_3c", 11,
        "Step 3c   —   context: the attention-weighted sum of the cache",
        "ctx  =  ( exp(s - m_j) * alpha_j ) @ c_kv        =  exp(s - M) @ c_kv,  unnormalised",
        "alpha_j is folded into the staging load, so what lands in shared memory is already the softmax numerator",
        ("scores", 16384, 4989, KV_C),
        ("c_kv", 4989, 512, KV_C),
        ("ctx", 16384, 512, NEUT, (128, 128), "grid.y = 4 rank tiles"),
        "grid.x -> flat / 128 = 128 tiles   |   grid.y -> 4 tiles over kv_rank = 512   |   grid.z -> split-K over Sk\n"
        "ctx has no Sk axis, so at B=Sq=1 the output alone yields 4 blocks and ~95% of the SMs would idle — hence split-K here.",
        ))

    # ---- 3d --------------------------------------------------------------
    figs.append(gemm_step(
        "step_3d", 11,
        "Step 3d   —   reconstruct V and project out  (once per head)",
        "out[:,h]  =  ( ctx[:,h] / row_sum ) @ W_uv[h]^T        repeated for h = 0 .. 127",
        "the deferred softmax divide rides along on a staging load this kernel was already doing",
        ("ctx[:,h] / row_sum", 128, 512, NEUT),
        ("W_uv[h]^T", 512, 128, W_C),
        ("out[:,h]", 128, 128, Q_C, (128, 128), None),
        "grid.x -> bq row tiles   |   grid.y -> 1 tile over v_dim = 128   |   grid.z -> the head index, 0..127\n"
        "Narrows 512 -> 128, undoing the widening that step 2c did. One head per block, same reuse argument as 2c."))

    return figs


# =============================================================================
# RESULTS
# =============================================================================
def fig_results():
    # ordered worst-to-best speedup, so the bars read as a ramp
    data = [
        ("decode_speculative_2tok",      2.59,   3.38),
        ("decode_single_user_long_ctx",  1.91,   2.39),
        ("prefill_chat_batch",          70.55,  87.32),
        ("decode_speculative_4tok",      6.16,   7.53),
        ("prefill_chunk_2k",            94.15, 113.11),
        ("decode_serving_avg_ctx",      14.27,  16.92),
        ("prefill_chunk_4k",           355.46, 417.61),
        ("decode_serving_high_batch",   12.91,  15.06),
        ("decode_speculative_long_ctx", 47.19,  54.90),
        ("decode_serving_long_ctx",     23.61,  26.55),
    ]
    fig = plt.figure(figsize=(10, 5.8))
    ax = fig.add_axes([0.30, 0.17, 0.64, 0.68])
    ax.set_facecolor(SURFACE)
    names = [d[0] for d in data]
    sp = [d[1] / d[2] for d in data]
    ypos = list(range(len(data)))[::-1]
    for yy, s, n in zip(ypos, sp, names):
        c = KV_C if n.startswith("decode") else Q_C
        ax.barh(yy, s, height=0.6, color=c, alpha=0.85, edgecolor=c, linewidth=1.4,
                zorder=3)
        ax.text(s + 0.012, yy, f"{s:.2f}x", va="center", ha="left", fontsize=8.5,
                color=INK, fontweight="bold", zorder=4)
    ax.axvline(1.0, color=INK2, lw=1.6, zorder=5)
    ax.text(1.005, -1.35, "1.00x = PyTorch / cuBLAS baseline", fontsize=8.4,
            color=INK2, va="center", ha="left")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8.4, color=INK)
    ax.set_ylim(-0.75, len(data) - 0.25)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25x", "0.50x", "0.75x", "1.00x"], fontsize=8.4,
                       color=INK2)
    ax.set_xlabel("speedup vs baseline   (higher is better)", fontsize=9, color=INK2,
                  labelpad=8)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.03, 0.955, "End-to-end result: 0.83x geometric mean", fontsize=14.5,
             fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.895,
             "FP32, RTX A6000, L2 flushed between iterations, TF32 disabled on both sides.  "
             "All 10 scenarios pass correctness.", fontsize=8.6, color=INK2, va="top")
    fig.text(0.30, 0.025,
             "Decode (aqua) and prefill (blue) land in the same narrow band: the tiling "
             "generalises, but cuBLAS keeps a ~17% edge.", fontsize=8.2, color=INK2)
    fig.savefig(os.path.join(OUT, "fig_results.png"), dpi=190, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.14)
    return fig


def main():
    figs = [fig_idea(), fig_pipeline()] + steps() + [fig_results()]
    with PdfPages(os.path.join(OUT, "mla_figures.pdf")) as pdf:
        for f in figs:
            pdf.savefig(f, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    for f in figs:
        plt.close(f)
    print(f"wrote {len(figs)} figures + mla_figures.pdf")


if __name__ == "__main__":
    main()
