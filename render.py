import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ============================================================
# OUTPUT
# ============================================================
OUT = "ARDS_Normal_vs_ARDS_StressStrain_60s.mp4"

# ============================================================
# VIDEO
# ============================================================
FPS = 12
DURATION_S = 60
W, H = 1280, 720

# ============================================================
# BREATH SHAPE (VCV + inspiratory hold -> plateau)
# ============================================================
T_CYCLE = 3.5          # seconds per breath
INSP_FRAC = 0.35
PAUSE_FRAC = 0.12
EXP_FRAC = 1 - INSP_FRAC - PAUSE_FRAC

# ============================================================
# PHYSIO MODELS (simplified)
# - Normal: higher compliance, higher FRC
# - ARDS: low compliance, "baby lung" low FRC
# ============================================================
NORMAL = {
    "name": "Pulmão Normal",
    "C": 60.0,        # mL/cmH2O
    "FRC": 2500.0,    # mL
    "R": 10.0,        # cmH2O/(L/s)
    "color": "navy",
}

ARDS = {
    "name": "ARDS (Baby Lung)",
    "C": 25.0,        # mL/cmH2O
    "FRC": 1000.0,    # mL
    "R": 10.0,        # cmH2O/(L/s)
    "color": "maroon",
}

# ============================================================
# Targets / thresholds (modern conservative teaching points)
# ============================================================
DP_SAFE = 14.0       # cmH2O
DP_WARN = 18.0
STRAIN_SAFE = 0.25
STRAIN_WARN = 0.35
PPLAT_SAFE = 28.0

def risk_color(val, g, y):
    """Return 'green'|'orange'|'red' based on thresholds."""
    if val <= g:
        return "green"
    if val <= y:
        return "orange"
    return "red"

def one_cycle_waveforms(VT, PEEP, C, R, n_frames):
    """
    Generate one breath cycle for VCV with inspiratory pause:
    - V(t) in mL, Flow(t) in L/s, Paw(t) in cmH2O.
    Simplified equation:
      Paw = PEEP + V/C + Flow*R
    """
    t = np.linspace(0, T_CYCLE, n_frames, endpoint=False)
    ph = (t % T_CYCLE) / T_CYCLE

    V = np.zeros_like(ph)        # mL
    Flow = np.zeros_like(ph)     # L/s

    insp = ph < INSP_FRAC
    pause = (ph >= INSP_FRAC) & (ph < INSP_FRAC + PAUSE_FRAC)
    exp = ph >= (INSP_FRAC + PAUSE_FRAC)

    insp_time = INSP_FRAC * T_CYCLE
    const_flow = (VT/1000.0) / insp_time  # L/s

    # Inspiration: linear volume increase, constant flow
    V[insp] = VT * (ph[insp] / INSP_FRAC)
    Flow[insp] = const_flow

    # Pause: volume hold, flow zero
    V[pause] = VT
    Flow[pause] = 0.0

    # Expiration: exponential decay of volume
    exp_phase = (ph[exp] - (INSP_FRAC + PAUSE_FRAC)) / EXP_FRAC
    tau = 0.35
    V[exp] = VT * np.exp(-exp_phase / tau)

    # Flow from derivative
    dVdt = np.gradient(V, t, edge_order=1)     # mL/s
    Flow = dVdt / 1000.0                       # L/s

    Pel = V / C
    Pres = Flow * R
    Paw = PEEP + Pel + Pres

    # Summary values (useful for labels)
    Pplat = PEEP + (VT / C)
    Ppeak = Pplat + (const_flow * R)
    DP = Pplat - PEEP

    return t, V, Flow, Paw, Ppeak, Pplat, DP

def canvas_to_rgb(fig):
    """Robust matplotlib (Agg) frame capture for modern versions."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

# ============================================================
# Scenario timeline (60 s):
# 0-15s: Normal ventilated with safe settings
# 15-30s: ARDS with SAME settings (shows high strain/stress)
# 30-55s: Stepwise optimization for ARDS (VT↓ then PEEP↑ then VT↓)
# 55-60s: Freeze summary
# ============================================================

def settings_at_time(t_sec):
    """
    Returns (VT, PEEP) for both Normal and ARDS at time t_sec.
    Normal stays stable. ARDS changes stepwise after 30s.
    """
    # Baseline settings
    VT_base = 450  # mL
    PEEP_base = 6  # cmH2O

    # Normal is always baseline (teaching contrast)
    vt_n, peep_n = VT_base, PEEP_base

    # ARDS:
    if t_sec < 30:
        # same as normal to show danger
        vt_a, peep_a = VT_base, PEEP_base
    else:
        # stepwise optimization over 30-55 seconds
        # segments:
        # 30-38: reduce VT to 380
        # 38-46: increase PEEP to 10
        # 46-55: reduce VT to 330 (final protective)
        if t_sec < 38:
            vt_a = int(np.round(np.interp(t_sec, [30, 38], [VT_base, 380])))
            peep_a = PEEP_base
        elif t_sec < 46:
            vt_a = 380
            peep_a = float(np.interp(t_sec, [38, 46], [PEEP_base, 10]))
        else:
            vt_a = int(np.round(np.interp(t_sec, [46, 55], [380, 330])))
            peep_a = 10.0

    return (vt_n, peep_n), (vt_a, peep_a)

# ============================================================
# Render
# ============================================================
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "28"]
)

frames_total = DURATION_S * FPS
frames_per_cycle = int(T_CYCLE * FPS)

# Precompute one-cycle time base for indexing convenience
cycle_idx = np.arange(frames_per_cycle)

for frame_i in range(frames_total):
    t_sec = frame_i / FPS

    # Freeze summary in last 5 seconds
    freeze_mode = (t_sec >= 55)

    # Settings
    (VT_n, PEEP_n), (VT_a, PEEP_a) = settings_at_time(t_sec)

    # Get per-cycle waveforms for each lung type with current settings
    _, Vn, Fn, Pawn, Ppeak_n, Pplat_n, DP_n = one_cycle_waveforms(VT_n, PEEP_n, NORMAL["C"], NORMAL["R"], frames_per_cycle)
    _, Va, Fa, Pawa, Ppeak_a, Pplat_a, DP_a = one_cycle_waveforms(VT_a, PEEP_a, ARDS["C"], ARDS["R"], frames_per_cycle)

    k = frame_i % frames_per_cycle

    # Dynamic strain(t): V(t)/FRC
    strain_n_t = Vn[k] / NORMAL["FRC"]
    strain_a_t = Va[k] / ARDS["FRC"]

    # Peak strain (end inspiration): VT/FRC
    strain_n_peak = VT_n / NORMAL["FRC"]
    strain_a_peak = VT_a / ARDS["FRC"]

    # Dynamic stress(t) proxy: (Paw - PEEP) at that moment
    # (didactic proxy for transpulmonary load; plateau/ΔP shown separately)
    stress_n_t = max(Pawn[k] - PEEP_n, 0.0)
    stress_a_t = max(Pawa[k] - PEEP_a, 0.0)

    # Risk colors based on peak strain and ΔP (teaching targets)
    strain_col_n = risk_color(strain_n_peak, STRAIN_SAFE, STRAIN_WARN)
    strain_col_a = risk_color(strain_a_peak, STRAIN_SAFE, STRAIN_WARN)
    dp_col_n = risk_color(DP_n, DP_SAFE, DP_WARN)
    dp_col_a = risk_color(DP_a, DP_SAFE, DP_WARN)

    # ========================================================
    # Layout (2 columns): Normal vs ARDS
    # Each column:
    # - Lung schematic (top)
    # - Two vertical meters (stress(t), strain(t)) in the middle
    # Right side: ventilator waveforms shared (pressure/volume/flow)
    # ========================================================
    fig.clf()

    ax_left = plt.subplot2grid((3, 3), (0, 0), rowspan=3)  # Normal panel
    ax_mid  = plt.subplot2grid((3, 3), (0, 1), rowspan=3)  # ARDS panel
    axP     = plt.subplot2grid((3, 3), (0, 2))
    axV     = plt.subplot2grid((3, 3), (1, 2))
    axF     = plt.subplot2grid((3, 3), (2, 2))

    # -------------------- Helper to draw a panel --------------------
    def draw_panel(ax, title, V_inst, VT, PEEP, Ppeak, Pplat, DP, strain_t, strain_peak, stress_t,
                   strain_col, dp_col):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.05, 0.95, title, fontsize=13, weight="bold")
        ax.text(0.05, 0.90, f"VT={int(VT)} mL | PEEP={PEEP:.0f}", fontsize=11)
        ax.text(0.05, 0.86, f"Ppeak≈{Ppeak:.0f} | Pplat≈{Pplat:.0f} | ΔP≈{DP:.0f}", fontsize=11, color=dp_col)

        # Lung schematic size linked to current volume fraction
        size = 0.18 + 0.18 * (V_inst / max(VT, 1))
        ax.add_patch(plt.Circle((0.50, 0.70), size, fill=False, linewidth=6, color=strain_col))

        # Vertical meters area
        # Stress meter (left)
        # Strain meter (right)
        ax.text(0.17, 0.48, "STRESS(t)", fontsize=10, weight="bold", ha="center")
        ax.text(0.70, 0.48, "STRAIN(t)", fontsize=10, weight="bold", ha="center")

        # Meter frames
        ax.add_patch(plt.Rectangle((0.10, 0.10), 0.14, 0.35, fill=False, linewidth=2))
        ax.add_patch(plt.Rectangle((0.63, 0.10), 0.14, 0.35, fill=False, linewidth=2))

        # Scale caps (didactic)
        # Stress scale: 0 to 25 cmH2O above PEEP (proxy)
        stress_norm = min(stress_t / 25.0, 1.0)
        # Strain scale: 0 to 0.6 (display range)
        strain_norm = min(strain_t / 0.6, 1.0)

        # Fill bars (dynamic)
        ax.add_patch(plt.Rectangle((0.10, 0.10), 0.14, 0.35 * stress_norm, color=dp_col, alpha=0.90))
        ax.add_patch(plt.Rectangle((0.63, 0.10), 0.14, 0.35 * strain_norm, color=strain_col, alpha=0.90))

        # Numeric readouts
        ax.text(0.17, 0.07, f"{stress_t:.0f}", fontsize=10, ha="center", color=dp_col)
        ax.text(0.70, 0.07, f"{strain_t:.2f}", fontsize=10, ha="center", color=strain_col)

        # Peak strain callout
        ax.text(0.05, 0.55, f"Strain pico≈{strain_peak:.2f}", fontsize=10, color=strain_col)

        # Teaching note
        ax.text(0.05, 0.02, "Verde/laranja/vermelho = risco mecânico", fontsize=9)

    # Panels
    draw_panel(
        ax_left,
        "Pulmão Normal",
        Vn[k], VT_n, PEEP_n, Ppeak_n, Pplat_n, DP_n,
        strain_n_t, strain_n_peak, stress_n_t,
        strain_col_n, dp_col_n
    )

    draw_panel(
        ax_mid,
        "ARDS (Baby Lung)",
        Va[k], VT_a, PEEP_a, Ppeak_a, Pplat_a, DP_a,
        strain_a_t, strain_a_peak, stress_a_t,
        strain_col_a, dp_col_a
    )

    # -------------------- Waveforms (right column) --------------------
    # show last ~2 cycles for readability
    hist = min(frame_i + 1, 2 * frames_per_cycle)
    x = np.linspace(0, hist / FPS, hist)

    def history(arr):
        return np.array([arr[(k - j) % frames_per_cycle] for j in reversed(range(hist))])

    Paw_hist_n = history(Pawn)
    Paw_hist_a = history(Pawa)

    V_hist_n = history(Vn)
    V_hist_a = history(Va)

    F_hist_n = history(Fn) * 60
    F_hist_a = history(Fa) * 60

    # Pressure
    axP.plot(x, Paw_hist_n, lw=2.0, color="navy", label="Paw Normal")
    axP.plot(x, Paw_hist_a, lw=2.0, color="maroon", label="Paw ARDS")
    axP.axhline(PEEP_a, ls="--", color="gray", lw=1.2, label="PEEP (ARDS)")
    axP.axhline(Pplat_a, ls="--", color="maroon", lw=1.2, label="Pplat (ARDS)")

    # Shade ΔP for ARDS
    axP.fill_between([0, x.max() if len(x) else 1], PEEP_a, Pplat_a, color=dp_col_a, alpha=0.12, label="ΔP (ARDS)")

    axP.set_ylim(0, 45)
    axP.set_ylabel("Pressão (cmH₂O)")
    axP.grid(True, alpha=0.25)
    axP.set_title("VCV: Pico + Plateau | Driving Pressure (ΔP)")
    axP.legend(loc="upper right", fontsize=7, frameon=True)

    # Volume
    axV.plot(x, V_hist_n, lw=1.8, color="navy")
    axV.plot(x, V_hist_a, lw=1.8, color="maroon")
    axV.set_ylim(0, 650)
    axV.set_ylabel("Volume (mL)")
    axV.grid(True, alpha=0.25)
    axV.set_title("Volume")

    # Flow
    axF.plot(x, F_hist_n, lw=1.8, color="navy")
    axF.plot(x, F_hist_a, lw=1.8, color="maroon")
    axF.axhline(0, color="gray", lw=1.0)
    axF.set_ylim(-80, 80)
    axF.set_ylabel("Fluxo (L/min)")
    axF.set_xlabel("Tempo (s)")
    axF.grid(True, alpha=0.25)
    axF.set_title("Fluxo")

    # -------------------- Top banners / didactic stage --------------------
    if t_sec < 15:
        stage = "FASE 1: Pulmão Normal (referência)"
    elif t_sec < 30:
        stage = "FASE 2: ARDS com os mesmos settings (stress/strain sobem)"
    elif t_sec < 55:
        stage = "FASE 3: Optimização stepwise (VT↓, PEEP↑, ΔP↓)"
    else:
        stage = "SÍNTESE: Alvo = Pplat < 28 | ΔP < 14 | Strain pico < 0.25"

    fig.suptitle(stage, fontsize=14, weight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Frame capture
    rgb = canvas_to_rgb(fig)

    # Freeze summary for last 5 seconds (hold frame)
    if freeze_mode:
        # keep same frame; still write
        writer.append_data(rgb)
    else:
        writer.append_data(rgb)

writer.close()
print("OK ->", OUT)
