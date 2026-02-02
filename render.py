import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# -----------------------------
# OUTPUT
# -----------------------------
OUT = "ARDS_Stepwise_Long_Final_ICU.mp4"

# -----------------------------
# VIDEO SETTINGS
# -----------------------------
fps = 12
freeze_seconds = 1.0

# -----------------------------
# PHYSIO / MODEL (ARDS)
# -----------------------------
FRC = 1000  # mL (baby lung, simplificado)
R = 10      # cmH2O/(L/s)  (resistência global)
C = 25      # mL/cmH2O     (complacência ARDS)

# Targets conservadores modernos (didático)
DP_SAFE = 14
PPLAT_SAFE = 28
STRAIN_SAFE = 0.25

# -----------------------------
# BREATH SHAPE (VCV + INSP PAUSE)
# -----------------------------
T = 3.5
insp_frac = 0.35
pause_frac = 0.12
exp_frac = 1 - insp_frac - pause_frac

# -----------------------------
# STEPWISE (ARDS alto -> protegido)
# -----------------------------
steps = [
    {"VT": 500, "PEEP": 6,  "label": "ARDS NÃO PROTEGIDO", "sec": 10},
    {"VT": 420, "PEEP": 6,  "label": "STEP 1 — VT ↓",      "sec": 10},
    {"VT": 420, "PEEP": 10, "label": "STEP 2 — PEEP ↑",    "sec": 10},
    {"VT": 360, "PEEP": 10, "label": "STEP 3 — PROTETOR",  "sec": 10},
]

def risk_color(val, g, y):
    if val <= g:
        return "green"
    if val <= y:
        return "orange"
    return "red"

def one_cycle_waveforms(VT, PEEP, n_frames):
    """Return one breath cycle arrays for VCV with inspiratory hold:
       V (mL), Flow (L/s), Paw (cmH2O), plus Ppeak/Pplat/DP."""
    t = np.linspace(0, T, n_frames, endpoint=False)
    ph = (t % T) / T

    V = np.zeros_like(ph)     # mL
    Flow = np.zeros_like(ph)  # L/s

    insp = ph < insp_frac
    pause = (ph >= insp_frac) & (ph < insp_frac + pause_frac)
    exp = ph >= (insp_frac + pause_frac)

    insp_time = insp_frac * T
    const_flow = (VT / 1000.0) / insp_time  # L/s

    # Inspiration (constant flow -> linear volume)
    V[insp] = VT * (ph[insp] / insp_frac)
    Flow[insp] = const_flow

    # Pause (hold)
    V[pause] = VT
    Flow[pause] = 0.0

    # Expiration (simple exponential decay)
    exp_phase = (ph[exp] - (insp_frac + pause_frac)) / exp_frac
    tau = 0.35
    V[exp] = VT * np.exp(-exp_phase / tau)

    # Flow from derivative of V
    dVdt = np.gradient(V, t, edge_order=1)  # mL/s
    Flow = dVdt / 1000.0                    # L/s

    # Pressures
    Pel = V / C                 # elastic
    Pres = Flow * R             # resistive
    Paw = PEEP + Pel + Pres

    Pplat = PEEP + (VT / C)
    Ppeak = PEEP + (VT / C) + (const_flow * R)
    DP = Pplat - PEEP

    return t, V, Flow, Paw, Ppeak, Pplat, DP

# -----------------------------
# WRITE VIDEO
# -----------------------------
W, H = 1280, 720
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)

writer = imageio.get_writer(
    OUT,
    fps=fps,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "28"]
)

for s in steps:
    VT = s["VT"]
    PEEP = s["PEEP"]
    label = s["label"]
    sec = s["sec"]

    phase_frames = int(sec * fps)
    cyc_frames = int(T * fps)

    tc, Vc, Fc, Pawc, Ppeak, Pplat, DP = one_cycle_waveforms(VT, PEEP, cyc_frames)

    strain_peak = VT / FRC
    strain_col = risk_color(strain_peak, 0.25, 0.35)
    dp_col = risk_color(DP, 14, 18)

    # Phase animation
    for i in range(phase_frames):
        fig.clf()

        axL = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
        axP = plt.subplot2grid((3, 2), (0, 1))
        axV = plt.subplot2grid((3, 2), (1, 1))
        axF = plt.subplot2grid((3, 2), (2, 1))

        k = i % cyc_frames

        # ---------------- Left panel: Lung + bars ----------------
        # Lung size linked to instantaneous volume (didático)
        size = 0.20 + 0.16 * (Vc[k] / max(VT, 1))
        axL.add_patch(plt.Circle((0.5, 0.72), size, fill=False, linewidth=7, color=strain_col))

        # Dynamic strain(t) and stress(t)
        strain_t = Vc[k] / FRC
        # dynamic stress proxy: load above PEEP (didático)
        stress_t = max(Pawc[k] - PEEP, 0.0)

        axL.text(0.05, 0.93, label, fontsize=16, weight="bold")
        axL.text(0.05, 0.86, f"VT={VT} mL | PEEP={PEEP} cmH₂O", fontsize=12)
        axL.text(0.05, 0.80, f"Ppeak≈{Ppeak:.0f} | Pplat≈{Pplat:.0f} | ΔP≈{DP:.0f}",
                 fontsize=12, color=dp_col)

        # STRAIN(t) bar
        axL.text(0.05, 0.52, "STRAIN(t) = V(t)/FRC", fontsize=11, weight="bold")
        axL.add_patch(plt.Rectangle((0.05, 0.46), 0.90, 0.06, fill=False, linewidth=2))
        axL.add_patch(plt.Rectangle((0.05, 0.46), 0.90 * min(strain_t / 0.6, 1.0), 0.06,
                                    color=strain_col, alpha=0.90))
        axL.text(0.96, 0.46, f"{strain_t:.2f}", fontsize=11, ha="right", va="bottom", color=strain_col)

        # STRESS(t) bar
        axL.text(0.05, 0.36, "STRESS(t) ~ carga dinâmica (proxy)", fontsize=11, weight="bold")
        axL.add_patch(plt.Rectangle((0.05, 0.30), 0.90, 0.06, fill=False, linewidth=2))
        axL.add_patch(plt.Rectangle((0.05, 0.30), 0.90 * min(stress_t / 25.0, 1.0), 0.06,
                                    color=dp_col, alpha=0.90))
        axL.text(0.96, 0.30, f"{stress_t:.0f}", fontsize=11, ha="right", va="bottom", color=dp_col)

        axL.text(0.05, 0.08,
                 "Zonas: verde (seguro) | laranja (limite) | vermelho (risco)",
                 fontsize=10)
        axL.text(0.05, 0.04,
                 f"Targets modernos: Pplat<{PPLAT_SAFE} | ΔP<{DP_SAFE} | Strain pico<{STRAIN_SAFE}",
                 fontsize=10)

        axL.set_xlim(0, 1)
        axL.set_ylim(0, 1)
        axL.axis("off")
        axL.set_title("ARDS — titulação stepwise (UCI)", fontsize=12)

        # ---------------- Right panel: ventilator waveforms ----------------
        hist = min(i + 1, 2 * cyc_frames)
        x = np.linspace(0, hist / fps, hist)

        Paw_hist = np.array([Pawc[(k - j) % cyc_frames] for j in reversed(range(hist))])
        V_hist = np.array([Vc[(k - j) % cyc_frames] for j in reversed(range(hist))])
        F_hist = np.array([Fc[(k - j) % cyc_frames] for j in reversed(range(hist))])

        # Pressure: show PEEP + plateau and ΔP shading
        axP.plot(x, Paw_hist, lw=2.5, color="black")
        axP.axhline(PEEP, ls="--", color="gray", lw=1.5, label="PEEP")
        axP.axhline(Pplat, ls="--", color="blue", lw=1.2, label="Pplat")
        axP.fill_between([0, x.max() if len(x) else 1], PEEP, Pplat, color=dp_col, alpha=0.12, label="ΔP")
        axP.set_ylim(0, 45)
        axP.set_ylabel("Pressão (cmH₂O)")
        axP.grid(True, alpha=0.25)
        axP.set_title("VCV: Pico + Plateau (pausa inspiratória)")
        axP.legend(loc="upper right", fontsize=8, frameon=True)

        # Volume
        axV.plot(x, V_hist, lw=2, color="purple")
        axV.set_ylim(0, max(650, VT * 1.2))
        axV.set_ylabel("Volume (mL)")
        axV.grid(True, alpha=0.25)
        axV.set_title("Volume vs tempo")

        # Flow
        axF.plot(x, F_hist * 60, lw=2, color="teal")
        axF.axhline(0, color="gray", lw=1.2)
        axF.set_ylim(-80, 80)
        axF.set_ylabel("Fluxo (L/min)")
        axF.set_xlabel("Tempo (s)")
        axF.grid(True, alpha=0.25)
        axF.set_title("Fluxo vs tempo")

        plt.tight_layout()

        # Render frame (Agg modern-safe)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)

    # ---------------- Freeze 1s between steps ----------------
    freeze_frames = int(freeze_seconds * fps)
    fig.clf()
    ax = plt.subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.66, label, fontsize=26, weight="bold", ha="center")
    ax.text(0.5, 0.52, f"VT={VT} mL | PEEP={PEEP} cmH₂O", fontsize=18, ha="center")
    ax.text(0.5, 0.40,
            f"Pplat≈{Pplat:.0f} | ΔP≈{DP:.0f} | Strain pico≈{strain_peak:.2f}",
            fontsize=18, ha="center", color=dp_col)
    ax.text(0.5, 0.20,
            "Pausa didáctica (1s): interpretar mecânica → decidir próximo passo",
            fontsize=16, ha="center")

    plt.tight_layout()
    fig.canvas.draw()
    freeze = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    for _ in range(freeze_frames):
        writer.append_data(freeze)

writer.close()
print("OK ->", OUT)
