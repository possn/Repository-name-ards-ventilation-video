import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# ---------- OUTPUT ----------
out_mp4 = "ARDS_Stepwise_Long_Final_ICU.mp4"

# ---------- VIDEO ----------
fps = 12
freeze_seconds = 1.0

# ---------- MODEL (ARDS) ----------
FRC = 1000  # mL (baby lung)
R = 10      # cmH2O/(L/s)  resistência global
C = 25      # mL/cmH2O     complacência ARDS (baixa)

# ---------- BREATH SHAPE (VCV + pause) ----------
T = 3.5          # s por ciclo
insp_frac = 0.35 # inspiração
pause_frac = 0.12# pausa inspiratória (plateau)
exp_frac = 1 - insp_frac - pause_frac

def one_cycle_waveforms(VT, PEEP, n_frames):
    t = np.linspace(0, T, n_frames, endpoint=False)
    ph = (t % T) / T

    V = np.zeros_like(ph)          # mL
    Flow = np.zeros_like(ph)       # L/s

    insp = ph < insp_frac
    pause = (ph >= insp_frac) & (ph < insp_frac + pause_frac)
    exp = ph >= (insp_frac + pause_frac)

    # insp: fluxo constante -> volume linear
    insp_time = insp_frac * T
    const_flow = (VT/1000) / insp_time  # L/s
    V[insp] = VT * (ph[insp] / insp_frac)
    Flow[insp] = const_flow

    # pause: hold (fluxo 0, volume constante)
    V[pause] = VT
    Flow[pause] = 0.0

    # exp: decaimento exponencial (modelo simples)
    exp_time = exp_frac * T
    exp_phase = (ph[exp] - (insp_frac + pause_frac)) / exp_frac
    tau = 0.35
    Vexp = VT * np.exp(-exp_phase/tau)
    V[exp] = Vexp

    # derivada aproximada para fluxo expiratório (negativo)
    dVdt = np.gradient(V, t, edge_order=1)  # mL/s
    Flow = dVdt/1000.0                      # L/s

    # pressões
    Pel = V / C
    Pres = Flow * R
    Paw = PEEP + Pel + Pres

    # valores úteis
    Pplat = PEEP + (VT/C)
    Ppeak = PEEP + (VT/C) + (const_flow*R)
    DP = Pplat - PEEP

    return t, V, Flow, Paw, Ppeak, Pplat, DP

def risk_color(val, g, y):
    return "green" if val <= g else ("orange" if val <= y else "red")

# ---------- STEPWISE (ARDS alto -> optimizado) ----------
steps = [
    {"VT": 500, "PEEP": 6,  "label":"ARDS NÃO PROTEGIDO", "seconds":10},
    {"VT": 420, "PEEP": 6,  "label":"STEP 1 — VT ↓",      "seconds":10},
    {"VT": 420, "PEEP": 10, "label":"STEP 2 — PEEP ↑",    "seconds":10},
    {"VT": 360, "PEEP": 10, "label":"STEP 3 — PROTETOR",  "seconds":10},
]

# targets modernos
DP_safe = 14
Pplat_safe = 28
strain_safe = 0.25

writer = imageio.get_writer(
    out_mp4,
    fps=fps,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset","ultrafast","-crf","28"]
)

W, H = 1280, 720
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)

for s in steps:
    VT = s["VT"]; PEEP = s["PEEP"]
    label = s["label"]
    total_frames = int(s["seconds"] * fps)

    # para ter plateau e pico visíveis: usar ciclos repetidos
    frames_per_cycle = int(T * fps)
    # waveforms de um ciclo
    tc, Vc, Fc, Pawc, Ppeak, Pplat, DP = one_cycle_waveforms(VT, PEEP, frames_per_cycle)

    # strain dinâmico: V(t)/FRC  (varia com inspiração/expiração)
    # strain pico: VT/FRC
    strain_peak = VT / FRC
    strain_col = risk_color(strain_peak, 0.25, 0.35)
    stress_col = risk_color(DP, 14, 18)

    # gerar frames desta fase
    for i in range(total_frames):
        fig.clf()

        ax_left = plt.subplot2grid((3,2),(0,0),rowspan=3)
        ax_p    = plt.subplot2grid((3,2),(0,1))
        ax_v    = plt.subplot2grid((3,2),(1,1))
        ax_f    = plt.subplot2grid((3,2),(2,1))

        # índice dentro do ciclo
        k = i % frames_per_cycle

        # -------- Pulmão (esquemático) --------
        # tamanho proporcional ao volume instantâneo (didático)
        size = 0.20 + 0.16*(Vc[k]/max(VT,1))
        lung = plt.Circle((0.5,0.65), size, fill=False, linewidth=7, color=strain_col)
        ax_left.add_patch(lung)

        # -------- Barras dinâmicas --------
        strain_t = (Vc[k] / FRC)            # dinâmico
        strain_bar = min(strain_t/0.6, 1.0)

        # stress proxy dinâmico: usar (Paw-PEEP) como “carga dinâmica” instantânea (didático)
        # e mapear cor pelo risco ΔP (porque é isso que se ensina)
        stress_t = max(Pawc[k]-PEEP, 0.0)
        stress_bar = min(stress_t/25.0, 1.0)

        ax_left.text(0.05, 0.93, label, fontsize=16, weight="bold")
        ax_left.text(0.05, 0.86, f"VT={VT} mL | PEEP={PEEP} cmH₂O", fontsize=12)
        ax_left.text(0.05, 0.80, f"Ppeak≈{Ppeak:.0f} | Pplat≈{Pplat:.0f} | ΔP≈{DP:.0f}", fontsize=12, color=stress_col)

        # STRAIN bar (dinâmico)
        ax_left.text(0.05, 0.52, "STRAIN(t) = V(t)/FRC", fontsize=11, weight="bold")
        ax_left.add_patch(plt.Rectangle((0.05,0.46), 0.90,0.06, fill=False, linewidth=2))
        ax_left.add_patch(plt.Rectangle((0.05,0.46), 0.90*strain_bar,0.06, color=strain_col, alpha=0.9))
        ax_left.text(0.96, 0.46, f"{strain_t:.2f}", fontsize=11, ha="right", color=strain_col, va="bottom")

        # STRESS bar (dinâmico)
        ax_left.text(0.05, 0.36, "STRESS(t) ~ carga dinâmica (proxy)", fontsize=11, weight="bold")
        ax_left.add_patch(plt.Rectangle((0.05,0.30), 0.90,0.06, fill=False, linewidth=2))
        ax_left.add_patch(plt.Rectangle((0.05,0.30), 0.90*stress_bar,0.06, color=stress_col, alpha=0.9))
        ax_left.text(0.96, 0.30, f"{stress_t:.0f}", fontsize=11, ha="right", color=stress_col, va="bottom")

        ax_left.text(0.05, 0.05,
                     f"Targets modernos: Pplat<{Pplat_safe} | ΔP<{DP_safe} | Strain pico<{strain_safe}",
                     fontsize=10)

        ax_left.set_xlim(0,1); ax_left.set_ylim(0,1); ax_left.axis("off")
        ax_left.set_title("ARDS — Stepwise titration (didático UCI)", fontsize=12)

        # -------- Curvas ventilador (mostrar pico + plateau) --------
        # construir histórico curto (últimos ~2 ciclos) para leitura clara
        hist = min(i+1, 2*frames_per_cycle)
        idx = np.arange(hist)
        Paw_hist = np.array([Pawc[(k-j)%frames_per_cycle] for j in reversed(range(hist))])
        V_hist   = np.array([Vc[(k-j)%frames_per_cycle]   for j in reversed(range(hist))])
        F_hist   = np.array([Fc[(k-j)%frames_per_cycle]   for j in reversed(range(hist))])

        x = np.linspace(0, hist/fps, hist)

        # PRESSÃO
        ax_p.plot(x, Paw_hist, linewidth=2.5, color="black")
        ax_p.axhline(PEEP,  linestyle="--", color="gray", linewidth=1.5, label="PEEP")
        ax_p.axhline(Pplat, linestyle="--", color="blue", linewidth=1.2, label="Pplat")
        ax_p.fill_between([0, x.max() if len(x)>0 else 1], PEEP, Pplat, color=stress_col, alpha=0.10, label="ΔP")
        ax_p.set_ylim(0, 45)
        ax_p.set_ylabel("Pressão (cmH₂O)")
        ax_p.grid(True, alpha=0.25)
        ax_p.set_title("VCV: Pico + Plateau (pausa inspiratória)")
        ax_p.legend(loc="upper right", fontsize=8, frameon=True)

        # VOLUME
        ax_v.plot(x, V_hist, linewidth=2.0, color="purple")
        ax_v.set_ylabel("Volume (mL)")
        ax_v.set_ylim(0, max(600, VT*1.2))
        ax_v.grid(True, alpha=0.25)
        ax_v.set_title("Volume vs tempo")

        # FLUXO
        ax_f.plot(x, F_hist*60, linewidth=2.0, color="teal")
        ax_f.axhline(0, color="gray", linewidth=1.2)
        ax_f.set_ylabel("Fluxo (L/min)")
        ax_f.set_xlabel("Tempo (s)")
        ax_f.set_ylim(-80, 80)
        ax_f.grid(True, alpha=0.25)
        ax_f.set_title("Fluxo vs tempo")

        plt.tight_layout()

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape((H, W, 3))
        writer.append_data(frame)

    # ---------- Freeze 1s between steps ----------
    freeze_frames = int(freeze_seconds * fps)
    fig.clf()
    ax = plt.subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.65, label, fontsize=26, weight="bold", ha="center")
    ax.text(0.5, 0.50, f"VT={VT} mL | PEEP={PEEP} cmH₂O", fontsize=18, ha="center")
    ax.text(0.5, 0.38, f"Pplat≈{Pplat:.0f} | ΔP≈{DP:.0f} | Strain pico≈{strain_peak:.2f}", fontsize=18, ha="center", color=stress_col)
    ax.text(0.5, 0.20,
            "Pausa didáctica (1s) — interpretar mecânica e decidir próximo passo",
            fontsize=16, ha="center")
    plt.tight_layout()
    fig.canvas.draw()
    freeze = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((H, W, 3))
    for _ in range(freeze_frames):
        writer.append_data(freeze)

writer.close()
print("Gerado:", out_mp4)
