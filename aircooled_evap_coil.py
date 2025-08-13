# evap_coil_textbook_plus.py
# Streamlit evaporator coil first-cut using textbook correlations
# Adds: wet-coil factor, ADP/BPF psychrometrics, air & refrigerant ΔP, CSV/Excel export.

import math
from math import pi, sqrt, tanh, log
import numpy as np
import pandas as pd
import io
import streamlit as st

# ------------------ Props (CoolProp) ------------------
try:
    import CoolProp.CoolProp as CP
    from CoolProp.HumidAirProp import HAPropsSI
    COOLPROP = True
except Exception:
    COOLPROP = False

INCH = 0.0254
MM = 1e-3

def K(tC): return tC + 273.15
def C(tK): return tK - 273.15

def air_props(Tdb_C, RH_pct, P=101325.0):
    if not COOLPROP:
        # dry-air fallback (rough)
        T = K(Tdb_C)
        rho = 1.2
        mu  = 1.85e-5
        k   = 0.026
        cp  = 1006.0
        return dict(rho=rho, mu=mu, k=k, cp=cp, Pr=cp*mu/k)
    rho = HAPropsSI("Rho","T",K(Tdb_C),"P",P,"R",RH_pct/100.0)
    mu  = HAPropsSI("Mu","T",K(Tdb_C),"P",P,"R",RH_pct/100.0)
    k   = HAPropsSI("K","T",K(Tdb_C),"P",P,"R",RH_pct/100.0)
    cp  = HAPropsSI("C","T",K(Tdb_C),"P",P,"R",RH_pct/100.0)
    Pr  = cp*mu/k
    return dict(rho=rho, mu=mu, k=k, cp=cp, Pr=Pr)

# Psychro helpers (need CoolProp)
def state_from_T_RH(T_C, RH_pct, P=101325.0):
    if not COOLPROP:
        return dict(h=1006.0*(T_C-0), W=0.0, T=T_C, RH=RH_pct, P=P)
    h = HAPropsSI("H","T",K(T_C),"P",P,"R",RH_pct/100.0)   # J/kg dry air
    W = HAPropsSI("W","T",K(T_C),"P",P,"R",RH_pct/100.0)   # kg/kg
    return dict(h=h, W=W, T=T_C, RH=RH_pct, P=P)

def state_from_T_W(T_C, W, P=101325.0):
    if not COOLPROP:
        return dict(h=1006.0*(T_C-0), W=W, T=T_C, RH=None, P=P)
    h = HAPropsSI("H","T",K(T_C),"P",P,"W",W)
    RH = HAPropsSI("R","T",K(T_C),"P",P,"W",W)*100
    return dict(h=h, W=W, T=T_C, RH=RH, P=P)

# ------------------ Fin efficiency ------------------
def fin_efficiency_infinite_plate(h, k_fin, t_fin, Lc):
    if Lc <= 0 or t_fin <= 0 or k_fin <= 0: return 1.0
    m = sqrt(2.0*h/(k_fin*t_fin))
    x = max(m*Lc, 1e-9)
    return tanh(x)/x

# ------------------ Geometry & areas ------------------
def geometry_areas(W, H, Nr, St, Sl, Do, tf, FPI):
    face_area = W*H
    depth = Nr*Sl
    fins_count = int(round(FPI*(depth/INCH)))
    # tubes per row (vertical): assume horizontal tubes across width
    N_tpr = max(int(math.floor(H / St)), 1)
    N_tubes = N_tpr * Nr
    L_tube = W

    # fin pitch and free area
    s = (1.0/FPI) * INCH  # fin spacing (m)
    A_holes_one_fin = N_tubes * (pi*(Do/2)**2)
    A_fin_one = max(2.0*(W*H - A_holes_one_fin), 0.0)  # both sides of one fin
    A_fin_total = A_fin_one * fins_count

    exposed_frac = max((s-tf)/max(s,1e-9), 0.0)
    A_bare = N_tubes * (pi * Do * L_tube) * exposed_frac
    A_total = A_fin_total + A_bare

    fin_blockage = min(tf/max(s,1e-9), 0.95)
    tube_blockage = min(A_holes_one_fin/max(W*H,1e-9), 0.5)
    A_min = max(face_area * (1.0 - fin_blockage - tube_blockage), 1e-4)

    return dict(face_area=face_area, depth=depth, fins=fins_count, s=s,
                N_tpr=N_tpr, N_tubes=N_tubes, L_tube=L_tube,
                A_fin=A_fin_total, A_bare=A_bare, A_total=A_total, A_min=A_min)

# ------------------ Air side correlations ------------------
# A) Zukauskas for staggered tube banks
def zukauskas_constants(Re):
    Re = max(Re, 1.0)
    if 1e2 <= Re < 1e3:   C, m = 0.9, 0.4
    elif 1e3 <= Re < 2e5: C, m = 0.27, 0.63
    else:                 C, m = (0.27, 0.63) if Re >= 2e5 else (0.9, 0.4)
    return C, m

def row_correction(Nr):
    if Nr <= 1: return 0.70
    if Nr == 2: return 0.80
    if Nr == 3: return 0.88
    if Nr == 4: return 0.94
    return 1.00

def air_htc_zukauskas(props, geom, Do, Nr, mdot_air):
    rho, mu, k, Pr = props['rho'], props['mu'], props['k'], props['Pr']
    Vmax = mdot_air/(rho*geom['A_min'])
    Re = rho*Vmax*Do/max(mu,1e-12)
    C, m = zukauskas_constants(Re)
    Nu = C*(Re**m)*(Pr**0.36)
    Nu *= row_correction(Nr)
    h = Nu * k / Do
    # crude air-side DP factor tied to Re & rows
    K = 3.0 * Nr * 0.02
    meta = dict(model="Zukauskas (tube-bank)", Re=Re, Nu=Nu, Vmax=Vmax, K=K)
    return h, meta

# B) Manglik–Bergles for offset-strip fin (compact HX family)
def manglik_bergles_jf(Re, Pr, s_l, s_t, t_fin, L_off=0.003):
    phi1 = max(s_l/L_off, 1e-6)
    phi2 = max(s_t/L_off, 1e-6)
    phi3 = max(t_fin/L_off, 1e-6)
    Re = max(Re, 1.0)
    # j (dimensionless) and f (Fanning) compact forms capturing their original trends
    j = (0.6522*(Re**-0.5403)*(1 + (5.269e-5*(Re**1.340)) * (phi1**0.504)) *
         (1 + (Re/2712.0)**1.279)**0.928 *
         (1 + (phi3/phi2)**3.537)**0.1) * (Pr**(-1/3))
    f = (9.6243*(Re**-0.7422)*(1 + (1.0/ ( (phi1)**0.185 )) ) *
         (1 + (Re/2712.0)**1.5)**0.17 *
         (1 + (phi3/phi2)**0.5))
    return j, f

def air_htc_manglik_bergles(props, geom, Nr, St, tf, mdot_air):
    rho, mu, k, Pr = props['rho'], props['mu'], props['k'], props['Pr']
    Dh = 2.0*geom['s']           # parallel-plate proxy
    Vmax = mdot_air/(rho*geom['A_min'])
    Re_h = rho*Vmax*Dh/max(mu,1e-12)
    j, f = manglik_bergles_jf(Re_h, Pr, s_l=geom['s'], s_t=max(St, geom['s']), t_fin=tf, L_off=0.003)
    St = j*(Pr**(2.0/3.0))
    h = St * rho * Vmax * props['cp']
    # ΔP via Fanning friction f (rows as length multiplier)
    K = 4.0 * Nr * max(f, 1e-3)
    meta = dict(model="Manglik–Bergles (offset-strip)", Re=Re_h, j=j, f=f, Vmax=Vmax, Dh=Dh, K=K)
    return h, meta

def air_dp_from_meta(props, meta):
    rho = props['rho']
    V = meta['Vmax']
    K = meta.get('K', 0.1)
    return K * 0.5 * rho * V**2

# ------------------ ε–NTU (boiling side ~ infinite C) ------------------
def UA_required_eNTU(Q_kW, mdot_air, cp_air, T_air_in_C, T_sat_evap_C):
    Q = Q_kW*1000.0
    C_air = mdot_air*cp_air
    DTstar = (T_air_in_C - T_sat_evap_C)
    if DTstar <= 0 or C_air <= 0: return None, None, None
    eps = min(max(Q/(C_air*DTstar), 1e-6), 0.999999)
    NTU = -math.log(1.0 - eps)
    UA = NTU * C_air
    return UA, eps, NTU

# ------------------ Refrigerant side (homogeneous two-phase) ------------------
def mixture_density(x, rho_l, rho_g):
    return 1.0 / (x/max(rho_g,1e-12) + (1.0-x)/max(rho_l,1e-12))

def mixture_viscosity_McAdams(x, mu_l, mu_g):
    # μ_mix ≈ μ_l^(1−x) μ_g^x
    return (mu_l**(1.0-x))*(mu_g**x)

def dp_fric_homogeneous(G, Di, L, x_mean, rho_l, rho_g, mu_l, mu_g):
    # Blasius f for turbulent: f = 0.3164 / Re^0.25
    mu_m = mixture_viscosity_McAdams(x_mean, mu_l, mu_g)
    rho_m = mixture_density(x_mean, rho_l, rho_g)
    Re_m = G*Di/max(mu_m,1e-12)
    f = 0.3164 / max(Re_m,1.0)**0.25
    dp = f * (L/max(Di,1e-12)) * (G**2/(2.0*max(rho_m,1e-9)))
    return dp, dict(Re_m=Re_m, f=f, rho_m=rho_m, mu_m=mu_m)

def dp_single_phase(G, Di, L, rho, mu):
    Re = G*Di/max(mu,1e-12)
    f = 0.3164 / max(Re,1.0)**0.25
    dp = f * (L/max(Di,1e-12)) * (G**2/(2.0*max(rho,1e-9)))
    return dp, dict(Re=Re, f=f)

# ------------------ ADP/BPF block ------------------
def estimate_bpf(Nr, FPI, v_face):
    # rough trend: deeper & tighter fins => lower BPF
    base = 0.20
    depth_factor = min(0.08*Nr, 0.6)
    fpi_factor   = min(0.006*(FPI-10), 0.15) if FPI>=10 else -0.04
    vel_factor   = max(0.04*(v_face-2.0), -0.08)
    bpf = base - depth_factor - fpi_factor + vel_factor
    return float(np.clip(bpf, 0.03, 0.25))

def adp_bpf_leaving(air_in, ADP_C, BPF):
    # Along a straight line on the psych chart (ADP line): h_out = h_ADP + BPF*(h_in - h_ADP)
    if not COOLPROP:
        return dict(T_out=ADP_C + BPF*(air_in['T']-ADP_C),
                    h_out=air_in['h']*BPF, W_out=air_in['W']*BPF) # very rough fallback
    h_ADP = HAPropsSI("H","T",K(ADP_C),"P",air_in['P'],"R",1.0)  # saturated at ADP
    W_ADP = HAPropsSI("W","T",K(ADP_C),"P",air_in['P'],"R",1.0)
    h_out = h_ADP + BPF*(air_in['h'] - h_ADP)
    # Find leaving state at same line: use h & ADP line intersection -> iterate on T to match h
    # Practical shortcut: assume the leaving point lies along a straight mix line towards ADP in (h,W)
    W_out = W_ADP + BPF*(air_in['W'] - W_ADP)
    # Compute T_out from (h_out,W_out)
    T_out = C(HAPropsSI("T","H",h_out,"P",air_in['P'],"W",W_out))
    return dict(T_out=T_out, h_out=h_out, W_out=W_out)

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Evaporator Coil Designer (Textbook+Wet/ADP/ΔP)", layout="wide")
st.title("Evaporator Coil Designer — Textbook correlations + Wet coil + ADP/BPF + ΔP")

with st.sidebar:
    st.header("Load & Air")
    Q_kW = st.number_input("Cooling load (kW)", 1.0, 5000.0, 50.0, 1.0)
    Tdb_in = st.number_input("Entering air dry-bulb (°C)", -20.0, 60.0, 27.0, 0.1)
    RH_in  = st.number_input("Entering air RH (%)", 0.0, 100.0, 50.0, 1.0)
    T_sat_ev = st.number_input("Evap saturation temp (°C)", -40.0, 20.0, 6.0, 0.1)

    st.markdown("---")
    st.subheader("Airflow")
    mode = st.radio("Specify airflow", ["Face velocity (m/s)", "Volumetric flow (m³/h)"], horizontal=True)
    W = st.number_input("Face width W (m)", 0.2, 4.0, 1.2, 0.05)
    H = st.number_input("Face height H (m)", 0.2, 4.0, 1.0, 0.05)

    st.markdown("---")
    st.subheader("Refrigerant")
    if COOLPROP:
        fluids = CP.get_global_param_string("fluids_list").split(',')
        pref = [f for f in ["R410A","R454B","R32","R407C","R134a","R290","R22","R513A","R1234yf","CO2"] if f in fluids]
        fluids = pref + [f for f in fluids if f not in pref]
    else:
        fluids = ["(CoolProp not available)"]
    ref = st.selectbox("Refrigerant", fluids, index=0)
    mdot_ref_hr = st.number_input("ṁ_ref total (kg/h)", 0.0, 100000.0, 800.0, 10.0)
    N_circuits = st.number_input("Circuits (parallel)", 1, 200, 8, 1)
    SH_out = st.number_input("Superheat at outlet (K)", 0.0, 20.0, 6.0, 0.5)

st.markdown("### Geometry & Fins")
c1, c2, c3, c4 = st.columns(4)
with c1:
    tube_pick = st.selectbox("Tube OD", ["3/8 in (9.525 mm)","1/2 in (12.7 mm)","Custom"])
    if tube_pick == "3/8 in (9.525 mm)":
        Do = 3/8*INCH
    elif tube_pick == "1/2 in (12.7 mm)":
        Do = 0.5*INCH
    else:
        Do = st.number_input("Tube OD (mm)", 4.0, 20.0, 9.525, 0.1)*MM
    ti = st.number_input("Tube wall thickness (mm)", 0.3, 0.8, 0.5, 0.05)*MM
    Di = max(Do - 2*ti, 1e-4)

with c2:
    pitch_pick = st.selectbox("Triangular pitch", ["1.00 in","1.25 in","Custom"])
    if pitch_pick == "1.00 in":
        pitch = 1.00*INCH
    elif pitch_pick == "1.25 in":
        pitch = 1.25*INCH
    else:
        pitch = st.number_input("Triangular pitch (mm)", 12.0, 60.0, 25.4, 0.5)*MM
    St = pitch
    Sl = pitch
    Nr = st.number_input("Rows (depth)", 1, 12, 4, 1)

with c3:
    FPI = st.number_input("Fins per inch (FPI)", 6.0, 22.0, 12.0, 1.0)
    tf = st.number_input("Fin thickness (mm)", 0.10, 0.15, 0.12, 0.01)*MM
    fin_mat = st.selectbox("Fin material", ["Aluminum","Copper"])
    k_fin = 200.0 if fin_mat == "Aluminum" else 380.0

with c4:
    air_model = st.selectbox("Air-side model", ["Zukauskas (tube-bank; plain fin baseline)",
                                                 "Manglik–Bergles (offset-strip ≈ louvered)"])
    user_ho = st.number_input("Override h_air (W/m²K) [optional]", 0.0, 1500.0, 0.0, 1.0)
    wet_coil = st.checkbox("Wet coil (enhanced h by Lewis analogy)", value=True)
    wet_factor = st.slider("Wet enhancement factor (×)", 1.10, 1.80, 1.40, 0.01)

# Air / Flow
air_in_props = air_props(Tdb_in, RH_in)
face_area = W*H
if mode == "Face velocity (m/s)":
    v_face = st.number_input("Face velocity (m/s)", 0.5, 4.0, 2.2, 0.1)
    vol_flow = v_face * face_area
else:
    m3ph = st.number_input("Airflow (m³/h)", 500.0, 300000.0, 10000.0, 100.0)
    vol_flow = m3ph/3600.0
    v_face = vol_flow/face_area

mdot_air = air_in_props['rho']*vol_flow
geom = geometry_areas(W, H, Nr, St, Sl, Do, tf, FPI)

# Air-side HTC
if user_ho > 0:
    h_air_dry = user_ho
    meta = dict(model="User override", Vmax=mdot_air/(air_in_props['rho']*geom['A_min']), K=0.1)
else:
    if air_model.startswith("Zukauskas"):
        h_air_dry, meta = air_htc_zukauskas(air_in_props, geom, Do, Nr, mdot_air)
    else:
        h_air_dry, meta = air_htc_manglik_bergles(air_in_props, geom, Nr, St, tf, mdot_air)

# Wet enhancement (Lewis analogy style)
h_air = h_air_dry * (wet_factor if wet_coil else 1.0)

# Fin/overall efficiency & UA_air
Lc = min(0.5*geom['s'], 0.003)  # characteristic half-gap capped
eta_f = fin_efficiency_infinite_plate(h_air, k_fin, tf, Lc)
Ao = geom['A_total']
eta_o = 1.0 - (geom['A_fin']/max(Ao,1e-9))*(1.0 - eta_f)
Uo = eta_o * h_air
UA_air = Uo * Ao

# ε–NTU requirement
UA_req, eps_needed, NTU_needed = UA_required_eNTU(Q_kW, mdot_air, air_in_props['cp'], Tdb_in, T_sat_ev)

# Air ΔP
dP_air = air_dp_from_meta(air_in_props, meta)

# Refrigerant-side checks & ΔP
mdot_ref = mdot_ref_hr/3600.0
mdot_per = mdot_ref/max(N_circuits,1)
Ai_in = pi*(Di**2)/4.0
G = mdot_per/max(Ai_in,1e-12)

h_tp = None
rho_l = rho_g = mu_l = mu_g = None
if COOLPROP and ref in CP.get_global_param_string("fluids_list").split(','):
    T_sat = K(T_sat_ev)
    rho_l = CP.PropsSI("D","T",T_sat,"Q",0.0,ref)
    rho_g = CP.PropsSI("D","T",T_sat,"Q",1.0,ref)
    mu_l  = CP.PropsSI("V","T",T_sat,"Q",0.0,ref)
    mu_g  = CP.PropsSI("V","T",T_sat,"Q",1.0,ref)
    k_l   = CP.PropsSI("L","T",T_sat,"Q",0.0,ref)
    cp_l  = CP.PropsSI("C","T",T_sat,"Q",0.0,ref)
    Pr_l  = cp_l*mu_l/k_l
    # Liquid-only estimate (Dittus-Boelter) for reference HTC
    Re_lo = G*Di/max(mu_l,1e-12)
    Nu_lo = 0.023*(Re_lo**0.8)*(Pr_l**0.4)
    h_lo  = Nu_lo*k_l/max(Di,1e-12)
    # Two-phase HTC simple boost (placeholder): factor 1.2–2.0 typical; keep h_air dominant
    h_tp = max(1.2*h_lo, h_lo)

# Per-circuit length (no bends counted — first cut)
tubes_per_circ = max(int(round(geom['N_tubes']/max(N_circuits,1))), 1)
L_circ = tubes_per_circ * geom['L_tube']

# Two-phase frictional ΔP (homogeneous), assume evaporation zone length fraction (1 - Lsh_frac)
Lsh_frac = st.slider("Superheat zone length fraction of circuit", 0.00, 0.40, 0.10, 0.01)
L_tp = L_circ * (1.0 - Lsh_frac)
L_sh = L_circ * Lsh_frac
x_mean = 0.5  # average quality in boiling region (first cut)

dp_tp = dp_vap = None
meta_tp = meta_v = {}
if (rho_l and rho_g and mu_l and mu_g) is not None:
    if L_tp > 0:
        dp_tp, meta_tp = dp_fric_homogeneous(G, Di, L_tp, x_mean, rho_l, rho_g, mu_l, mu_g)
    if L_sh > 0:
        dp_vap, meta_v = dp_single_phase(G, Di, L_sh, rho_g, mu_g)

dp_ref_total = (dp_tp or 0.0) + (dp_vap or 0.0)

# ------------------ ADP/BPF block ------------------
st.markdown("---")
st.subheader("ADP / Bypass Factor (psychrometrics)")
adp_mode = st.radio("ADP/BPF inputs", ["Estimate BPF from geometry & velocity", "Manually specify ADP & BPF"], horizontal=True)
if adp_mode == "Estimate BPF from geometry & velocity":
    BPF = estimate_bpf(Nr, FPI, v_face)
    ADP_C = T_sat_ev + 2.0  # common first cut
else:
    ADP_C = st.number_input("ADP (°C)", -10.0, 25.0, T_sat_ev + 2.0, 0.1)
    BPF = st.slider("Bypass factor", 0.02, 0.30, 0.10, 0.005)

air_in = state_from_T_RH(Tdb_in, RH_in) if COOLPROP else dict(h=1006*(Tdb_in-0), W=0.0, T=Tdb_in, P=101325)
leave = adp_bpf_leaving(air_in, ADP_C, BPF)

if COOLPROP:
    mdot_dryair = mdot_air   # HAProps returns per kg *dry* air; our mdot from bulk density is close enough here
    Q_total_calc = mdot_dryair * (air_in['h'] - leave['h_out'])/1000.0  # kW
    # sensible estimate:
    cp_da = air_in_props['cp']
    Q_sens_est = mdot_air*cp_da*(air_in['T'] - leave['T_out'])/1000.0
else:
    Q_total_calc = Q_kW
    Q_sens_est = Q_kW * 0.7

# ------------------ Tables & Export ------------------
left, right = st.columns([1.2,1.0])

with left:
    st.subheader("Capacity & Air Side")
    meets = (UA_req is not None) and (UA_air >= UA_req)
    st.markdown(f"**UA check:** {'✅ Meets (first-cut)' if meets else '⚠️ UA short'}")
    df_air = pd.DataFrame({
        "Metric":[
            "Duty (kW)","Face area (m²)","Face velocity (m/s)","Airflow (m³/h)",
            "Rows","FPI","Fin thickness (mm)","Fin material",
            "Tube OD (mm)","Tube wall (mm)","Triangular pitch (mm)",
            "Tubes / row (est.)","Total tubes (est.)",
            "Coil depth (m)","Fins (count)","Min free area A_min (m²)",
            "Air model","Vmax (m/s)","Re/Dh (shown)","h_air dry (W/m²K)","Wet factor (×)","h_air used (W/m²K)",
            "η_f","η_o","Ao (m²)","UA_air (W/K)","UA_req (W/K)","ε needed","NTU needed","ΔP_air (Pa)"
        ],
        "Value":[
            f"{Q_kW:,.1f}", f"{face_area:,.3f}", f"{v_face:,.2f}", f"{vol_flow*3600:,.0f}",
            f"{Nr}", f"{FPI:.0f}", f"{tf/MM:.2f}", f"{fin_mat}",
            f"{Do/MM:.3f}", f"{ti/MM:.2f}", f"{St/MM:.1f}",
            f"{geom['N_tpr']}", f"{geom['N_tubes']}",
            f"{geom['depth']:.3f}", f"{geom['fins']}", f"{geom['A_min']:.3f}",
            meta.get("model",""), f"{meta.get('Vmax',0):.2f}",
            f"{meta.get('Re',0):,.0f}" if "Re" in meta else f"Dh={meta.get('Dh',0)/MM:.2f} mm",
            f"{h_air_dry:.1f}", f"{wet_factor if wet_coil else 1.00:.2f}", f"{h_air:.1f}",
            f"{fin_efficiency_infinite_plate(h_air,k_fin,tf,Lc):.3f}",
            f"{eta_o:.3f}", f"{Ao:.2f}",
            f"{UA_air:,.0f}" if UA_air else "—",
            f"{UA_req:,.0f}" if UA_req else "—",
            f"{eps_needed:.3f}" if eps_needed else "—",
            f"{NTU_needed:.3f}" if NTU_needed else "—",
            f"{air_dp_from_meta(air_in_props, meta):,.0f}"
        ]
    })
    st.dataframe(df_air, use_container_width=True)

    st.subheader("ADP/BPF Result (psychrometrics)")
    df_psy = pd.DataFrame({
        "Item":["ADP (°C)","BPF","Leaving T (°C)","h_in (kJ/kg_da)","h_out (kJ/kg_da)",
                "W_in (kg/kg)","W_out (kg/kg)","Q_total from psychro (kW)","Q_sensible est (kW)"],
        "Value":[
            f"{ADP_C:.2f}", f"{BPF:.03f}", f"{leave['T_out']:.2f}",
            f"{air_in['h']/1000:.2f}" if COOLPROP else "—",
            f"{leave['h_out']/1000:.2f}" if COOLPROP else "—",
            f"{air_in['W']:.5f}" if COOLPROP else "—",
            f"{leave['W_out']:.5f}" if COOLPROP else "—",
            f"{Q_total_calc:,.1f}" if COOLPROP else "—",
            f"{Q_sens_est:,.1f}"
        ]
    })
    st.dataframe(df_psy, use_container_width=True)

with right:
    st.subheader("Refrigerant Side (ΔP & sanity)")
    df_ref = pd.DataFrame({
        "Metric":[
            "Refrigerant","ṁ_ref total (kg/s)","Circuits","ṁ per circuit (kg/s)",
            "Di (mm)","G per circuit (kg/m²·s)","Tubes/circuit (est.)","L_circuit (m)",
            "Two-phase length L_tp (m)","Vapor length L_sh (m)",
            "ρ_l (kg/m³)","ρ_g (kg/m³)","μ_l (Pa·s)","μ_g (Pa·s)",
            "ΔP_tp (kPa)","ΔP_vapor (kPa)","ΔP_total (kPa)"
        ],
        "Value":[
            f"{ref}", f"{mdot_ref:.3f}", f"{N_circuits}", f"{mdot_per:.4f}",
            f"{Di/MM:.2f}", f"{G:,.0f}", f"{tubes_per_circ}", f"{L_circ:.2f}",
            f"{L_tp:.2f}", f"{L_sh:.2f}",
            f"{rho_l:.1f}" if rho_l else "—",
            f"{rho_g:.2f}" if rho_g else "—",
            f"{mu_l:.2e}" if mu_l else "—",
            f"{mu_g:.2e}" if mu_g else "—",
            f"{(dp_tp or 0)/1000:.2f}", f"{(dp_vap or 0)/1000:.2f}",
            f"{(dp_ref_total or 0)/1000:.2f}"
        ]
    })
    st.dataframe(df_ref, use_container_width=True)

    # ---------- Export buttons ----------
    st.subheader("Export")
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_air.to_excel(writer, index=False, sheet_name="Air/UA")
            df_psy.to_excel(writer, index=False, sheet_name="ADP_BPF")
            df_ref.to_excel(writer, index=False, sheet_name="Refrigerant")
        xlsx_bytes = buffer.getvalue()
    st.download_button("Download XLSX", data=xlsx_bytes, file_name="evap_coil_firstcut.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    csv_pack = f"# Air/UA\n{df_air.to_csv(index=False)}\n# ADP_BPF\n{df_psy.to_csv(index=False)}\n# Refrigerant\n{df_ref.to_csv(index=False)}"
    st.download_button("Download CSV (all tables)", data=csv_pack, file_name="evap_coil_firstcut.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    "**Models included**  \n"
    "• **Zukauskas** (staggered tube-bank) for external convection over plain plate-fin coils.  \n"
    "• **Manglik–Bergles (1995)** (offset-strip fin) to approximate louvered/strongly interrupted fins.  \n"
    "• **Wet-coil** heuristic via Lewis analogy (tunable enhancement factor).  \n"
    "• **Fin efficiency** (infinite plate) → overall surface efficiency η_o.  \n"
    "• **ε–NTU** with boiling on refrigerant side (C*→∞).  \n"
    "• **Air ΔP** derived from the selected air-side model.  \n"
    "• **Refrigerant ΔP** via **homogeneous two-phase** friction + single-phase vapor in the superheat tail."
)
st.caption("Engineering caution: These generalized correlations are suitable for first cuts. Calibrate factors (esp. wet enhancement and BPF) with your own test data.")
