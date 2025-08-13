
# evap_coil_textbook_plus_strict.py
# Streamlit evaporator coil first-cut using textbook correlations
# Robust moist-air psychrometrics implemented in pure Python (no HAPropsSI required).
# Refrigerant properties via CoolProp PropsSI (still recommended).

import math
from math import pi, sqrt, tanh
import numpy as np
import pandas as pd
import io
import streamlit as st

# ------------------ Optional CoolProp for refrigerants ------------------
try:
    import CoolProp.CoolProp as CP
    COOLPROP = True
except Exception:
    COOLPROP = False

INCH = 0.0254
MM = 1e-3
P_ATM = 101325.0
R_DA = 287.055  # J/kg-K, dry air gas constant

def K(tC): return tC + 273.15
def C(tK): return tK - 273.15

# ------------------ Moist-air psychrometrics (no CoolProp needed) ------------------
def psat_water_Pa(T_C: float) -> float:
    """Saturation vapor pressure over liquid water (Pa). Buck equation (good 0..50°C)."""
    return 611.21 * math.exp((18.678 - T_C/234.5) * (T_C/(257.14 + T_C)))

def humidity_ratio_from_T_RH(T_C: float, RH_pct: float, P: float = P_ATM) -> float:
    RH = max(min(RH_pct, 100.0), 0.1) / 100.0
    Psat = psat_water_Pa(T_C)
    Pv = RH * Psat
    return 0.62198 * Pv / max(P - Pv, 1.0)

def cp_moist_air_J_per_kgK(T_C: float, W: float) -> float:
    """Specific heat of moist air per kg dry air (J/kg_da-K)."""
    cp_da = 1006.0
    cp_v = 1860.0
    return cp_da + W * cp_v

def h_moist_air_J_per_kg(T_C: float, W: float) -> float:
    """Moist air enthalpy per kg dry air (J/kg_da)."""
    return 1000.0 * (1.006 * T_C + W * (2501.0 + 1.86 * T_C))

def rho_moist_air_kg_per_m3(T_C: float, W: float, P: float = P_ATM) -> float:
    """Moist air density (kg/m3) using ideal-gas mixing (per kg moist air approximation)."""
    T_K = K(T_C)
    return P / (R_DA * T_K * (1.0 + 1.6078 * W))

def mu_air_Pa_s(T_C: float) -> float:
    """Dynamic viscosity via Sutherland's law for air (Pa·s)."""
    T = K(T_C)
    mu0 = 1.716e-5  # Pa·s at 273.15 K
    S = 110.4  # K
    return mu0 * ((T / 273.15) ** 1.5) * ((273.15 + S) / (T + S))

def k_air_W_per_mK(T_C: float) -> float:
    """Thermal conductivity of air (W/m-K); simple linear approx around room temp."""
    # 0°C: ~0.024, 40°C: ~0.027
    return 0.024 + (0.027 - 0.024) * (T_C / 40.0)

def air_props(Tdb_C: float, RH_pct: float, P: float = P_ATM):
    W = humidity_ratio_from_T_RH(Tdb_C, RH_pct, P)
    rho = rho_moist_air_kg_per_m3(Tdb_C, W, P)
    mu  = mu_air_Pa_s(Tdb_C)
    k   = k_air_W_per_mK(Tdb_C)
    cp  = cp_moist_air_J_per_kgK(Tdb_C, W)
    Pr  = cp * mu / max(k, 1e-9)
    return dict(rho=rho, mu=mu, k=k, cp=cp, Pr=Pr, W=W)

def state_from_T_RH(T_C: float, RH_pct: float, P: float = P_ATM):
    W = humidity_ratio_from_T_RH(T_C, RH_pct, P)
    h = h_moist_air_J_per_kg(T_C, W)
    return dict(h=h, W=W, T=T_C, RH=RH_pct, P=P)

def state_from_T_W(T_C: float, W: float, P: float = P_ATM):
    h = h_moist_air_J_per_kg(T_C, W)
    # Back-calc RH roughly (optional; not used in core calcs)
    Psat = psat_water_Pa(T_C)
    Pv = W * P / (0.62198 + W)
    RH = max(min(100.0 * Pv / max(Psat, 1.0), 100.0), 0.0)
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
    N_tpr = max(int(math.floor(H / St)), 1)
    N_tubes = N_tpr * Nr
    L_tube = W
    s = (1.0/FPI) * INCH
    A_holes_one_fin = N_tubes * (pi*(Do/2)**2)
    A_fin_one = max(2.0*(W*H - A_holes_one_fin), 0.0)
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
    Kdp = 3.0 * Nr * 0.02
    meta = dict(model="Zukauskas (tube-bank)", Re=Re, Nu=Nu, Vmax=Vmax, K=Kdp)
    return h, meta

def manglik_bergles_jf(Re, Pr, s_l, s_t, t_fin, L_off=0.003):
    phi1 = max(s_l/L_off, 1e-6)
    phi2 = max(s_t/L_off, 1e-6)
    phi3 = max(t_fin/L_off, 1e-6)
    Re = max(Re, 1.0)
    j = (0.6522*(Re**-0.5403)*(1 + (5.269e-5*(Re**1.340)) * (phi1**0.504)) *
         (1 + (Re/2712.0)**1.279)**0.928 *
         (1 + (phi3/phi2)**3.537)**0.1) * (Pr**(-1/3))
    f = (9.6243*(Re**-0.7422)*(1 + (1.0/ ( (phi1)**0.185 )) ) *
         (1 + (Re/2712.0)**1.5)**0.17 *
         (1 + (phi3/phi2)**0.5))
    return j, f

def air_htc_manglik_bergles(props, geom, Nr, St, tf, mdot_air):
    rho, mu, k, Pr = props['rho'], props['mu'], props['k'], props['Pr']
    Dh = 2.0*geom['s']
    Vmax = mdot_air/(rho*geom['A_min'])
    Re_h = rho*Vmax*Dh/max(mu,1e-12)
    j, f = manglik_bergles_jf(Re_h, Pr, s_l=geom['s'], s_t=max(St, geom['s']), t_fin=tf, L_off=0.003)
    Stj = j*(Pr**(2.0/3.0))
    h = Stj * rho * Vmax * props['cp']
    Kdp = 4.0 * Nr * max(f, 1e-3)
    meta = dict(model="Manglik–Bergles (offset-strip)", Re=Re_h, j=j, f=f, Vmax=Vmax, Dh=Dh, K=Kdp)
    return h, meta


def air_dp_from_meta(props, meta, geom, Nr, St, tf):
    """
    More realistic air-side ΔP estimate using channel friction + entry/exit + row form losses.

    For Manglik–Bergles (offset-strip): uses Fanning f from correlation:
        ΔP ≈ [4 f (L/Dh) + K_entry + K_exit] * 0.5 * ρ * Vmax^2
    For plain plate-fin with tube-bank (Zukauskas baseline):
        Use hydraulic diameter Dh ≈ 2*s (parallel plates), friction factor:
            f_D (Darcy) = 64/Re (laminar Re<2300) else 0.3164/Re^0.25
        Convert to equivalent form:
            ΔP ≈ [f_D (L/Dh) + K_entry + K_exit + K_row*Nr] * 0.5 * ρ * Vmax^2
        where K_row ~ 0.9 per row (form drag due to tubes), K_entry ~0.5, K_exit ~1.0
    """
    rho = props['rho']
    Vmax = meta.get('Vmax', 0.0)
    L = Nr * St  # coil depth
    Dh = 2.0*geom['s']  # hydraulic diameter (parallel-plate proxy)

    Re = max(rho*Vmax*Dh/max(props['mu'],1e-12), 1.0)

    K_entry = 0.5
    K_exit  = 1.0

    if 'f' in meta and meta.get('model','').lower().startswith('manglik'):
        # Fanning f from MB correlation
        f_fanning = max(meta.get('f', 0.01), 0.001)
        K_core = 4.0 * f_fanning * (L/max(Dh,1e-9))
        K_total = K_core + K_entry + K_exit
    else:
        # Plain plate-fin + tube bank baseline
        # Darcy friction factor
        if Re < 2300.0:
            f_D = 64.0/Re
        else:
            f_D = 0.3164/(Re**0.25)
        K_row = 0.9  # per-row form loss (tunable)
        K_core = f_D * (L/max(Dh,1e-9)) + K_row * max(Nr,1)
        K_total = K_core + K_entry + K_exit

    dP = K_total * 0.5 * rho * Vmax**2
    return dP


# ------------------ ε–NTU ------------------
def UA_required_eNTU(Q_kW, mdot_air, cp_air, T_air_in_C, T_sat_evap_C):
    Q = Q_kW*1000.0
    C_air = mdot_air*cp_air
    DTstar = (T_air_in_C - T_sat_evap_C)
    if DTstar <= 0 or C_air <= 0: return None, None, None
    eps = min(max(Q/(C_air*DTstar), 1e-6), 0.999999)
    NTU = -math.log(1.0 - eps)
    UA = NTU * C_air
    return UA, eps, NTU

# ------------------ Refrigerant side (homogeneous two-phase ΔP) ------------------
def mixture_density(x, rho_l, rho_g):
    return 1.0 / (x/max(rho_g,1e-12) + (1.0-x)/max(rho_l,1e-12))

def mixture_viscosity_McAdams(x, mu_l, mu_g):
    return (mu_l**(1.0-x))*(mu_g**x)

def dp_fric_homogeneous(G, Di, L, x_mean, rho_l, rho_g, mu_l, mu_g):
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

# ------------------ ADP/BPF ------------------
def estimate_bpf(Nr, FPI, v_face):
    base = 0.20
    depth_factor = min(0.08*Nr, 0.6)
    fpi_factor   = min(0.006*(FPI-10), 0.15) if FPI>=10 else -0.04
    vel_factor   = max(0.04*(v_face-2.0), -0.08)
    bpf = base - depth_factor - fpi_factor + vel_factor
    return float(np.clip(bpf, 0.03, 0.25))

def adp_bpf_leaving(air_in, ADP_C, BPF):
    # Saturated enthalpy and humidity at ADP using our formulas
    Psat = psat_water_Pa(ADP_C)
    W_ADP = 0.62198 * Psat / max(P_ATM - Psat, 1.0)
    h_ADP = h_moist_air_J_per_kg(ADP_C, W_ADP)
    h_out = h_ADP + BPF*(air_in['h'] - h_ADP)
    W_out = W_ADP + BPF*(air_in['W'] - W_ADP)
    # Approximate T_out by solving enthalpy equation iteratively (simple secant)
    def f_T(Tc):
        return h_moist_air_J_per_kg(Tc, W_out) - h_out
    T1, T2 = ADP_C, air_in['T']
    for _ in range(20):
        f1, f2 = f_T(T1), f_T(T2)
        if abs(f2 - f1) < 1e-6: break
        T3 = T2 - f2*(T2-T1)/max(f2-f1, 1e-6)
        T1, T2 = T2, T3
    T_out = T2
    return dict(T_out=T_out, h_out=h_out, W_out=W_out)

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Evaporator Coil Designer (No-HAPropsSI)", layout="wide")
st.title("Evaporator Coil Designer — Textbook + Wet coil + ADP/BPF + ΔP (no HAPropsSI)")

with st.sidebar:
    st.header("Load & Air")
    Q_kW = st.number_input("Cooling load (kW)", 1.0, 5000.0, 50.0, 1.0)
    Tdb_in = st.number_input("Entering air dry-bulb (°C)", -20.0, 60.0, 27.0, 0.1)
    RH_in  = st.number_input("Entering air RH (%)", 0.1, 100.0, 50.0, 0.1)
    T_sat_ev = st.number_input("Evap saturation temp (°C)", -40.0, 20.0, 6.0, 0.1)

    st.markdown("---")
    st.subheader("Airflow")
    mode = st.radio("Specify airflow", ["Face velocity (m/s)", "Volumetric flow (m³/h)"], horizontal=True)
    W = st.number_input("Face width W (m)", 0.2, 4.0, 1.2, 0.05)
    H = st.number_input("Face height H (m)", 0.2, 4.0, 1.0, 0.05)

    st.markdown("---")
    st.subheader("Refrigerant (CoolProp for fluids)")
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
    wet_coil = st.checkbox("Wet coil (enhanced h via Lewis analogy)", value=True)
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
Lc = min(0.5*geom['s'], 0.003)
eta_f = fin_efficiency_infinite_plate(h_air, k_fin, tf, Lc)
Ao = geom['A_total']
eta_o = 1.0 - (geom['A_fin']/max(Ao,1e-9))*(1.0 - eta_f)
Uo = eta_o * h_air
UA_air = Uo * Ao

# ε–NTU requirement
UA_req, eps_needed, NTU_needed = UA_required_eNTU(Q_kW, mdot_air, air_in_props['cp'], Tdb_in, T_sat_ev)

# Air ΔP
dP_air = air_dp_from_meta(air_in_props, meta, geom, Nr, St, tf)

# Refrigerant-side checks & ΔP
mdot_ref = mdot_ref_hr/3600.0
mdot_per = mdot_ref/max(N_circuits,1)
Ai_in = pi*(Di**2)/4.0
G = mdot_per/max(Ai_in,1e-12)

rho_l = rho_g = mu_l = mu_g = None
if COOLPROP and ref in CP.get_global_param_string("fluids_list").split(','):
    T_sat = K(T_sat_ev)
    rho_l = CP.PropsSI("D","T",T_sat,"Q",0.0,ref)
    rho_g = CP.PropsSI("D","T",T_sat,"Q",1.0,ref)
    mu_l  = CP.PropsSI("V","T",T_sat,"Q",0.0,ref)
    mu_g  = CP.PropsSI("V","T",T_sat,"Q",1.0,ref)

tubes_per_circ = max(int(round(geom['N_tubes']/max(N_circuits,1))), 1)
L_circ = tubes_per_circ * geom['L_tube']
Lsh_frac = st.slider("Superheat zone length fraction of circuit", 0.00, 0.40, 0.10, 0.01)
L_tp = L_circ * (1.0 - Lsh_frac)
L_sh = L_circ * Lsh_frac
x_mean = 0.5

dp_tp = dp_vap = None
meta_tp = meta_v = {}
if all(v is not None for v in [rho_l, rho_g, mu_l, mu_g]) and Ai_in > 0:
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
    ADP_C = T_sat_ev + 2.0
else:
    ADP_C = st.number_input("ADP (°C)", -10.0, 25.0, T_sat_ev + 2.0, 0.1)
    BPF = st.slider("Bypass factor", 0.02, 0.30, 0.10, 0.005)

air_in = state_from_T_RH(Tdb_in, RH_in)
leave = adp_bpf_leaving(air_in, ADP_C, BPF)

mdot_dryair = mdot_air  # close enough for first cut
Q_total_calc = mdot_dryair * (air_in['h'] - leave['h_out'])/1000.0  # kW
cp_da = air_in_props['cp']
Q_sens_est = mdot_air*cp_da*(air_in['T'] - leave['T_out'])/1000.0

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
            f"{Do/MM:.3f}", f"{(Do-Di)/2/MM:.2f}", f"{St/MM:.1f}",
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
            f"{air_dp_from_meta(air_in_props, meta, geom, Nr, St, tf):,.0f}"
        ]
    })
    st.dataframe(df_air, use_container_width=True)
    # --- ΔP Air metrics (explicit on-screen) ---
    colA1, colA2 = st.columns(2)
    colA1.metric("ΔP_air (Pa)", f"{(dP_air if ('dP_air' in globals()) and (dP_air is not None) else 0):,.0f}")
    colA2.metric("Vmax (m/s)", f"{meta.get('Vmax',0):.2f}")


    st.subheader("ADP/BPF Result (psychrometrics)")
    df_psy = pd.DataFrame({
        "Item":["ADP (°C)","BPF","Leaving T (°C)","h_in (kJ/kg_da)","h_out (kJ/kg_da)",
                "W_in (kg/kg)","W_out (kg/kg)","Q_total from psychro (kW)","Q_sensible est (kW)"],
        "Value":[
            f"{ADP_C:.2f}", f"{BPF:.03f}", f"{leave['T_out']:.2f}",
            f"{air_in['h']/1000:.2f}", f"{leave['h_out']/1000:.2f}",
            f"{air_in['W']:.5f}", f"{leave['W_out']:.5f}",
            f"{Q_total_calc:,.1f}", f"{Q_sens_est:,.1f}"
        ]
    })
    st.dataframe(df_psy, use_container_width=True)

with right:
    st.subheader("Refrigerant Side (ΔP & sanity)")
    df_ref = pd.DataFrame({
        "Metric":[
            "Refrigerant","CoolProp available?","ṁ_ref total (kg/s)","Circuits","ṁ per circuit (kg/s)",
            "Di (mm)","G per circuit (kg/m²·s)","Tubes/circuit (est.)","L_circuit (m)",
            "Two-phase length L_tp (m)","Vapor length L_sh (m)",
            "ρ_l (kg/m³)","ρ_g (kg/m³)","μ_l (Pa·s)","μ_g (Pa·s)",
            "ΔP_tp (kPa)","ΔP_vapor (kPa)","ΔP_total (kPa)"
        ],
        "Value":[
            f"{ref}", f"{'Yes' if COOLPROP else 'No'}", f"{mdot_ref:.3f}", f"{N_circuits}", f"{mdot_per:.4f}",
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
    # --- ΔP Refrigerant metrics (explicit on-screen) ---
    colR1, colR2, colR3 = st.columns(3)
    colR1.metric("ΔP_two-phase (kPa)", f"{((dp_tp or 0)/1000 if ('dp_tp' in globals()) else 0):.2f}")
    colR2.metric("ΔP_vapor (kPa)", f"{((dp_vap or 0)/1000 if ('dp_vap' in globals()) else 0):.2f}")
    colR3.metric("ΔP_total (kPa)", f"{((dp_ref_total or 0)/1000 if ('dp_ref_total' in globals()) else 0):.2f}")


    # ---------- Export buttons ----------
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_air.to_excel(writer, index=False, sheet_name="Air_UA")
            df_psy.to_excel(writer, index=False, sheet_name="ADP_BPF")
            df_ref.to_excel(writer, index=False, sheet_name="Refrigerant")
            # New sheet: combine key pressure drops for quick review
            try:
                pd_data = {
                    "Item": ["ΔP_air (Pa)", "ΔP_two-phase (kPa)", "ΔP_vapor (kPa)", "ΔP_total_refrigerant (kPa)", "Mass flux G (kg/m²·s)"],
                    "Value": [f"{dP_air:,.0f}" if "dP_air" in globals() and dP_air is not None else "—",
                              f"{(dp_tp or 0)/1000:.2f}" if "dp_tp" in globals() and dp_tp is not None else "—",
                              f"{(dp_vap or 0)/1000:.2f}" if "dp_vap" in globals() and dp_vap is not None else "—",
                              f"{(dp_ref_total or 0)/1000:.2f}" if "dp_ref_total" in globals() and dp_ref_total is not None else "—",
                              f"{G:,.0f}" if "G" in globals() and G is not None else "—"]
                }
                df_dp = pd.DataFrame(pd_data)
                df_dp.to_excel(writer, index=False, sheet_name="Pressure_Drops")
            except Exception as _e:
                pass
        
        xlsx_bytes = buffer.getvalue()
    st.download_button("Download XLSX", data=xlsx_bytes, file_name="evap_coil_firstcut.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    csv_pack = f"# Air/UA\n{df_air.to_csv(index=False)}\n# ADP_BPF\n{df_psy.to_csv(index=False)}\n# Refrigerant\n{df_ref.to_csv(index=False)}"
    st.download_button("Download CSV (all tables)", data=csv_pack, file_name="evap_coil_firstcut.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    "**Models included**  \n"
    "• **Zukauskas** (staggered tube-bank) and **Manglik–Bergles** (offset-strip) for air-side.  \n"
    "• **Moist-air psychrometrics** implemented internally: saturation (Buck), humidity ratio, enthalpy, density, μ(T) (Sutherland), k(T) linear, cp(T,W).  \n"
    "• **Wet-coil** heuristic via Lewis analogy factor.  \n"
    "• **Fin efficiency** (infinite plate) → overall η_o.  \n"
    "• **ε–NTU** with boiling on refrigerant side (C*→∞).  \n"
    "• **Air ΔP** via correlation meta.  \n"
    "• **Refrigerant ΔP**: homogeneous two-phase friction + single-phase vapor tail (CoolProp fluids if available)."
)
st.caption("Engineering caution: Generalized correlations; validate against your test data.")


# ======================= DESIGN ADVISOR (minimal add-on) =======================
# The following block adds actionable suggestions using variables already computed.
# Appends at end to avoid interfering with any multi-line strings above.

import numpy as np
import pandas as pd

st.markdown("---")
st.subheader("Design Advisor (what to change to meet target)")

advice_rows = []     # for a table of computed targets
advice_bullets = []  # short write-up bullets

def _fmt(x, nd=2):
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

try:
    # --- 1) UA gap (if any) ---
    if ('UA_req' in globals()) and ('UA_air' in globals()) and (UA_req is not None) and (UA_air is not None) and (UA_air < UA_req):
        gap = UA_req - UA_air
        ratio = UA_req / max(UA_air, 1e-9)

        # Rows/FPI/Area scaling: first-cut linear scaling of outside area Ao ~ UA (air-side dominated)
        rows_needed = int(np.ceil(Nr * ratio)) if 'Nr' in globals() else None
        fpi_needed  = int(np.clip(np.ceil(FPI * ratio), 6, 22)) if 'FPI' in globals() else None
        area_needed = face_area * ratio if 'face_area' in globals() else None

        # Increase face velocity option (boost h): h ~ V^alpha (alpha ~= 0.6 plain, 0.8 offset-strip)
        model_name = str(air_model).lower() if 'air_model' in globals() else ""
        alpha = 0.6 if "zukauskas" in model_name else 0.8
        v_needed = v_face * (ratio ** (1.0 / max(alpha, 0.25))) if 'v_face' in globals() else None
        dp_scale = (v_needed / max(v_face, 1e-6)) ** 2 if ('v_needed' in locals() and v_needed is not None) else None  # ΔP ~ V^2 approximation

        advice_rows += [
            ("UA short by (W/K)", f"{gap:,.0f}"),
            ("UA ratio needed (×)", _fmt(ratio, 2)),
        ]
        if rows_needed is not None and 'Nr' in globals():
            advice_rows += [("Rows →", f"{Nr} ➜ {rows_needed}")]
        if fpi_needed is not None and 'FPI' in globals():
            advice_rows += [("FPI →", f"{int(FPI)} ➜ {fpi_needed}")]
        if area_needed is not None and 'face_area' in globals():
            advice_rows += [("Face area (m²) →", f"{face_area:.3f} ➜ {area_needed:.3f}")]
        if v_needed is not None and 'v_face' in globals():
            dp_txt = f" (ΔP ×{dp_scale:.2f})" if dp_scale is not None else ""
            advice_rows += [(f"Face velocity (m/s) → (α≈{alpha:.1f})", f"{v_face:.2f} ➜ {v_needed:.2f}{dp_txt}")]

        # Bullets
        bullet = "UA is short. "
        if rows_needed is not None: bullet += f"Increase **rows** to ~{rows_needed}, "
        if fpi_needed is not None:  bullet += f"or **FPI** to ~{fpi_needed}, "
        if area_needed is not None: bullet += f"or **face area** to ~{area_needed:.3f} m², "
        if v_needed is not None:    bullet += f"or raise **face velocity** to ~{v_needed:.2f} m/s"
        if dp_scale is not None:    bullet += f" (expect air ΔP ×{dp_scale:.2f})."
        advice_bullets += [bullet.strip().rstrip(',') + ".",
                           "Switching to **offset-strip/louvered fins** (if currently plain) will raise h but also ΔP."]
    else:
        advice_bullets.append("UA meets requirement on this first cut. Fine-tune ΔP and psychrometrics if needed.")

    # --- 2) Air-side ΔP guardrail ---
    if 'dP_air' in globals():
        air_dp_limit = 150.0  # Pa, typical for comfort systems (adjust for fan budget)
        if dP_air is not None and dP_air > air_dp_limit:
            advice_rows += [("Air ΔP (Pa) (limit ~150)", f"{dP_air:,.0f} (>limit)")]
            advice_bullets += [
                f"**Air ΔP** is high (~{int(dP_air)} Pa). Reduce **FPI** or **rows**, or enlarge **face area** to lower velocity; "
                "plain fins reduce ΔP vs. offset-strip."
            ]

    # --- 3) Refrigerant mass flux & ΔP (typical targets) ---
    if 'G' in globals() and (G is not None):
        G_low, G_high = 150.0, 400.0  # kg/m²·s typical evaporator band
        advice_rows += [("Refrigerant mass flux G (kg/m²·s)", f"{G:,.0f} (target ~150–400)") ]
        if ('N_circuits' in globals()) and (N_circuits is not None) and (N_circuits >= 1):
            if G > G_high:
                G_target = 320.0
                N_circ_target = int(np.ceil(N_circuits * (G / G_target)))
                advice_rows += [("Circuits →", f"{N_circuits} ➜ {N_circ_target}")]
                advice_bullets += [
                    f"**G is high** (~{int(G)}). Increase **circuits** to ~{N_circ_target} (or use larger **Di**) to reduce ΔP and stabilize boiling."
                ]
            elif G < G_low and N_circuits > 1:
                G_target = 220.0
                N_circ_target = max(1, int(np.floor(N_circuits * (G / G_target))))
                advice_rows += [("Circuits →", f"{N_circuits} ➜ {N_circ_target}")]
                advice_bullets += [
                    f"**G is low** (~{int(G)}). Consider **fewer circuits** (~{N_circ_target}) to raise boiling HTC (watch distributor authority)."
                ]

    if 'dp_ref_total' in globals() and (dp_ref_total is not None):
        dp_ref_limit = 70_000.0  # Pa (~70 kPa)
        if dp_ref_total > dp_ref_limit:
            advice_rows += [("Refrigerant ΔP total (kPa)", f"{dp_ref_total/1000:,.1f} (>70)")]
            advice_bullets += [
                "Refrigerant **ΔP is high**. Increase **circuits** or **Di**, shorten circuits (fewer tubes/pass), "
                "or reduce **superheat length** if possible."
            ]

    # --- 4) Psychrometric consistency (sanity) ---
    if ('Q_total_calc' in globals()) and (Q_total_calc is not None) and ('Q_kW' in globals()):
        mismatch = abs(Q_total_calc - Q_kW) / max(Q_kW, 1e-6)
        advice_rows += [("Q_psychro vs duty (kW)", f"{Q_total_calc:,.1f} vs {Q_kW:,.1f}")]
        if mismatch > 0.15:
            advice_bullets += [
                f"Psychrometric load differs from design by ~{mismatch*100:0.1f}%. Revisit **ADP/BPF**, airflow, and air properties."
            ]

    # --- Render advisor outputs ---
    if advice_rows:
        df_advice = pd.DataFrame(advice_rows, columns=["Parameter / Action", "Suggested value"])
        st.dataframe(df_advice, use_container_width=True)

    if advice_bullets:
        st.markdown("**Recommendations:**")
        for b in advice_bullets:
            st.markdown(f"- {b}")

    # Optional: downloadable CSV
    if advice_rows:
        st.download_button(
            "Download improvement suggestions (CSV)",
            data=pd.DataFrame(advice_rows, columns=["Parameter/Action","Suggested value"]).to_csv(index=False),
            file_name="design_improvements.csv",
            mime="text/csv"
        )

except Exception as e:
    st.warning(f"Design Advisor skipped due to: {e}")
# =================== END DESIGN ADVISOR (minimal add-on) ===================

