
# aircooled_evap_coil_corrected.py
# Full Streamlit app: evaporator coil first-cut with DB+RH / DB+WB toggles and airflow m/s <-> m³/h toggle.
# Includes: Zukauskas / Manglik–Bergles air-side, fin efficiency, UA check, ADP/BPF diagnostic & override,
# refrigerant-side ΔP (homogeneous two-phase + vapor) and optional Shah-style h_ref estimate.

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
    """Moist air density (kg/m3)."""
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
    return 0.024 + (0.027 - 0.024) * (T_C / 40.0)

def air_props(Tdb_C: float, RH_pct: float, P: float = P_ATM):
    W = humidity_ratio_from_T_RH(Tdb_C, RH_pct, P)
    rho = rho_moist_air_kg_per_m3(Tdb_C, W, P)
    mu  = mu_air_Pa_s(Tdb_C)
    k   = k_air_W_per_mK(Tdb_C)
    cp  = cp_moist_air_J_per_kgK(Tdb_C, W)
    Pr  = cp * mu / max(k, 1e-9)
    return dict(rho=rho, mu=mu, k=k, cp=cp, Pr=Pr, W=W, T=Tdb_C, RH=RH_pct)

# --- DB+WB helpers ---
def RH_from_T_W(T_C, W, P=P_ATM):
    Pv = W*P/(0.62198 + W)
    Ps = psat_water_Pa(T_C)
    RH = max(min(100.0*Pv/max(Ps,1e-9), 100.0), 0.1)
    return RH

def humidity_ratio_from_T_WB(Tdb_C: float, Twb_C: float, P: float = P_ATM) -> float:
    Ps_wb = psat_water_Pa(Twb_C)
    Ws_wb = 0.62198 * Ps_wb / max(P - Ps_wb, 1.0)
    W = Ws_wb + (1.006*(Tdb_C - Twb_C)) / max(2501.0 - 2.381*Twb_C, 1e-3)
    return float(max(W, 1e-6))

def state_from_T_RH(T_C: float, RH_pct: float, P: float = P_ATM):
    W = humidity_ratio_from_T_RH(T_C, RH_pct, P)
    h = h_moist_air_J_per_kg(T_C, W)
    rho = rho_moist_air_kg_per_m3(T_C, W, P)
    mu  = mu_air_Pa_s(T_C)
    k   = k_air_W_per_mK(T_C)
    cp  = cp_moist_air_J_per_kgK(T_C, W)
    Pr  = cp * mu / max(k, 1e-9)
    return dict(h=h, W=W, T=T_C, RH=RH_pct, P=P, rho=rho, mu=mu, k=k, cp=cp, Pr=Pr)

# ------------------ Fin efficiency ------------------
def fin_efficiency_infinite_plate(h, k_fin, t_fin, Lc):
    if Lc <= 0 or t_fin <= 0 or k_fin <= 0: return 1.0
    m = sqrt(2.0*h/(k_fin*t_fin))
    x = max(m*Lc, 1e-9)
    return tanh(x)/x

# ------------------ Geometry & areas ------------------
def geometry_areas(W, H, Nr, St, Sl, Do, tf, FPI):
    face_area = W*H
    depth = Nr*Sl  # includes half-pitch buffers
    fins_count = int(round(FPI * (H/INCH)))  # vertical fins across height
    N_tpr = max(int(math.floor(H / max(St,1e-9))), 1)
    N_tubes = N_tpr * Nr
    L_tube = W
    s = (1.0/FPI) * INCH
    A_holes_one_fin = N_tubes * (pi*(Do/2)**2)
    A_fin_one = max(2.0*(W*H - A_holes_one_fin), 0.0)
    A_fin_total = A_fin_one * fins_count
    exposed_frac = max((s - tf)/max(s,1e-9), 0.0)
    A_bare = N_tubes * (pi * Do * L_tube) * exposed_frac
    A_total = A_fin_total + A_bare
    fin_blockage = min(tf/max(s,1e-9), 0.95)
    tube_blockage = min(A_holes_one_fin/max(W*H,1e-9), 0.5)
    A_min = max(face_area * (1.0 - fin_blockage - tube_blockage), 1e-4)
    return dict(face_area=face_area, depth=depth, fins=fins_count, s=s,
                N_tpr=N_tpr, N_tubes=N_tubes, L_tube=L_tube,
                A_fin=A_fin_total, A_bare=A_bare, A_total=A_total, A_min=A_min)

# ------------------ Air-side correlations ------------------
def zukauskas_constants(Re):
    Re = max(Re, 1.0)
    if 1e2 <= Re < 1e3:   C, m = 0.9, 0.4
    elif 1e3 <= Re < 2e5: C, m = 0.27, 0.63
    else:                 C, m = (0.27, 0.63) if Re >= 2e5 else (0.9, 0.4)
    return C, m

def row_correction(Nr):
    return 0.70 if Nr<=1 else (0.80 if Nr==2 else (0.88 if Nr==3 else (0.94 if Nr==4 else 1.00)))

def air_htc_zukauskas(props, geom, Do, Nr, mdot_air):
    rho, mu, k, Pr = props['rho'], props['mu'], props['k'], props['Pr']
    Vmax = mdot_air/(rho*geom['A_min'])
    Re = rho*Vmax*Do/max(mu,1e-12)
    C, m = zukauskas_constants(Re)
    Nu = C*(Re**m)*(Pr**0.36) * row_correction(Nr)
    h = Nu * k / Do
    meta = dict(model="Zukauskas (tube-bank)", Re=Re, Nu=Nu, Vmax=Vmax)
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
    meta = dict(model="Manglik–Bergles (offset-strip)", Re=Re_h, j=j, f=f, Vmax=Vmax, Dh=Dh)
    return h, meta

def air_dp_model(props, meta, geom, Nr, St, tf):
    rho = props['rho']
    Vmax = meta.get('Vmax', 0.0)
    L = Nr * St
    Dh = 2.0*geom['s']
    Re = max(rho*Vmax*Dh/max(props['mu'],1e-12), 1.0)
    K_entry, K_exit = 0.5, 1.0
    if meta.get('model','').lower().startswith('manglik'):
        f_f = max(meta.get('f', 0.01), 0.001)
        K_core = 4.0 * f_f * (L/max(Dh,1e-9))
        K_total = K_core + K_entry + K_exit
    else:
        f_D = 64.0/Re if Re < 2300.0 else 0.3164/(Re**0.25)
        K_row = 0.9
        K_core = f_D * (L/max(Dh,1e-9)) + K_row * max(Nr,1)
        K_total = K_core + K_entry + K_exit
    return K_total * 0.5 * rho * Vmax**2

# ------------------ Refrigerant-side ΔP and HTC (Shah approx) ------------------
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

def h_ref_shah_firstcut(Q_total_kW, N_circuits, Di, L_tp, T_sat_ev, ref, G_circ, x=0.5):
    if not COOLPROP: return None
    try:
        T_evK = K(T_sat_ev)
        rho_l = CP.PropsSI("D","T",T_evK,"Q",0.0,ref)
        rho_g = CP.PropsSI("D","T",T_evK,"Q",1.0,ref)
        mu_l  = CP.PropsSI("V","T",T_evK,"Q",0.0,ref)
        mu_g  = CP.PropsSI("V","T",T_evK,"Q",1.0,ref)
        k_l   = CP.PropsSI("L","T",T_evK,"Q",0.0,ref)
        cp_l  = CP.PropsSI("C","T",T_evK,"Q",0.0,ref)
        h_l   = CP.PropsSI("H","T",T_evK,"Q",0.0,ref)
        h_g   = CP.PropsSI("H","T",T_evK,"Q",1.0,ref)
        h_fg  = max(h_g - h_l, 1e-6)
        Q_circ = (Q_total_kW*1000.0)/max(N_circuits,1)
        A_tp   = pi*Di*max(L_tp, 1e-6)
        q_flux = Q_circ/max(A_tp,1.0)
        Re_l = max(G_circ*(1.0-x)*Di/max(mu_l,1e-12), 1.0)
        Pr_l = max(cp_l*mu_l/max(k_l,1e-12), 1e-6)
        Nu_lo = 0.023*(Re_l**0.8)*(Pr_l**0.4)
        h_lo = Nu_lo* k_l / max(Di,1e-9)
        Bo = q_flux / max(G_circ*h_fg, 1e-6)
        F_bo = 1.8*math.sqrt(max(Bo,1e-9))
        X_tt = ((1.0-x)/max(x,1e-3))**0.9 * (rho_g/max(rho_l,1e-9))**0.5 * (mu_l/max(mu_g,1e-12))**0.1
        F_x = 2.0/(1.0 + X_tt**0.5)
        h_ref = h_lo * (1.0 + F_bo) * F_x
        return float(np.clip(h_ref, 300.0, 8000.0))
    except Exception:
        return None

# ------------------ ADP/BPF helper ------------------
def estimate_bpf(Nr, FPI, v_face):
    base = 0.20
    depth_factor = min(0.08*Nr, 0.6)
    fpi_factor   = min(0.006*(FPI-10), 0.15) if FPI>=10 else -0.04
    vel_factor   = max(0.04*(v_face-2.0), -0.08)
    return float(np.clip(base - depth_factor - fpi_factor + vel_factor, 0.03, 0.25))

def h_sat_at_T(Tc):
    Wsat = humidity_ratio_from_T_RH(Tc, 100.0, P_ATM)
    return h_moist_air_J_per_kg(Tc, Wsat)

def invert_ADP_from_h(h_target, Tmin=-20.0, Tmax=40.0, it=40):
    a, b = Tmin, Tmax
    fa, fb = h_sat_at_T(a) - h_target, h_sat_at_T(b) - h_target
    for _ in range(it):
        m = 0.5*(a+b)
        fm = h_sat_at_T(m) - h_target
        if fm == 0 or abs(b-a) < 1e-4: return m
        if (fa*fm) <= 0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5*(a+b)

# ======================================================
# UI
# ======================================================
st.set_page_config(page_title="Evaporator Designer (Corrected)", layout="wide")
st.title("Air-Cooled Evaporator Designer — Corrected")

with st.sidebar:
    st.header("Air & Load")
    Q_kW = st.number_input("Design capacity (kW)", 1.0, 5000.0, 120.0, 1.0)
    W_face = st.number_input("Coil width W (m)", 0.2, 4.0, 1.2, 0.05)
    H_face = st.number_input("Coil height H (m)", 0.2, 4.0, 1.0, 0.05)
    face_area = W_face * H_face

    af_mode = st.radio("Airflow input", ["Face velocity (m/s)", "Volumetric (m³/h)"], horizontal=True)
    if af_mode == "Face velocity (m/s)":
        v_face = st.number_input("Face velocity (m/s)", 0.2, 6.0, 3.0, 0.1)
        vol_flow = v_face * face_area
        airflow_m3h = vol_flow * 3600.0
    else:
        airflow_m3h = st.number_input("Volumetric flow (m³/h)", 500.0, 200000.0, 20000.0, 100.0)
        vol_flow = airflow_m3h / 3600.0
        v_face = vol_flow / max(face_area, 1e-9)
    st.metric("Airflow (m³/h)", f"{airflow_m3h:,.0f}")
    st.metric("Face velocity (m/s)", f"{v_face:.2f}")

    st.subheader("On-coil (entering) air")
    in_mode = st.radio("Enter by", ["DB + RH", "DB + WB"], horizontal=True)
    Tdb_in = st.number_input("Entering dry-bulb (°C)", -10.0, 60.0, 27.0, 0.1)
    if in_mode == "DB + RH":
        RH_in  = st.number_input("Entering RH (%)", 1.0, 100.0, 50.0, 0.5)
        Twb_in = None
    else:
        Twb_in = st.number_input("Entering wet-bulb (°C)", -10.0, 40.0, 19.0, 0.1)
        W_in_calc = humidity_ratio_from_T_WB(Tdb_in, Twb_in, P_ATM)
        RH_in = RH_from_T_W(Tdb_in, W_in_calc, P_ATM)
        st.metric("Computed entering RH (%)", f"{RH_in:.1f}")

    st.subheader("Target OFF-coil air")
    out_mode = st.radio("Enter by ", ["DB + RH", "DB + WB"], horizontal=True)
    Tdb_off_tgt = st.number_input("Target leaving dry-bulb (°C)", -10.0, 40.0, 14.0, 0.1)
    if out_mode == "DB + RH":
        RH_off_tgt  = st.number_input("Target leaving RH (%)", 5.0, 100.0, 90.0, 0.5)
        Twb_off = None
    else:
        Twb_off = st.number_input("Target leaving wet-bulb (°C)", -10.0, 40.0, 13.0, 0.1)
        W_off_calc = humidity_ratio_from_T_WB(Tdb_off_tgt, Twb_off, P_ATM)
        RH_off_tgt = RH_from_T_W(Tdb_off_tgt, W_off_calc, P_ATM)
        st.metric("Computed target RH (%)", f"{RH_off_tgt:.1f}")

    st.subheader("Fin & Correlation")
    fin_type = st.selectbox("Fin type", ["Plain plate (no louvers)", "Louvered / Offset-strip"])

    st.subheader("Fouling & Film Coefficients")
    fouling_air_pct = st.number_input("Air-side fouling derate (%)", 0.0, 50.0, 10.0, 1.0)
    fouling_ref_pct = st.number_input("Refrigerant-side fouling derate (%)", 0.0, 50.0, 10.0, 1.0)

    st.subheader("Refrigerant h-method")
    h_method = st.selectbox("h_ref method", ["User estimate", "Shah (approx)"])
    h_ref_guess = st.number_input("Estimated refrigerant-side h (W/m²K)", 100.0, 8000.0, 1200.0, 50.0)
    h_ref_override = st.number_input("Override h_ref (W/m²K) [optional]", 0.0, 20000.0, 0.0, 50.0)

    st.subheader("Refrigerant (CoolProp for fluids)")
    if COOLPROP:
        fluids = CP.get_global_param_string("fluids_list").split(',')
        pref = [f for f in ["R410A","R454B","R32","R407C","R134a","R290","R22","R513A","R1234yf","CO2"] if f in fluids]
        fluids = pref + [f for f in fluids if f not in pref]
    else:
        fluids = ["(CoolProp not available)"]
    ref = st.selectbox("Refrigerant", fluids, index=0)
    T_sat_ev = st.number_input("Evaporator saturation temp (°C)", -40.0, 20.0, 6.0, 0.1)
    SH_out   = st.number_input("Suction superheat (K)", 0.0, 20.0, 6.0, 0.5)
    mdot_ref_hr = st.number_input("ṁ_ref total (kg/h)", 0.0, 100000.0, 800.0, 10.0)
    N_circuits = st.number_input("Refrigerant circuits (parallel)", 1, 48, 8, 1)
    T_cond   = st.number_input("Condenser saturation temp (°C)", 25.0, 70.0, 45.0, 0.5)
    Subcool  = st.number_input("Liquid subcooling (K)", 0.0, 20.0, 5.0, 0.5)

st.markdown("### Geometry & Fins")
c1, c2, c3, c4 = st.columns(4)
with c1:
    tube_pick = st.selectbox("Tube OD", ["3/8 in (9.525 mm)","1/2 in (12.7 mm)","Custom"])
    Do = 3/8*INCH if tube_pick.startswith("3/8") else (0.5*INCH if tube_pick.startswith("1/2") else st.number_input("Tube OD (mm)", 4.0, 20.0, 9.525, 0.1)*MM)
    ti = st.number_input("Tube wall thickness (mm)", 0.3, 0.8, 0.5, 0.05)*MM
    Di = max(Do - 2*ti, 1e-4)

with c2:
    St = st.number_input("Transverse pitch St (mm) — across height", 10.0, 80.0, 25.4, 0.5)*MM
    Sl = st.number_input("Longitudinal pitch Sl (mm) — along airflow", 10.0, 80.0, 22.0, 0.5)*MM
    Nr = st.number_input("Rows (depth)", 1, 12, 4, 1)

with c3:
    FPI = st.number_input("Fins per inch (FPI)", 6.0, 22.0, 12.0, 1.0)
    tf = st.number_input("Fin thickness (mm)", 0.10, 0.15, 0.12, 0.01)*MM
    fin_mat = st.selectbox("Fin material", ["Aluminum","Copper"])
    k_fin = 200.0 if fin_mat == "Aluminum" else 380.0

with c4:
    air_model = "Zukauskas (tube-bank)" if fin_type.startswith("Plain") else "Manglik–Bergles (offset-strip)"
    user_ho = st.number_input("Override h_air (W/m²K) [optional]", 0.0, 1500.0, 0.0, 1.0)
    wet_coil = st.checkbox("Wet coil (enhancement ×)", value=True)
    wet_factor = st.slider("Wet enhancement factor", 1.10, 1.80, 1.40, 0.01)

# States and flows
if in_mode == "DB + RH":
    air_in_props = state_from_T_RH(Tdb_in, RH_in)
else:
    W_in_calc = humidity_ratio_from_T_WB(Tdb_in, Twb_in, P_ATM)
    RH_in_calc = RH_from_T_W(Tdb_in, W_in_calc, P_ATM)
    air_in_props = state_from_T_RH(Tdb_in, RH_in_calc)

mdot_air = air_in_props['rho'] * (airflow_m3h/3600.0)

if out_mode == "DB + RH":
    air_out_tgt = state_from_T_RH(Tdb_off_tgt, RH_off_tgt)
else:
    W_off_calc = humidity_ratio_from_T_WB(Tdb_off_tgt, Twb_off, P_ATM)
    RH_off_calc = RH_from_T_W(Tdb_off_tgt, W_off_calc, P_ATM)
    air_out_tgt = state_from_T_RH(Tdb_off_tgt, RH_off_calc)

# Geometry
geom = geometry_areas(W_face, H_face, Nr, St, Sl, Do, tf, FPI)
Lc = max(0.5*(min(St, Sl) - Do), 1e-6)

# Air-side HTC & ΔP
if user_ho > 0:
    h_air_dry = user_ho
    meta = dict(model="User override", Vmax=mdot_air/(air_in_props['rho']*geom['A_min']), Re=None)
else:
    if air_model.startswith("Zukauskas"):
        h_air_dry, meta = air_htc_zukauskas(air_in_props, geom, Do, Nr, mdot_air)
    else:
        h_air_dry, meta = air_htc_manglik_bergles(air_in_props, geom, Nr, St, tf, mdot_air)

h_air = h_air_dry * (wet_factor if wet_coil else 1.0)
eta_f = fin_efficiency_infinite_plate(h_air, k_fin, tf, Lc)
Ao = geom['A_total']
eta_o = 1.0 - (geom['A_fin']/max(Ao,1e-9))*(1.0 - eta_f)
Uo = eta_o * h_air
UA_air = Uo * Ao

# Apply fouling derates for display
h_air_eff = h_air * (1.0 - fouling_air_pct/100.0)
Uo_air_only = eta_o * h_air_eff

dP_air = air_dp_model(air_in_props, meta, geom, Nr, St, tf)

# Refrigerant side
mdot_ref = mdot_ref_hr/3600.0
Ai_in = pi*(Di**2)/4.0
G = mdot_ref/max(Ai_in*max(geom['N_tubes'],1),1e-12)

rho_l = rho_g = mu_l = mu_g = None
dp_tp = dp_vap = None
dp_ref_total = None
Re_ref_mix = Re_ref_vap = None

if COOLPROP and ref in CP.get_global_param_string("fluids_list").split(','):
    T_evK = K(T_sat_ev)
    rho_l = CP.PropsSI("D","T",T_evK,"Q",0.0,ref)
    rho_g = CP.PropsSI("D","T",T_evK,"Q",1.0,ref)
    mu_l  = CP.PropsSI("V","T",T_evK,"Q",0.0,ref)
    mu_g  = CP.PropsSI("V","T",T_evK,"Q",1.0,ref)

    tubes_per_circ = max(int(round(geom['N_tubes']/max(N_circuits,1))), 1)
    L_circ = tubes_per_circ * geom['L_tube']
    Lsh_frac = 0.10
    L_tp = L_circ * (1.0 - Lsh_frac)
    L_sh = L_circ * Lsh_frac
    G_circ = mdot_ref / max(N_circuits,1) / max(Ai_in,1e-9)

    if L_tp > 0:
        _dp_tp, meta_tp = dp_fric_homogeneous(G_circ, Di, L_tp, 0.5, rho_l, rho_g, mu_l, mu_g)
        dp_tp = _dp_tp
        Re_ref_mix = meta_tp['Re_m']
    if L_sh > 0:
        _dp_vap, meta_sp = dp_single_phase(G_circ, Di, L_sh, rho_g, mu_g)
        dp_vap = _dp_vap
        Re_ref_vap = meta_sp['Re']

    dp_ref_total = (dp_tp or 0) + (dp_vap or 0)

# h_ref method
if h_method == "Shah (approx)":
    h_ref_from_shah = h_ref_shah_firstcut(Q_kW, N_circuits, Di, L_tp if 'L_tp' in locals() else 1.0, T_sat_ev, ref, G_circ if 'G_circ' in locals() else 200.0)
else:
    h_ref_from_shah = None

h_ref_raw = (h_ref_from_shah if h_ref_from_shah is not None else (h_ref_override if h_ref_override > 0 else h_ref_guess))
h_ref_eff = h_ref_raw * (1.0 - fouling_ref_pct/100.0)

# Loads
Q_psy_tgt_kW = (mdot_air * (air_in_props['h'] - air_out_tgt['h']))/1000.0

Q_freon_kW = None
if COOLPROP and ref in CP.get_global_param_string("fluids_list").split(','):
    try:
        T_evK = K(T_sat_ev); P_ev = CP.PropsSI("P","T",T_evK,"Q",0.0,ref)
        T_ck  = K(T_cond);   P_ck = CP.PropsSI("P","T",T_ck,"Q",0.0,ref)
        h_liq_sc = CP.PropsSI("H","T",K(T_cond) - Subcool, "P", P_ck, ref)
        h_out = CP.PropsSI("H","T", K(T_sat_ev + SH_out), "P", P_ev, ref)
        Q_freon_kW = mdot_ref * (h_out - h_liq_sc) / 1000.0
    except Exception:
        Q_freon_kW = None

# ε–NTU requirement based on design capacity
def UA_required_eNTU(Q_kW, mdot_air, cp_air, T_air_in_C, T_sat_ev_C):
    Q = Q_kW*1000.0
    C_air = mdot_air*cp_air
    DTstar = (T_air_in_C - T_sat_ev_C)
    if DTstar <= 0 or C_air <= 0: return None, None, None
    eps = min(max(Q/(C_air*DTstar), 1e-6), 0.999999)
    NTU = -math.log(1.0 - eps)
    UA = NTU * C_air
    return UA, eps, NTU
UA_req, eps_need, NTU_need = UA_required_eNTU(Q_kW, mdot_air, air_in_props['cp'], Tdb_in, T_sat_ev)

# ADP/BPF diagnostic
BPF_est = estimate_bpf(Nr, FPI, v_face)
h_in = air_in_props['h']; h_out_tgt = air_out_tgt['h']
ADP_C = None
if BPF_est < 0.9999:
    h_ADP = (h_out_tgt - BPF_est*h_in)/max(1.0 - BPF_est, 1e-6)
    ADP_C = invert_ADP_from_h(h_ADP)

# ----------------- Display -----------------
left, right = st.columns([1.3,1.0])
with left:
    st.subheader("Air Side & Capacity")
    df_air = pd.DataFrame({
        "Metric":[
            "Design capacity (kW)","Airflow (m³/h)","Face area (m²)","Face velocity (m/s)",
            "Entering DB/RH","Target OFF DB/RH",
            "Fin type","Rows","Tubes/row","Total tubes","Fins (count)","FPI","Fin thickness (mm)","Tube OD (mm)","St (mm)","Sl (mm)",
            "Ao (m²)","A_fin (m²)","A_tube (m²)","η_f","η_o","h_air used (W/m²K)","UA_air (W/K)","UA_req (W/K)",
            "ΔP_air (Pa)","V_face (m/s)","Vmax (m/s)","Re_air","U_air (W/m²K)","h_air (W/m²K)"
        ],
        "Value":[
            f"{Q_kW:,.1f}", f"{airflow_m3h:,.0f}", f"{geom['face_area']:.3f}", f"{v_face:.2f}",
            f"{air_in_props['T']:.1f} °C / {air_in_props.get('RH',0):.0f} %", f"{air_out_tgt['T']:.1f} °C / {air_out_tgt.get('RH',0):.0f} %",
            fin_type, f"{Nr}", f"{geom['N_tpr']}", f"{geom['N_tubes']}", f"{geom['fins']}", f"{FPI:.0f}", f"{(tf/MM):.2f}", f"{(Do/MM):.3f}", f"{(St/MM):.1f}", f"{(Sl/MM):.1f}",
            f"{Ao:.2f}", f"{geom['A_fin']:.2f}", f"{geom['A_bare']:.2f}", f"{eta_f:.3f}", f"{eta_o:.3f}", f"{h_air:.1f}", f"{UA_air:,.0f}", f"{UA_req:,.0f}" if UA_req else "—",
            f"{dP_air:,.0f}", f"{v_face:.2f}", f"{meta.get('Vmax',0):.2f}", f"{meta.get('Re', meta.get('Re_h', 0)):.0f}", f"{Uo_air_only:.1f}", f"{h_air_eff:.1f}"
        ]
    })
    st.dataframe(df_air, use_container_width=True)

    # ADP/BPF diagnostics
    rows = [
        ("ADP (°C)", f"{ADP_C:.1f}" if ADP_C is not None else "—"),
        ("BPF (est.)", f"{BPF_est:.3f}"),
        ("h_in (kJ/kg_da)", f"{h_in/1000.0:.2f}"),
        ("h_out target (kJ/kg_da)", f"{h_out_tgt/1000.0:.2f}")
    ]
    st.markdown("**ADP/BPF (wet-coil diagnostic)**")
    st.dataframe(pd.DataFrame(rows, columns=["Item","Value"]), use_container_width=True)

with right:
    st.subheader("Refrigerant Side")
    df_ref = pd.DataFrame({
        "Metric":[
            "Refrigerant","ṁ_ref (kg/s)","T_evap sat (°C)","Superheat (K)","T_cond sat (°C)","Subcool (K)",
            "G (kg/m²·s)","Re_ref_mix","Re_ref_vapor","h_ref (W/m²K)",
            "ΔP_two-phase (kPa)","ΔP_vapor (kPa)","ΔP_total (kPa)"
        ],
        "Value":[
            ref, f"{mdot_ref:.3f}", f"{T_sat_ev:.1f}", f"{SH_out:.1f}", f"{T_cond:.1f}", f"{Subcool:.1f}",
            f"{G:,.0f}" if np.isfinite(G) else "—",
            f"{Re_ref_mix:.0f}" if 'Re_ref_mix' in locals() and Re_ref_mix is not None else "—",
            f"{Re_ref_vap:.0f}" if 'Re_ref_vap' in locals() and Re_ref_vap is not None else "—",
            f"{h_ref_eff:.0f}",
            f"{(dp_tp or 0)/1000:.2f}" if dp_tp is not None else "—",
            f"{(dp_vap or 0)/1000:.2f}" if dp_vap is not None else "—",
            f"{(dp_ref_total or 0)/1000:.2f}" if dp_ref_total is not None else "—"
        ]
    })
    st.dataframe(df_ref, use_container_width=True)

    # ---------- Export buttons ----------
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_air.to_excel(writer, index=False, sheet_name="Air_UA")
            pd.DataFrame(rows, columns=["Item","Value"]).to_excel(writer, index=False, sheet_name="ADP_BPF")
            df_ref.to_excel(writer, index=False, sheet_name="Refrigerant")
        xlsx_bytes = buffer.getvalue()
    st.download_button("Download XLSX", data=xlsx_bytes, file_name="evap_coil_corrected.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
