
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


# --- Additional psychrometric helpers (DB+WB <-> W/RH) ---
def RH_from_T_W(T_C, W, P=P_ATM):
    Pv = W*P/(0.62198 + W)
    Ps = psat_water_Pa(T_C)
    RH = max(min(100.0*Pv/max(Ps,1e-9), 100.0), 0.1)
    return RH

def humidity_ratio_from_T_WB(Tdb_C: float, Twb_C: float, P: float = P_ATM) -> float:
    """
    First-cut ASHRAE-style approximation for humidity ratio from dry-bulb and wet-bulb at pressure P.
    T in °C. Returns W in kg/kg dry air.
    W ≈ W_s(Twb) + [1.006*(Tdb - Twb)] / [2501 - 2.381*Twb]
    """
    Ps_wb = psat_water_Pa(Twb_C)
    Ws_wb = 0.62198 * Ps_wb / max(P - Ps_wb, 1.0)
    W = Ws_wb + (1.006*(Tdb_C - Twb_C)) / max(2501.0 - 2.381*Twb_C, 1e-3)
    return float(max(W, 1e-6))

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
if af_mode == "Face velocity (m/s)":
    v_face = st.number_input("Face velocity (m/s)", 0.2, 6.0, 3.0, 0.1)
    vol_flow = v_face * face_area
    airflow_m3h = vol_flow * 3600.0
else:
    airflow_m3h = st.number_input("Airflow (m³/h)", 500.0, 300000.0, 20000.0, 100.0)
    vol_flow = airflow_m3h / 3600.0
    v_face = vol_flow / max(face_area, 1e-9)
st.caption(f"Airflow ≈ {airflow_m3h:,.0f} m³/h   |   Face velocity ≈ {v_face:.2f} m/s")
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
dP_air = air_dp_from_meta(air_in_props, meta)

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

    # ---------- Export buttons ----------
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_air.to_excel(writer, index=False, sheet_name="Air_UA")
            df_psy.to_excel(writer, index=False, sheet_name="ADP_BPF")
            df_ref.to_excel(writer, index=False, sheet_name="Refrigerant")
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
