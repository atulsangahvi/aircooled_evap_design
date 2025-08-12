
# Air Evaporator (DX coil) — ε–NTU + ADP/BF moist-air model
# Version: v1 (2025-08-12)
import math, io, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from CoolProp.CoolProp import PropsSI, HAPropsSI, get_global_param_string

APP_VERSION = "Evap v1 (2025-08-12)"

st.title("❄️ Air Evaporator (DX Coil) — ε–NTU with Moist Air (ADP/BF)")
st.caption("Same geometry style as the condenser tool. Calculates air outlet T & RH using a wet-coil bypass-factor model.")

# ---------------- Inputs ----------------
st.header("🧩 Geometry")
tube_od_mm     = st.number_input("Tube OD (mm)", 0.1, 50.0, 9.525, 0.001)
tube_thk_mm    = st.number_input("Tube Wall Thickness (mm)", 0.01, 2.0, 0.35, 0.01)
tube_pitch_mm  = st.number_input("Vertical Tube Pitch (mm)", 5.0, 60.0, 25.4, 0.1)
row_pitch_mm   = st.number_input("Row Pitch (depth) (mm)",    5.0, 60.0, 22.0, 0.1)
fpi            = st.number_input("Fins per inch (FPI)", 4, 24, 12, 1)
fin_thk_mm     = st.number_input("Fin thickness (mm)", 0.05, 0.5, 0.12, 0.01)
fin_material   = st.selectbox("Fin material", ["Aluminum", "Copper"])
face_width_m   = st.number_input("Coil face width (m)", 0.1, 5.0, 2.0, 0.01)
face_height_m  = st.number_input("Coil face height (m)", 0.1, 5.0, 1.20, 0.01)
num_rows       = st.number_input("Rows (depth) available", 1, 60, 4, 1)
free_area_percent = st.number_input("Free Area Percentage (%)", min_value=0.0, max_value=100.0, value=75.0, format="%.1f")
num_feeds      = st.number_input("Number of circuits (parallel)", 1, 256, 6, 1)

st.header("❄️ Refrigerant Inputs")
fluid_list = get_global_param_string("FluidsList").split(',')
refrigerants = sorted([f for f in fluid_list if f.startswith("R")])
fluid = st.selectbox("Refrigerant", refrigerants, index=refrigerants.index("R134a") if "R134a" in refrigerants else 0)
T4 = st.number_input("Inlet Subcooled Temp T4 (°C)", value=30.0, format="%.2f")
T_evap = st.number_input("Evaporating Temp (°C)", value=5.0, format="%.2f")
SH_out = st.number_input("Outlet Superheat (K)", value=8.0, format="%.1f")
m_dot_total = st.number_input("Refrigerant Mass Flow (kg/s)", min_value=0.01, value=0.60, format="%.4f")

st.header("🌬️ Air Inputs")
air_temp = st.number_input("Air Inlet Dry-bulb (°C)", value=27.0, format="%.2f")
air_rh   = st.number_input("Air Inlet RH (%)", min_value=0.0, max_value=100.0, value=40.0, format="%.1f")
airflow_cmh = st.number_input("Air Flow (m³/hr)", min_value=1.0, value=28000.0, format="%.1f")

st.header("⚙️ Assumptions & Fouling")
Rf_o = st.number_input("Air-side fouling (m²·K/W) on A_o basis", 0.0, 0.005, 0.0002, 0.0001, format="%.5f")
Rf_i = st.number_input("Refrigerant-side fouling (m²·K/W) on A_i basis", 0.0, 0.001, 0.00005, 0.00001, format="%.5f")
k_tube = 385.0 if fin_material == "Copper" else 385.0  # tube is copper by default
cond_enhance = st.number_input("Evap/boil h_i enhancement factor", 0.8, 5.0, 1.5, 0.1)

# -------------- Derived geometry --------------
tube_od = tube_od_mm/1000.0
tube_thk = tube_thk_mm/1000.0
tube_id = max(1e-4, tube_od - 2.0*tube_thk)
pt_vert = tube_pitch_mm/1000.0
pl_depth = row_pitch_mm/1000.0
A_frontal = face_width_m * face_height_m
fins_per_m = fpi * 39.37007874
s_fin = 1.0/fins_per_m if fins_per_m>0 else 1e9

# approximate external areas
N_row = max(1, int(face_height_m / max(1e-9, pt_vert)))
tubes_per_row = N_row
N_tubes_total = tubes_per_row * num_rows
A_tube_ext = math.pi * tube_od * (face_width_m * N_tubes_total)  # external tube area
fin_perimeter = 2.0*(face_width_m + face_height_m)
N_fins = int(fins_per_m * face_width_m) * num_rows
A_fin_raw = N_fins * fin_perimeter * fin_thk_mm/1000.0
A_total = A_tube_ext + A_fin_raw

# External h (simplified: flat-plate correlation)
airflow_m3s = airflow_cmh/3600.0
free_area = A_frontal * (free_area_percent/100.0)
V_face = airflow_m3s / max(1e-9, A_frontal)
V_free = airflow_m3s / max(1e-9, free_area)

# Moist air properties (with fallback)
P_amb = 101325.0
T_air_K = air_temp + 273.15
RH_frac = max(1e-4, min(0.9999, air_rh/100.0))
try:
    w_in = HAPropsSI("W","T",T_air_K,"P",P_amb,"R",RH_frac)
    rho_air = HAPropsSI("Rho","T",T_air_K,"P",P_amb,"R",RH_frac)
    h_in = HAPropsSI("H","T",T_air_K,"P",P_amb,"R",RH_frac)
    cp_da = (HAPropsSI("H","T",T_air_K+0.1,"P",P_amb,"R",RH_frac) - h_in)/0.1  # J/kg_dry-K
except Exception:
    w_in = 0.0
    rho_air = P_amb/(287.058*T_air_K)
    h_in = 1.006e3*air_temp  # approx
    cp_da = 1006.0

# refrigerant states
P_evap = PropsSI("P","T",T_evap+273.15,"Q",0,fluid)
h4 = PropsSI("H","P",P_evap,"T",T4+273.15,fluid)        # subcooled in
h5 = PropsSI("H","P",P_evap,"Q",0,fluid)               # sat liquid
h6 = PropsSI("H","P",P_evap,"Q",1,fluid)               # sat vapor
T1_out = T_evap + SH_out                               # superheated out
h1 = PropsSI("H","P",P_evap,"T",T1_out+273.15,fluid)   # superheated out

Q_desub_W = m_dot_total * max(0.0, (h5 - h4))
Q_evap_W  = m_dot_total * max(0.0, (h6 - h5))
Q_super_W = m_dot_total * max(0.0, (h1 - h6))

# Simple air-side HTC estimate
mu_air = 1.8e-5
k_air  = 0.026
rho_guess = rho_air
Re_ext = max(1e-9, rho_guess*V_free*tube_od/mu_air)
Nu = 0.3 + (0.62*(Re_ext**0.5)*(0.71**(1/3))) / ((1+(0.4/0.71)**(2/3))**0.25) * (1+(Re_ext/282000.0)**(5/8))**(4/5)
h_air = Nu * k_air / max(1e-9, tube_od)

# Fin efficiency (simple circular fin approximation, constant h_air)
k_fin = 235.0 if fin_material=="Aluminum" else 385.0
m_fin = math.sqrt(2*h_air/(k_fin*fin_thk_mm/1000.0 + 1e-9))
eta_fin = math.tanh(m_fin*(s_fin/2.0)) / (m_fin*(s_fin/2.0)) if fins_per_m>0 else 1.0
eta_o = 1.0 - (A_fin_raw/max(1e-9,A_total))*(1.0 - eta_fin)

# Internal hi (evaporator)
def h_i_single_phase(mu, k, cp, rho, Re_i):
    Pr = cp*mu/max(1e-12,k)
    Nu = 0.023*(max(2300.0,Re_i)**0.8)*(max(0.6,Pr)**0.4)
    return Nu * k / max(1e-9, tube_id)

# estimate properties at evap temp
rho_l = PropsSI("D","P",P_evap,"Q",0,fluid)
mu_l  = PropsSI("V","P",P_evap,"Q",0,fluid)
k_l   = PropsSI("L","P",P_evap,"Q",0,fluid)
cp_l  = PropsSI("C","P",P_evap,"Q",0,fluid)
rho_v = PropsSI("D","P",P_evap,"Q",1,fluid)
mu_v  = PropsSI("V","P",P_evap,"Q",1,fluid)
k_v   = PropsSI("L","P",P_evap,"Q",1,fluid)
cp_v  = PropsSI("C","P",P_evap,"Q",1,fluid)
# per-circuit
m_dot_circuit = m_dot_total / max(1, num_feeds)
A_i = math.pi*(tube_id**2)/4.0
G_i = m_dot_circuit / max(1e-12, A_i)
u_i_v = G_i / max(1e-12, rho_v)
u_i_l = G_i / max(1e-12, rho_l)
Re_v_i = rho_v * u_i_v * tube_id / max(1e-12, mu_v)
Re_l_i = rho_l * u_i_l * tube_id / max(1e-12, mu_l)

h_i_liq = h_i_single_phase(mu_l,k_l,cp_l,rho_l,Re_l_i)
h_i_vap = h_i_single_phase(mu_v,k_v,cp_v,rho_v,Re_v_i)
h_i_boil = max(400.0, cond_enhance*h_i_liq)

# Ao/Ai and Uo per zone
A_o_per_m = math.pi*tube_od
A_i_per_m = math.pi*tube_id
Ao_Ai = A_o_per_m / max(1e-12, A_i_per_m)
def Uo_from(h_air_local, h_i_local):
    invU = (1.0/max(1e-9, eta_o*h_air_local)) + Rf_o + Ao_Ai*((1.0/max(1e-9,h_i_local))+Rf_i) + math.log(tube_od/max(1e-12,tube_id))/(2.0*math.pi*k_tube) / max(1e-12, A_o_per_m)
    return 1.0/invU

Uo_desub = Uo_from(h_air, h_i_liq)
Uo_boil  = Uo_from(h_air, h_i_boil)
Uo_super = Uo_from(h_air, h_i_vap)

# ε–NTU per zone (air vs refrigerant with phase change)
def eps_cap_crossflow_one_mixed(Cr, is_phase_change):
    if is_phase_change or Cr <= 1e-9:
        return 1.0
    Cr = max(1e-9, min(0.999999, Cr))
    # one mixed crossflow cap effectiveness limit (approx upper bound)
    return (1.0 - math.exp(-1.0/Cr*(1.0-math.exp(-Cr))))

def eps_crossflow_one_mixed(NTU, Cr):
    if Cr <= 1e-9:
        return 1.0 - math.exp(-NTU)
    return 1.0 - math.exp((math.exp(-Cr*NTU)-1.0)/Cr)

def invert_for_NTU(Qreq_W, Cmin_WK, dT_in, Uo_zone):
    NTU_cap = 20.0
    lo, hi = 0.0, NTU_cap
    for _ in range(40):
        mid = 0.5*(lo+hi)
        eps = 1.0 - math.exp(-mid)
        Q = eps * Cmin_WK * dT_in
        if Q < Qreq_W: lo = mid
        else: hi = mid
    NTU = 0.5*(lo+hi)
    Areq = NTU * Cmin_WK / max(1e-12, Uo_zone)
    return NTU, Areq

# Air capacity rate (moist) at inlet
h_surf_star = HAPropsSI("H","T",T_evap+273.15,"P",P_amb,"R",1.0)
cp_eq = (h_in - h_surf_star)/max(1e-6, (air_temp - T_evap))  # J/kg-K on moist basis (Lewis≈1)
m_da = airflow_m3s * rho_air / max(1e-12, (1.0+w_in))        # kg_dry/s
C_air_WK = m_da * cp_eq                                      # W/K (enthalpy-based)

zones = [
    ("Desubcool",  Q_desub_W,  Uo_desub,  T4),
    ("Boiling",    Q_evap_W,   Uo_boil,   T_evap),
    ("Superheat",  Q_super_W,  Uo_super,  T1_out),
]

results = []
air_T_in = air_temp
for name, Qreq_W, Uo_zone, T_hot_in in zones:
    if Qreq_W <= 1e-9:
        results.append((name, 0.0, 0.0, 0.0, 0.0, air_T_in, Uo_zone, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        continue
    # For phase change (boiling) treat Cr≈0, else use C_air vs a pseudo refrigerant sensible capacity
    Cmin_WK = C_air_WK
    Cr = 0.0
    dT_in = max(0.1, (air_T_in - T_hot_in))  # air hot, surface cold
    NTU, A_req = invert_for_NTU(Qreq_W, Cmin_WK, dT_in, Uo_zone)
    area_per_row = max(1e-9, A_total/max(1, num_rows))
    rows_required_zone = A_req / area_per_row
    length_per_circuit_zone = rows_required_zone * face_width_m

    eps_used = (1.0 - math.exp(-NTU))
    Q_actual_W = eps_used * Cmin_WK * dT_in
    air_T_out = air_T_in - Q_actual_W / max(1e-9, m_da*1006.0)  # approximate temp drop using cp~1006 J/kg-K
    results.append((name, Qreq_W/1000.0, A_req, length_per_circuit_zone, rows_required_zone, air_T_out, Uo_zone, NTU, Cr, Cmin_WK/1000.0, dT_in, Q_actual_W/1000.0, eps_used))
    air_T_in = air_T_out

st.header("📊 Zone Results (Evaporator)")
df = pd.DataFrame(results, columns=[
    "Zone","Q_req (kW)","Area needed A_req (m²)","Serpentine length per circuit (m)","Rows required",
    "Air out (°C)","Uo (W/m²K)","NTU","C_r","C_min (kW/K)","ΔT_in (°C)","Q_actual (kW)","ε_used"
])
st.dataframe(df.style.format({
    "Q_req (kW)":"{:.2f}","Area needed A_req (m²)":"{:.2f}",
    "Serpentine length per circuit (m)":"{:.2f}","Rows required":"{:.3f}",
    "Air out (°C)":"{:.2f}","Uo (W/m²K)":"{:.1f}","NTU":"{:.2f}",
    "C_r":"{:.3f}","C_min (kW/K)":"{:.3f}","ΔT_in (°C)":"{:.2f}","Q_actual (kW)":"{:.2f}","ε_used":"{:.3f}"
}))

# ---------------- Wet coil outlet using ADP/Bypass Factor ----------------
UA_total = sum(row[1]*1000.0 / max(1e-12, row[10]) for row in results if row[1]>0 and row[10]>0)  # sum Q/ΔT_in ≈ Cmin*eps = UA*ΔT_lm approx
# safer: UA_total = sum(Uo*Area)
UA_total = sum(row[6]*row[2] for row in results)  # Uo * A_req
h_star = h_surf_star
NTU_enthalpy = UA_total / max(1e-9, (m_da))  # Lewis≈1 ⇒ capacity rate ≈ m_da
BF = math.exp(-max(0.0, NTU_enthalpy))
w_star = HAPropsSI("W","T",T_evap+273.15,"P",P_amb,"R",1.0)

T_out = BF*air_temp + (1.0-BF)*T_evap
w_out = BF*w_in + (1.0-BF)*w_star
RH_out = HAPropsSI("R","T",T_out+273.15,"P",P_amb,"W",w_out) * 100.0

st.header("🌡️ Air Outlet (ADP/BF model)")
c1,c2,c3 = st.columns(3)
c1.metric("T_out (°C)", f"{T_out:.2f}")
c2.metric("w_out (kg/kg_dry)", f"{w_out:.5f}")
c3.metric("RH_out (%)", f"{RH_out:.1f}")

st.caption("Notes: Simplified wet-coil model using ADP/Bypass Factor with Lewis≈1 assumption. For detailed segment-wise model, we can iterate along the face area later.")

# PDF export
st.header("🧾 Report")
def dataframe_to_string(df_in: pd.DataFrame, max_rows=1000):
    return df_in.to_string(index=False, max_rows=max_rows)

if st.button("Generate PDF report (Evaporator)"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        lines = [
            f"Date: {dt.datetime.now().isoformat(timespec='seconds')}",
            f"Refrigerant: {fluid}  |  m_dot={m_dot_total:.4f} kg/s  |  T_evap={T_evap:.2f} °C  |  SH_out={SH_out:.1f} K",
            f"Air in: {air_temp:.2f} °C, RH={air_rh:.1f}%, Flow={airflow_cmh:.1f} m³/h",
            f"Geometry: {face_width_m:.2f} x {face_height_m:.2f} m, rows={num_rows}, FPI={fpi}, free area={free_area_percent:.1f}%",
        ]
        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis('off'); plt.text(0.02,0.98,"\n".join(lines),va='top',family='monospace'); pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
        plt.text(0.02,0.98,"Zone Results",va='top',family='monospace',fontsize=12,fontweight='bold')
        plt.text(0.02,0.94, dataframe_to_string(df), va='top', family='monospace', fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
        out_lines = [f"T_out={T_out:.2f} °C,  w_out={w_out:.5f} kg/kg_dry,  RH_out={RH_out:.1f}%",
                     f"UA_total={UA_total:.1f} W/K,  NTU_enthalpy={NTU_enthalpy:.2f},  BF={BF:.3f}"]
        plt.text(0.02,0.98,"Air Outlet (ADP/BF)",va='top',family='monospace',fontsize=12,fontweight='bold')
        plt.text(0.02,0.94, "\n".join(out_lines), va='top', family='monospace', fontsize=10)
        pdf.savefig(fig); plt.close(fig)

    buffer.seek(0)
    st.download_button("Download PDF report", data=buffer.getvalue(),
                       file_name="evaporator_report.pdf", mime="application/pdf")

st.write(f"Version: {APP_VERSION}")
