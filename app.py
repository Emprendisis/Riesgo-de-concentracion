
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Riesgo por Concentración — App", layout="wide")

# ---------------------------
# Utilidades
# ---------------------------

def leer_archivo(archivo):
    """Lee CSV o Excel. Para Excel usa openpyxl si está disponible. Devuelve None si falla (sin colgar la app)."""
    if archivo is None:
        return None
    name = (archivo.name or "").lower()

    # CSV
    if name.endswith(".csv"):
        try:
            return pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No pude leer el CSV: {e}")
            return None

    # Excel
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        # Intento con openpyxl (requerido por pandas para .xlsx)
        try:
            return pd.read_excel(archivo, engine="openpyxl")
        except ImportError:
            st.error("Falta la dependencia 'openpyxl' para leer Excel. Sube un CSV o instala openpyxl.")
            return None
        except Exception as e1:
            # Fallback genérico
            try:
                return pd.read_excel(archivo)
            except Exception as e2:
                st.error(f"No pude leer el Excel: {e2}")
                return None

    st.error("Formato no soportado. Sube .csv o .xlsx")
    return None


def detectar_tabla_exposiciones(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta localizar la tabla con cabeceras reales. Siempre retorna algo (o DataFrame vacío)."""
    if df is None or df.empty:
        return pd.DataFrame()

    # 1) Buscar cabecera con 'Núm cliente' o variantes (primeras 50 filas)
    header_idx = None
    scan_rows = min(50, len(df))
    for i in range(scan_rows):
        row = df.iloc[i].astype(str).str.strip().str.lower()
        if (row == "núm cliente").any() or (row == "num cliente").any():
            header_idx = i
            break

    if header_idx is not None:
        sub = df.iloc[header_idx:, :].copy()
        sub.columns = sub.iloc[0]
        sub = sub.iloc[1:].reset_index(drop=True)
        return sub

    # 2) Intento por posición como tu ejemplo (rápido, acotado)
    if df.shape[1] >= 15 and df.shape[0] >= 13:
        sub = df.iloc[12:100, 10:15].copy()
        sub.columns = sub.iloc[0]
        sub = sub.iloc[1:].reset_index(drop=True)
        return sub

    # 3) Fallback: primera fila como cabecera
    sub = df.copy()
    sub.columns = sub.iloc[0]
    sub = sub.iloc[1:].reset_index(drop=True)
    return sub


def preparar_tabla(tabla: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres y tipos. No lanza excepciones."""
    if tabla is None or tabla.empty:
        return pd.DataFrame()

    rename_map = {}
    for col in tabla.columns:
        c = str(col).strip().lower()
        if ("núm" in c or "num" in c) and "cliente" in c:
            rename_map[col] = "Núm cliente"
        if "saldo" in c or "exposición" in c or "exposicion" in c:
            rename_map[col] = "Exposición (000)"
        if "participación" in c or "participacion" in c or "%" in c:
            rename_map[col] = "Participación (%)"
        if "contribución" in c and "riesgo" in c:
            rename_map[col] = "Contribución al riesgo (000)"

    t = tabla.rename(columns=rename_map).copy()

    keep = [c for c in ["Núm cliente", "Exposición (000)", "Participación (%)", "Contribución al riesgo (000)"] if c in t.columns]
    if not keep:
        return pd.DataFrame()

    t = t[keep]

    # Tipos
    for c in keep:
        t[c] = pd.to_numeric(t[c], errors="coerce")

    # Filtrar exposición válida
    if "Exposición (000)" in t.columns:
        t = t.dropna(subset=["Exposición (000)"], how="any")

    # Asegurar Núm cliente
    if "Núm cliente" not in t.columns and "Exposición (000)" in t.columns:
        t.insert(0, "Núm cliente", np.arange(1, len(t) + 1))

    # Ordenar
    if "Exposición (000)" in t.columns:
        t = t.sort_values("Exposición (000)", ascending=False).reset_index(drop=True)

    return t


def calcular_metricas(df_exp: pd.DataFrame):
    """Calcula HHI, Adelman y CRn. No se cuelga si faltan columnas."""
    if df_exp is None or df_exp.empty or "Exposición (000)" not in df_exp.columns:
        return {}, np.array([])

    total = df_exp["Exposición (000)"].sum()
    if total <= 0:
        return {"Total exposición (000)": float(total)}, np.array([])

    shares = df_exp["Exposición (000)"] / total
    hhi = float((shares ** 2).sum())
    adelman = float(1.0 / hhi) if hhi > 0 else np.nan
    metrics = {
        "Total exposición (000)": float(total),
        "HHI": hhi,
        "Adelman (1/HHI)": adelman,
        "CR1": float(shares.max()),
        "CR3": float(shares.nlargest(3).sum()) if len(shares) >= 3 else np.nan,
        "CR5": float(shares.nlargest(5).sum()) if len(shares) >= 5 else np.nan,
        "CR10": float(shares.nlargest(10).sum()) if len(shares) >= 10 else np.nan,
    }
    return metrics, shares.to_numpy()


def gini_from_shares(shares: np.ndarray):
    if shares is None or len(shares) == 0:
        return np.nan
    s = np.sort(shares)
    cum = np.cumsum(s)
    n = len(s)
    area = np.sum(cum) / n
    return float(1 - 2*area)


def stress_top1(df_exp: pd.DataFrame):
    if df_exp is None or df_exp.empty or "Exposición (000)" not in df_exp.columns:
        return pd.DataFrame(), {}
    idx = df_exp["Exposición (000)"].idxmax()
    stressed = df_exp.drop(index=idx).reset_index(drop=True)
    met, _ = calcular_metricas(stressed)
    return stressed, met


# ---------------------------
# Interfaz
# ---------------------------

st.title("Riesgo por Concentración — Cartera de Créditos")
st.caption("Sube tu base y presiona **Procesar**.")

with st.sidebar:
    st.header("Entrada de datos")
    archivo = st.file_uploader("Archivo (.csv o .xlsx)", type=["csv","xlsx","xlsm","xls"])
    procesar = st.button("Procesar")

# Estado
if "tabla" not in st.session_state:
    st.session_state["tabla"] = pd.DataFrame()

# Procesamiento bajo botón
if procesar:
    df_raw = leer_archivo(archivo)
    if df_raw is None or df_raw.empty:
        st.error("No pude leer datos del archivo.")
    else:
        tabla_detectada = detectar_tabla_exposiciones(df_raw)
        tabla = preparar_tabla(tabla_detectada)
        if tabla.empty or "Exposición (000)" not in tabla.columns:
            st.error("No encontré una columna de montos. Usa 'Exposición (000)' o sube CSV con dos columnas: Núm cliente, Exposición (000).")
        else:
            st.session_state["tabla"] = tabla

tabla = st.session_state["tabla"]
if tabla.empty:
    st.info("Aún no hay datos procesados.")
    st.stop()

# Salidas
st.subheader("Tabla de exposiciones")
st.dataframe(tabla, use_container_width=True)

metricas, shares = calcular_metricas(tabla)
if not metricas:
    st.error("No fue posible calcular métricas.")
    st.stop()

gini = gini_from_shares(shares)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total exposición (000)", f"{metricas.get('Total exposición (000)', 0):.0f}")
m2.metric("HHI", f"{metricas.get('HHI', np.nan):.6f}" if 'HHI' in metricas else "NA")
m3.metric("Adelman (1/HHI)", f"{metricas.get('Adelman (1/HHI)', np.nan):.2f}" if 'Adelman (1/HHI)' in metricas else "NA")
m4.metric("Gini", f"{gini:.3f}" if not np.isnan(gini) else "NA")

if 'HHI' in metricas and not np.isnan(metricas['HHI']):
    if metricas['HHI'] < 0.10:
        sem = "🟢 Baja"
    elif metricas['HHI'] <= 0.18:
        sem = "🟠 Media"
    else:
        sem = "🔴 Alta"
    st.write(f"**Semáforo HHI:** {sem}")

st.subheader("Razones de concentración")
cr_df = pd.DataFrame({
    "Métrica": ["CR1","CR3","CR5","CR10"],
    "Valor": [metricas.get("CR1"), metricas.get("CR3"), metricas.get("CR5"), metricas.get("CR10")]
})
st.dataframe(cr_df, use_container_width=True)

# Curva de Lorenz nativa
s = np.sort(shares) if shares is not None and len(shares) else np.array([])
cum = np.concatenate([[0], np.cumsum(s)]) if len(s) else np.array([0.0])
x = np.linspace(0, 1, len(cum))
lorenz_df = pd.DataFrame({"Proporción acumulada de créditos": x, "Proporción acumulada de exposición": cum})
st.subheader("Curva de Lorenz")
st.line_chart(lorenz_df.set_index("Proporción acumulada de créditos"))

# Stress Top-1
stress_tbl, stress_metrics = stress_top1(tabla)
st.subheader("Stress: default del mayor deudor (Top-1)")
if stress_metrics:
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Antes:**")
        st.dataframe(pd.DataFrame(metricas, index=["Actual"]).T)
    with c2:
        st.write("**Después:**")
        st.dataframe(pd.DataFrame(stress_metrics, index=["Stress"]).T)

    # Lorenz post-stress
    _, shares_stress = calcular_metricas(stress_tbl)
    s2 = np.sort(shares_stress) if shares_stress is not None and len(shares_stress) else np.array([])
    cum2 = np.concatenate([[0], np.cumsum(s2)]) if len(s2) else np.array([0.0])
    x2 = np.linspace(0, 1, len(cum2))
    lorenz2 = pd.DataFrame({"Proporción acumulada de créditos": x2, "Proporción acumulada de exposición": cum2})
    st.subheader("Curva de Lorenz post-stress")
    st.line_chart(lorenz2.set_index("Proporción acumulada de créditos"))
else:
    st.info("No se pudo calcular el escenario de stress.")
