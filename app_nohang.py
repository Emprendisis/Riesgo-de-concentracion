
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Riesgo por Concentración — App (NoHang)", layout="wide")

# ---------- Utilidades seguras ----------

def leer_archivo(archivo):
    if archivo is None:
        return None, "Sin archivo"
    name = (archivo.name or "").lower()
    # CSV primero (sin dependencias)
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(archivo)
            return df, "csv"
        except Exception as e:
            st.error(f"CSV inválido: {e}")
            return None, "error"
    # Excel con openpyxl si está disponible
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            df = pd.read_excel(archivo, engine="openpyxl")
            return df, "excel"
        except ImportError:
            st.error("No está instalado 'openpyxl'. Sube un CSV.")
            return None, "error"
        except Exception as e:
            st.error(f"Excel inválido: {e}")
            return None, "error"
    st.error("Formato no soportado. Usa CSV o XLSX.")
    return None, "error"

def detectar_tabla(df):
    if df is None or df.empty:
        return pd.DataFrame()
    # Buscar cabecera en primeras 20 filas máximo
    header_idx = None
    for i in range(min(20, len(df))):
        row = df.iloc[i].astype(str).str.strip().str.lower()
        if (row == "núm cliente").any() or (row == "num cliente").any():
            header_idx = i
            break
    if header_idx is not None:
        sub = df.iloc[header_idx:, :].copy()
        sub.columns = sub.iloc[0]
        sub = sub.iloc[1:].reset_index(drop=True)
        return sub
    # Fallback simple
    sub = df.copy()
    sub.columns = sub.iloc[0]
    sub = sub.iloc[1:].reset_index(drop=True)
    return sub

def preparar_tabla(tabla):
    if tabla is None or tabla.empty:
        return pd.DataFrame()
    rename = {}
    for col in tabla.columns:
        c = str(col).strip().lower()
        if ("núm" in c or "num" in c) and "cliente" in c:
            rename[col] = "Núm cliente"
        if "saldo" in c or "exposición" in c or "exposicion" in c:
            rename[col] = "Exposición (000)"
        if "participación" in c or "participacion" in c or "%" in c:
            rename[col] = "Participación (%)"
        if "contribución" in c and "riesgo" in c:
            rename[col] = "Contribución al riesgo (000)"
    t = tabla.rename(columns=rename).copy()
    keep = [c for c in ["Núm cliente","Exposición (000)","Participación (%)","Contribución al riesgo (000)"] if c in t.columns]
    if not keep:
        return pd.DataFrame()
    t = t[keep]
    for c in keep:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    if "Exposición (000)" in t.columns:
        t = t.dropna(subset=["Exposición (000)"], how="any")
        t = t.sort_values("Exposición (000)", ascending=False).reset_index(drop=True)
    if "Núm cliente" not in t.columns and "Exposición (000)" in t.columns:
        t.insert(0, "Núm cliente", np.arange(1, len(t)+1))
    return t

def calcular_metricas(df):
    if df is None or df.empty or "Exposición (000)" not in df.columns:
        return {}, np.array([])
    total = float(df["Exposición (000)"].sum())
    if total <= 0:
        return {"Total exposición (000)": total}, np.array([])
    shares = df["Exposición (000)"] / total
    hhi = float((shares**2).sum())
    metrics = {
        "Total exposición (000)": total,
        "HHI": hhi,
        "Adelman (1/HHI)": float(1/hhi) if hhi>0 else np.nan,
        "CR1": float(shares.max()),
        "CR3": float(shares.nlargest(3).sum()) if len(shares)>=3 else np.nan,
        "CR5": float(shares.nlargest(5).sum()) if len(shares)>=5 else np.nan,
        "CR10": float(shares.nlargest(10).sum()) if len(shares)>=10 else np.nan,
    }
    return metrics, shares.to_numpy()

def gini(shares):
    if shares is None or len(shares)==0:
        return np.nan
    s = np.sort(shares)
    cum = np.cumsum(s); n=len(s)
    return float(1 - 2*(cum.sum()/n))

# ---------- UI ----------

st.title("Riesgo por Concentración — NoHang")
st.caption("Sube archivo y pulsa **Procesar**.")

with st.sidebar:
    archivo = st.file_uploader("Archivo (.csv o .xlsx)", type=["csv","xlsx","xlsm","xls"])
    procesar = st.button("Procesar")

if "tabla" not in st.session_state:
    st.session_state["tabla"] = pd.DataFrame()

if procesar:
    df_raw, kind = leer_archivo(archivo)
    if df_raw is None or df_raw.empty:
        st.session_state["tabla"] = pd.DataFrame()
    else:
        tabla = preparar_tabla(detectar_tabla(df_raw))
        st.session_state["tabla"] = tabla

tabla = st.session_state["tabla"]

if tabla.empty:
    st.info("Sin datos procesados. Sube CSV/XLSX y pulsa **Procesar**.")
else:
    st.subheader("Tabla")
    st.dataframe(tabla, use_container_width=True)

    m, shares = calcular_metricas(tabla)
    if not m:
        st.error("Faltan columnas mínimas ('Exposición (000)').")
    else:
        g = gini(shares)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total exposición (000)", f"{m['Total exposición (000)']:.0f}")
        c2.metric("HHI", f"{m['HHI']:.6f}")
        c3.metric("Adelman (1/HHI)", f"{m['Adelman (1/HHI)']:.2f}")
        c4.metric("Gini", f"{g:.3f}" if not np.isnan(g) else "NA")

        sem = "🟢 Baja" if m["HHI"]<0.10 else ("🟠 Media" if m["HHI"]<=0.18 else "🔴 Alta")
        st.write(f"**Semáforo HHI:** {sem}")

        st.subheader("CRn")
        st.dataframe(pd.DataFrame({"Métrica":["CR1","CR3","CR5","CR10"],
                                   "Valor":[m["CR1"],m["CR3"],m["CR5"],m["CR10"]]}))

        # Lorenz (nativo)
        s = np.sort(shares); cum = np.concatenate([[0], np.cumsum(s)]) if len(s) else np.array([0.0])
        x = np.linspace(0,1,len(cum))
        st.subheader("Curva de Lorenz")
        st.line_chart(pd.DataFrame({"x":x,"y":cum}).set_index("x"))

        # Stress Top-1
        st.subheader("Stress Top-1")
        if len(tabla)>0:
            idx = tabla["Exposición (000)"].idxmax()
            stressed = tabla.drop(index=idx).reset_index(drop=True)
            m2,_ = calcular_metricas(stressed)
            colA,colB = st.columns(2)
            with colA: st.dataframe(pd.DataFrame(m, index=["Actual"]).T)
            with colB: st.dataframe(pd.DataFrame(m2, index=["Stress"]).T)
