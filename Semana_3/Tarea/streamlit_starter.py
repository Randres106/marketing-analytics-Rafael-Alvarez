"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    ruta_csv = BASE_DIR / "data" / "uplift_campaign.csv"
    return pd.read_csv(ruta_csv)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # TODO: entrenen el T-learner (dos modelos, igual que en el notebook)
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    #modelo_tratado = ...   # <-- reemplacen esto
    #modelo_control = ...   # <-- reemplacen esto

    modelo_tratado = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption("Análisis de una campaña de Starbucks Rewards para identificar a que clientes conviene contactar, utilizando modelos de Uplift.")

df = load_data()
st.subheader("Resumen de la campaña")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de clientes", f"{len(df):,}")

with col2:
    tratados = (df["treatment_group"] == "treatment").sum()
    st.metric("Clientes tratados", f"{tratados:,}")

with col3:
    control = (df["treatment_group"] == "control").sum()
    st.metric("Grupo de control", f"{control:,}")

# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider(
        "Porcentaje de clientes a contactar",
        min_value=10, 
        max_value=100, 
        value=20, 
        step=10,
        format="%d%%"
    )  

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

st.write("Verificación del T-learner")

st.write("Clientes de entrenamiento:", len(train))
st.write("Clientes de prueba:", len(test))

st.write(
    "Uplift promedio estimado:",
    round(test["uplift_estimado"].mean(), 6)
)

st.dataframe(
    test[
        ["customer_id", "p_tratado",
         "p_control", "uplift_estimado"]
    ].head()
)


# --- TODO: calculen y grafiquen la curva Qini de su T-learner ---
#Función para calcular la curva Qini
def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(
        score_col, ascending=False
    ).reset_index(drop=True)

    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values
    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))

    n_tratados = np.cumsum(es_tratado)
    n_control = np.cumsum(~es_tratado)

    n = len(orden)
    xs, ys = [0.0], [0.0]

    for i in np.linspace(
        max(1, int(n * 0.025)),
        n, steps
    ).astype(int):
        nt = n_tratados[i - 1]
        nc = n_control[i - 1]
        if nt == 0 or nc == 0:
            continue

        tasa_t = acumulado_tratados[i - 1] / nt
        tasa_c = acumulado_control[i - 1] / nc

        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * i)

    return np.array(xs), np.array(ys)


#Coeficiente Qini
def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(
        np.linspace(0, ys[-1], len(xs)), xs
    )

    return (area_modelo - area_azar) / n


#Curva y coeficiente del T-learner
xs_uplift, ys_uplift = curva_qini(test, "uplift_estimado")
qini_uplift = coeficiente_qini(xs_uplift, ys_uplift, len(test))

#Gráfica
st.subheader("Curva Qini del T-learner")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    xs_uplift, ys_uplift,
    color=GREEN, linewidth=3,
    label=f"T-learner (Qini={qini_uplift:+.3f})"
)

ax.plot(
    [0, 1], [0, ys_uplift[-1]],
    color=GRAY, linestyle="--",
    label="Selección aleatoria"
)

ax.set_xlabel("% de la base contactada")
ax.set_ylabel("Respuestas incrementales acumuladas")
ax.set_title("Impacto incremental de la campaña")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

st.metric("Coeficiente Qini", f"{qini_uplift:+.3f}")

# --- TODO: muestren la tabla de políticas (valor por cliente y total) para el % elegido ---
#% de contacto
resultados = []

for pct in range(10, 101, 10):
    n_c = int(len(test) * pct / 100)
    seleccionados = test.sort_values(
        "uplift_estimado",
        ascending=False
    ).head(n_c).index
    valor_cliente, n = valor_de_la_politica(
        test, seleccionados
    )

    resultados.append({
        "Porcentaje": pct,
        "Clientes": n,
        "Valor por cliente": round(valor_cliente, 2),
        "Valor total": round(valor_cliente * n, 2)
    })

tabla_politicas = pd.DataFrame(resultados)

st.subheader("Comparación de políticas de contacto")
st.dataframe(tabla_politicas, use_container_width=True)

#Política más rentable
mejor = tabla_politicas.loc[
    tabla_politicas["Valor total"].idxmax()
]

#Política seleccionada por el slider
seleccion = tabla_politicas.loc[
    tabla_politicas["Porcentaje"] == pct_contactar
].iloc[0]

st.subheader("Resultados de la política seleccionada")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Clientes a contactar",
        f"{int(seleccion['Clientes']):,}"
    )

with col2:
    st.metric(
        "Valor por cliente",
        f"${seleccion['Valor por cliente']:,.2f}"
    )

with col3:
    st.metric(
        "Beneficio total estimado",
        f"${seleccion['Valor total']:,.2f}"
    )

st.success(
    f"El porcentaje más rentable entre los evaluados es "
    f"{int(mejor['Porcentaje'])}%, con un beneficio total "
    f"estimado de ${mejor['Valor total']:,.2f}."
)


#st.info("Reemplacen cada TODO con su propio código. Revisen el notebook del laboratorio para la lógica exacta de cada pieza.")
