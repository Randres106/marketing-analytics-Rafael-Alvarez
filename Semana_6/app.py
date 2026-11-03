
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from kmodes.kprototypes import KPrototypes
from pathlib import Path

CARPETA = Path(__file__).resolve().parent

st.title("Segmentación de clientes")
st.write("Laboratorio de Marketing Analytics")

opcion = st.sidebar.selectbox(
    "Seleccionar opción",
    ["Ver segmentos", "Comparar técnicas"]
)

if opcion == "Ver segmentos":
    st.header("Visor de segmentos")
    if True:
        df = pd.read_csv(CARPETA / "segmentos_notebook.csv")
        df.columns = [x.split("#")[-1] for x in df.columns]
        columna = st.selectbox(
            "Seleccionar grupo",
            [x for x in df.columns if x == "Cluster" or x.startswith("seg_")]
        )
        tabla = df.groupby(columna).agg(
            clientes=("gasto_total", "count"),
            gasto_promedio=("gasto_total", "mean"),
            gasto_total=("gasto_total", "sum")
        ).reset_index()

        tabla["% clientes"] = tabla["clientes"] / len(df) * 100
        tabla["% gasto"] = tabla["gasto_total"] / df["gasto_total"].sum() * 100

        st.subheader("Resultados")
        st.dataframe(tabla.round(2))

        fig = px.bar(
            tabla,
            x=columna,
            y="clientes",
            title="Clientes por segmento"
        )

        st.plotly_chart(fig)

        fig2 = px.bar(
            tabla,
            x=columna,
            y="% gasto",
            title="Porcentaje del gasto"
        )

        st.plotly_chart(fig2)

else:
    st.header("Comparación de técnicas")
    if True:
        df = pd.read_csv(CARPETA / "clientes_retail.csv")
        df.columns = [x.split("#")[-1] for x in df.columns]
        variables = [
            "ingreso_anual", "gasto_total",
            "compras_tienda", "compras_web",
            "compras_catalogo", "visitas_web_mes",
            "recencia_dias", "hijos_pequenos", "edad"
        ]
        x = df[variables]
        x = x.fillna(x.median())
        x = StandardScaler().fit_transform(x)
        k = st.slider("Número de grupos", 2, 8, 3)
        eps = st.slider("Epsilon DBSCAN", 0.1, 4.0, 1.5, 0.1)

        if st.button("Comparar"):
            resultados = {}
            resultados["K-Means"] = KMeans(
                n_clusters=k, random_state=0, n_init=10
            ).fit_predict(x)

            resultados["Jerárquico"] = AgglomerativeClustering(
                n_clusters=k, linkage="ward"
            ).fit_predict(x)

            resultados["DBSCAN"] = DBSCAN(
                eps=eps, min_samples=5
            ).fit_predict(x)

            resultados["GMM"] = GaussianMixture(
                n_components=k, random_state=0
            ).fit_predict(x)

            #K-prototypes usa números y categorías
            categorias = df[["educacion", "estado_civil"]].fillna("Desconocido")
            datos = np.column_stack((x.astype(object), categorias.astype(str).values))

            modelo = KPrototypes(
                n_clusters=k, init="Cao", random_state=0
            )
            resultados["K-Prototypes"] = modelo.fit_predict(
                datos, categorical=[9, 10]
            )
            resumen = []

            for nombre, grupos in resultados.items():
                grupos_validos = grupos[grupos != -1]
                cantidad = len(set(grupos_validos))
                ruido = sum(grupos == -1)

                if cantidad > 1 and cantidad < len(grupos_validos):
                    silueta = silhouette_score(
                        x[grupos != -1],
                        grupos_validos,
                        sample_size=min(1000, len(grupos_validos)),
                        random_state=0
                    )
                else:
                    silueta = np.nan

                tamanos = pd.Series(grupos_validos).value_counts()

                resumen.append([
                    nombre,
                    cantidad,
                    ruido,
                    str(tamanos.tolist()),
                    round(silueta, 3) if not np.isnan(silueta) else "-"
                ])

            st.subheader("Resultados")

            tabla = pd.DataFrame(
                resumen,
                columns=["Técnica", "Grupos", "Ruido", "Tamaños", "Silueta"]
            )

            st.dataframe(tabla)
            st.subheader("Coincidencia entre técnicas (ARI)")
            nombres = list(resultados.keys())
            acuerdo = pd.DataFrame(index=nombres, columns=nombres)

            for a in nombres:
                for b in nombres:
                    acuerdo.loc[a, b] = round(
                        adjusted_rand_score(resultados[a], resultados[b]), 2
                    )

            st.dataframe(acuerdo)
            st.subheader("Conclusión")
            st.write(
                "K-Means es una buena opción para marketing "
                "porque permite identificar tres grupos de clientes "
                "y comparar sus ingresos y gastos. "
                "También podemos utilizar las otras técnicas "
                "para revisar si los resultados son similares."
            )
