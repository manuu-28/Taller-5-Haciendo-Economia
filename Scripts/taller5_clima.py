import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ARCHIVO_TEMP = os.path.join("RawData", "datos_taller_5.csv")
ARCHIVO_CO2 = os.path.join("RawData", "co2_data.txt")
CARPETA_SALIDA = "Resultados"
MES = "Jun"          
MES_NUM = 6      

os.makedirs(os.path.join(CARPETA_SALIDA, "Figuras"), exist_ok=True)
plt.rcParams["figure.dpi"] = 110


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_SALIDA, "Figuras", nombre)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.show()

def linea_con_cero(serie, titulo, ylabel, color="steelblue"):
    fig, ax = plt.subplots()
    ax.axhline(0, color="darkorange", linewidth=1)
    serie.plot(ax=ax, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Año")
    ax.set_ylabel(ylabel)
    return fig, ax

def tabla_frecuencias(df, año_ini, año_fin, meses=("Jun", "Jul", "Aug")):
    datos = df.loc[(df.index >= año_ini) & (df.index <= año_fin), list(meses)]
    bins = np.arange(-0.30, 1.10, 0.05)
    return pd.cut(datos.stack(), bins=bins, right=False).value_counts().sort_index()

# Temperatura (NASA GISS) 
temp = pd.read_csv(ARCHIVO_TEMP, skiprows=1, na_values="***")
assert "Year" in temp.columns, (
    "No se encontró la columna 'Year'. Abre el CSV y revisa cuántas filas "
    "de encabezado hay que saltar con skiprows."
)
temp = temp.set_index("Year")
print("Temperatura cargada:", temp.shape, "\n", temp.head(), "\n")

#  CO2 (NOAA Mauna Loa) 

co2 = pd.read_csv(
    ARCHIVO_CO2,
    comment="#",
    sep=r"\s+",
    names=["year", "month", "decimal_date", "average",
           "deseasonalized", "ndays", "sdev", "unc"],
    na_values=-99.99,
)
assert co2.shape[1] == 8, (
    f"El archivo trajo {co2.shape[1]} columnas, se esperaban 8. Mándame una "
    "captura de las primeras filas del .txt para ajustar los nombres."
)
print("CO2 cargado:", co2.shape, "\n", co2.head(), "\n")


# PARTE 1.1 - Anomalías de temperatura

# --- 1.1.2 / 1.1.3(i): línea de un mes 
fig, ax = linea_con_cero(
    temp[MES],
    f"Anomalía de temperatura en {MES} - Hemisferio Norte",
    "Anomalía de temperatura (°C)",
)
guardar(fig, f"1_1_2_linea_{MES}.png")

# --- 1.1.3(ii): línea por estación 
fig, ax = plt.subplots()
ax.axhline(0, color="darkorange", linewidth=1)
for est in ["DJF", "MAM", "JJA", "SON"]:
    temp[est].plot(ax=ax, label=est)
ax.set_title("Anomalía de temperatura por estación - Hemisferio Norte")
ax.set_xlabel("Año")
ax.set_ylabel("Anomalía de temperatura (°C)")
ax.legend(title="Estación")
guardar(fig, "1_1_3ii_estaciones.png")

# --- 1.1.3(iii): línea anual 
fig, ax = linea_con_cero(
    temp["J-D"],
    "Anomalía de temperatura anual - Hemisferio Norte",
    "Anomalía de temperatura anual (°C)",
    color="firebrick",
)
guardar(fig, "1_1_3iii_anual.png")


# PARTE 1.2 - Variación de la temperatura en el tiempo

# --- 1.2.1: tablas de frecuencia 
frec_51_80 = tabla_frecuencias(temp, 1951, 1980)
frec_81_10 = tabla_frecuencias(temp, 1981, 2010)
print("Frecuencias 1951-1980:\n", frec_51_80)
print("Frecuencias 1981-2010:\n", frec_81_10)

# --- 1.2.2: histogramas comparados 
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)
frec_51_80.plot(kind="bar", ax=axes[0], color="steelblue", title="1951-1980")
frec_81_10.plot(kind="bar", ax=axes[1], color="firebrick", title="1981-2010")
for a in axes:
    a.set_xlabel("Rango de anomalía (°C)")
    a.tick_params(axis="x", rotation=90, labelsize=7)
axes[0].set_ylabel("Frecuencia")
fig.suptitle("Distribución de anomalías de temperatura (JJA)")
guardar(fig, "1_2_2_histogramas.png")
# Discusión: compara moda, dispersión y hacia dónde se corre la masa de
# la distribución entre los dos períodos.

# --- 1.2.3: deciles 3 y 7 (1951-1980) 
valores_51_80 = temp.loc[1951:1980, "Jan":"Dec"].stack()
q30, q70 = np.quantile(valores_51_80, [0.3, 0.7])
print(f"Umbral 'frío' (decil 3): {q30:.3f} °C | Umbral 'caliente' (decil 7): {q70:.3f} °C")

# --- 1.2.4: % de meses "calientes" en 1981-2010 
valores_81_10 = temp.loc[1981:2010, "Jan":"Dec"].stack()
pct_caliente = (valores_81_10 > q70).mean() * 100
print(f"% de meses 'calientes' en 1981-2010 (umbral fijado en 1951-1980): {pct_caliente:.2f}%")

# --- 1.2.5: media y varianza por estación y período 
largo = (
    temp.loc[:, "DJF":"SON"]
    .stack()
    .reset_index()
    .rename(columns={"level_1": "Estacion", 0: "Valor"})
)
largo["Periodo"] = pd.cut(
    largo["Year"], bins=[1921, 1950, 1980, 2010],
    labels=["1921-1950", "1951-1980", "1981-2010"],
)
media_var = largo.groupby(["Estacion", "Periodo"], observed=True)["Valor"].agg(
    media="mean", varianza="var"
)
print("Media y varianza por estación y período:\n", media_var)
media_var.to_csv(os.path.join(CARPETA_SALIDA, "1_2_5_media_varianza.csv"))


# PARTE 1.3 - CO2 y su relación con la temperatura

# --- 1.3.3: línea de CO2 desde 1960 
co2_1960 = co2[co2["year"] >= 1960].copy()
co2_1960["fecha"] = pd.to_datetime(dict(year=co2_1960["year"], month=co2_1960["month"], day=1))

fig, ax = plt.subplots()
ax.plot(co2_1960["fecha"], co2_1960["average"], label="Promedio mensual", linewidth=1)
ax.plot(co2_1960["fecha"], co2_1960["deseasonalized"], label="Tendencia (desestacionalizada)", linewidth=1.5)
ax.set_title("Niveles de CO2 - Observatorio de Mauna Loa (1960-presente)")
ax.set_xlabel("Año")
ax.set_ylabel("CO2 (ppm)")
ax.legend()
guardar(fig, "1_3_3_co2_tiempo.png")

# --- 1.3.4: dispersión + correlación de Pearson 
co2_mes = co2.loc[co2["month"] == MES_NUM, ["year", "deseasonalized"]].rename(columns={"year": "Year"})
temp_co2 = pd.merge(temp.reset_index()[["Year", MES]], co2_mes, on="Year", how="inner")
assert len(temp_co2) > 0, "El merge no produjo filas; revisa el rango de años en ambos archivos."

fig, ax = plt.subplots()
ax.scatter(temp_co2[MES], temp_co2["deseasonalized"], color="black", s=25)
ax.set_title("Anomalía de temperatura vs. CO2 (tendencia)")
ax.set_xlabel(f"Anomalía de temperatura en {MES} (°C)")
ax.set_ylabel("CO2 - tendencia (ppm)")
guardar(fig, "1_3_4_dispersión.png")

r, p_valor = stats.pearsonr(temp_co2[MES], temp_co2["deseasonalized"])
print(f"Correlación de Pearson ({MES} vs. CO2): r = {r:.3f}  (p = {p_valor:.2e}, n = {len(temp_co2)})")
# Discusión: interpreta magnitud y signo; recuerda que ambas series están
# fuertemente auto-correlacionadas en el tiempo, lo que infla r
# mecánicamente, y que correlación no implica causalidad (ver 1.3.6).


print("\nListo. Gráficos en Output/Figuras/, tabla en "
      "Output/1_2_5_media_varianza.csv")
