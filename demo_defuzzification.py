import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lógica difusa - Defuzzificación", layout="centered")
st.title("🧠 Lógica difusa - Defuzzificación")

st.markdown("""
En este ejemplo simulamos el control de un calefactor mediante un Sistema difuso. Observamos dos cosas particulares:
1. Tenemos un procesamiento de los inputs previo al uso en el sistema de reglas. En los inputs recibimos la temperatura deseada y la actual y los procesamos para calcular un `error de temperatura` que es lo que se usa en las reglas.
2. Se agrega un módulo de defuzzificación con distintas funciones matemáticas que permite ver como se puede tomar la "ambigüedad" de las reglas y obtener un valor de output final.
""")

# ---------------- Variables ----------------
error = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'error')
delta = ctrl.Antecedent(np.arange(-5, 5, 0.1), 'delta')
potencia = ctrl.Consequent(np.arange(0, 100, 1), 'potencia')

# Funciones de membresía
error['frío'] = fuzz.zmf(error.universe, -5, 0)
error['ideal'] = fuzz.gaussmf(error.universe, 0, 3)
error['caliente'] = fuzz.smf(error.universe, 0, 5)

delta['bajando'] = fuzz.zmf(delta.universe, -2, 0)
delta['estable'] = fuzz.gaussmf(delta.universe, 0, 1)
delta['subiendo'] = fuzz.smf(delta.universe, 0, 2)

potencia['baja'] = fuzz.zmf(potencia.universe, 25, 50)
potencia['media'] = fuzz.gaussmf(potencia.universe, 50, 20)
potencia['alta'] = fuzz.smf(potencia.universe, 50, 75)

# Reglas
rule1 = ctrl.Rule(error['frío'] & delta['bajando'], potencia['alta'])
rule2 = ctrl.Rule(error['frío'] & delta['estable'], potencia['alta'])
rule3 = ctrl.Rule(error['ideal'] & delta['estable'], potencia['media'])
rule4 = ctrl.Rule(error['caliente'] | delta['subiendo'], potencia['baja'])

sistema_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
sistema_sim = ctrl.ControlSystemSimulation(sistema_ctrl)

# ---------------- Sliders ----------------
temp_deseada = st.slider("Temperatura deseada (°C)", 15, 30, 22)
temp_actual = st.slider("Temperatura actual (°C)", 10, 35, 18)
delta_val = st.slider("Cambio de temperatura (°C/min)", -5.0, 5.0, 0.0, 0.1)
error_val = temp_deseada - temp_actual

# ---------------- Simulación ----------------
sistema_sim.input['error'] = error_val
sistema_sim.input['delta'] = delta_val
sistema_sim.compute()
potencia_val = sistema_sim.output['potencia']

# ---------------- Selector de método ----------------
st.subheader("Defuzzificación")
metodos = ['centroid', 'bisector', 'mom', 'som', 'lom']
metodo_sel = st.selectbox("Seleccionar método de defuzzificación", metodos, index=0)

explicacion = {
    'centroid': "Centroide: centro de gravedad del área combinada.",
    'bisector': "Bisector: divide el área total en dos partes iguales.",
    'mom': "Mean of Maximum: promedio de los máximos.",
    'som': "Smallest of Maximum: menor de los máximos.",
    'lom': "Largest of Maximum: mayor de los máximos."
}
st.markdown(f"**Método seleccionado:** {explicacion[metodo_sel]}")

# ---------------- Defuzzificación ----------------
potencia_defuz = fuzz.defuzz(
    potencia.universe,
    potencia['baja'].mf * fuzz.interp_membership(potencia.universe, potencia['baja'].mf, potencia_val) +
    potencia['media'].mf * fuzz.interp_membership(potencia.universe, potencia['media'].mf, potencia_val) +
    potencia['alta'].mf * fuzz.interp_membership(potencia.universe, potencia['alta'].mf, potencia_val),
    metodo_sel
)
st.metric("Potencia final (defuzzificada)", f"{potencia_defuz:.1f} %")

# ---------------- Función para graficar input ----------------
def plot_input_membership(var, val, title):
    fig, ax = plt.subplots(figsize=(4,3))
    for label in var.terms:
        ax.plot(var.universe, var[label].mf, label=label)
    ax.axvline(val, color='red', linestyle='--', label='Valor actual')
    ax.set_title(title)
    ax.set_ylabel('Grado')
    ax.set_xlabel(var.label)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

# ---------------- Fila 1: Inputs y salida ----------------
st.subheader("Entradas y salida difusa")
col1, col2, col3 = st.columns(3)

with col1:
    st.pyplot(plot_input_membership(error, error_val, "Error térmico"))

with col2:
    st.pyplot(plot_input_membership(delta, delta_val, "Cambio de temperatura"))

with col3:
    fig_out, ax_out = plt.subplots(figsize=(9,6))
    ax_out.plot(potencia.universe, potencia['baja'].mf, label='Baja', color='skyblue')
    ax_out.plot(potencia.universe, potencia['media'].mf, label='Media', color='orange')
    ax_out.plot(potencia.universe, potencia['alta'].mf, label='Alta', color='green')
    ax_out.axvline(potencia_defuz, color='red', linestyle='--', label='Valor defuzzificado')
    ax_out.set_xlabel('Potencia (%)', fontsize=18)
    ax_out.set_ylabel('Grado de pertenencia', fontsize=18)
    ax_out.set_title('Salida difusa con valor final', fontsize=24)
    ax_out.legend(fontsize=18)  # tamaño de letra más grande
    st.pyplot(fig_out)

# ---------------- Fila 2: Gráfico de barras ----------------
st.subheader("Grado de pertenencia del output")
baja_val  = fuzz.interp_membership(potencia.universe, potencia['baja'].mf, potencia_defuz)
media_val = fuzz.interp_membership(potencia.universe, potencia['media'].mf, potencia_defuz)
alta_val  = fuzz.interp_membership(potencia.universe, potencia['alta'].mf, potencia_defuz)

fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.bar(['Baja', 'Media', 'Alta'], [baja_val, media_val, alta_val],
        color=['skyblue','orange','green'])
ax2.set_ylabel('Grado de pertenencia')
ax2.set_title('Intensidad por categoría (actualizada)')
st.pyplot(fig2)