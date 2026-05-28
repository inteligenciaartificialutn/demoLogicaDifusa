import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lógica difusa - Sistema difuso tradicional", layout="centered")
st.title("🧠 Lógica difusa - Sistema difuso tradicional")

st.markdown("""
Este ejemplo muestra como se podría calcular la contextura de una persona usando Lógica difusa con un sistema de reglas. En este ejemplo, se demuestra que las funciones de pertenencia pueden no ser lineales, en este caso usamos una función sigmoidal.
""")

# --- Definición de variables ---
altura = ctrl.Antecedent(np.arange(1.3, 2.0, 0.01), 'altura')
muñeca = ctrl.Antecedent(np.arange(10, 20, 0.1), 'muñeca')
contextura = ctrl.Consequent(np.arange(0, 10, 0.1), 'contextura')

# --- Funciones de pertenencia con solapamiento mayor ---
altura['baja']  = fuzz.sigmf(altura.universe, 1.55, -10)
altura['media'] = fuzz.gaussmf(altura.universe, 1.65, 0.12)
altura['alta']  = fuzz.sigmf(altura.universe, 1.75, 10)

muñeca['fina']   = fuzz.sigmf(muñeca.universe, 13.0, -5)
muñeca['media']  = fuzz.gaussmf(muñeca.universe, 15.0, 1.5)
muñeca['gruesa'] = fuzz.sigmf(muñeca.universe, 17.0, 5)

# Salida
contextura['pequeña'] = fuzz.trimf(contextura.universe, [0, 0, 5])
contextura['mediana']  = fuzz.trimf(contextura.universe, [2, 5, 8])
contextura['grande']   = fuzz.trimf(contextura.universe, [5, 10, 10])

# --- Reglas ---
rule1 = ctrl.Rule(muñeca['fina'], contextura['pequeña'])
rule2 = ctrl.Rule((altura['alta'] | altura['media']) & muñeca['gruesa'], contextura['mediana'])
rule3 = ctrl.Rule(altura['alta'] & ~muñeca['fina'], contextura['grande'])
rule4 = ctrl.Rule(muñeca['gruesa'], contextura['mediana'])  # catch-all

sistema_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
sistema = ctrl.ControlSystemSimulation(sistema_ctrl)

# --- Interfaz ---
altura_val = st.slider("Altura (m)", 1.3, 2.0, 1.6, 0.01)
muñeca_val = st.slider("Diámetro de muñeca (cm)", 10.0, 20.0, 14.0, 0.1)

sistema.input['altura'] = altura_val
sistema.input['muñeca'] = muñeca_val
sistema.compute()
contextura_val = sistema.output['contextura']

st.write(f"**Resultado difuso:** {contextura_val:.2f}")

# --- Graficar funciones de pertenencia con solapamiento ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Altura
for label in altura.terms:
    axes[0].plot(altura.universe, altura[label].mf, label=label)
axes[0].set_title('Funciones de pertenencia - Altura (solapamiento amplio)')
axes[0].legend()

# Muñeca
for label in muñeca.terms:
    axes[1].plot(muñeca.universe, muñeca[label].mf, label=label)
axes[1].set_title('Funciones de pertenencia - Muñeca (solapamiento amplio)')
axes[1].legend()

st.pyplot(fig)

# --- Visualizar salida difusa con barras ---
fig2, ax2 = plt.subplots(figsize=(6,3))
pequeña_val = fuzz.interp_membership(contextura.universe, contextura['pequeña'].mf, contextura_val)
mediana_val = fuzz.interp_membership(contextura.universe, contextura['mediana'].mf, contextura_val)
grande_val  = fuzz.interp_membership(contextura.universe, contextura['grande'].mf, contextura_val)

ax2.bar(['Pequeña', 'Mediana', 'Grande'],
        [pequeña_val, mediana_val, grande_val],
        color=['skyblue','orange','green'])
ax2.set_ylabel('Grado de pertenencia')
ax2.set_title('Salida difusa – Intensidad por categoría')
st.pyplot(fig2)