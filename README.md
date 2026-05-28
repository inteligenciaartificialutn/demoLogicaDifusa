# demoLogicaDifusa
Este repositorio contiene varias **demos interactivas de Lógica Difusa** desarrolladas en Python con **Streamlit** y **scikit-fuzzy**, donde puedes explorar:

- Sistemas difusos clásicos tipo Mamdani
- Funciones de membresía triangulares y sigmoidales
- Diferentes métodos de defuzzificación (`centroid`, `bisector`, `mom`, `som`, `lom`)
- Ejemplos visuales con sliders para ajustar entradas y ver resultados en tiempo real
- Salida visual mediante gráficos de funciones de membresía y barras de pertenencia

---

## Instalación
1. **Crear un entorno virtual (opcional, recomendado)**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

2. **Instalar dependencias**

```bash
pip install streamlit==1.36.0 scikit-fuzzy==0.4.2 matplotlib numpy tensorflow
```

---

## Uso

Para correr una demo, ejecuta:

```bash
streamlit run nombre_demo.py
```

Por ejemplo, para la demo de calefactor:

```bash
streamlit run demo_calefactor.py
```

Luego se abrirá una **interfaz web local** en tu navegador, donde podrás interactuar con sliders, ver gráficos y probar distintos métodos de defuzzificación.

---

## Estructura del repositorio

- `demo_contextura.py` – Estimación de contextura física usando altura y diámetro de muñeca.
- `demo_contextura_no_lineal.py` – Estimación de contextura física usando altura y diámetro de muñeca usando funciones de pertenencia sigmoidales.
- `demo_calefactor.py` – Control de calefactor con lógica difusa.
- `demo_defuzzification.py` – Control de calefactor con lógica difusa, agregando defuzzificación del output.
- `demo_neuro_difuso.py` – Ejemplo de sistema neuro-difuso (con TensorFlow si está instalado).

---

## Recomendaciones

- Usar **Python 3.11** o superior.
- Instalar las versiones exactas de las librerías para asegurar que las demos funcionen correctamente.
- Abrir cada demo en una ventana de navegador independiente para comparar resultados fácilmente.

---

¡Diviértete explorando la lógica difusa!


