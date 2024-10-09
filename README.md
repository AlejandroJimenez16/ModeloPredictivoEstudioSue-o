# Predicción de Notas y Energía

Este programa utiliza TensorFlow y Keras para predecir la nota de un examen y el nivel de energía de un estudiante, basándose en las horas que ha estudiado y las horas que ha dormido. El modelo se entrena con un conjunto de datos simple y realiza predicciones al final del entrenamiento.

## Descripción

El modelo se basa en una red neuronal que toma como entradas:

- **Horas estudiadas**: Cantidad de horas que el estudiante ha dedicado a estudiar.
- **Horas dormidas**: Cantidad de horas que el estudiante ha dormido.

A partir de estas entradas, el modelo predice:

- **Nota del examen**: La calificación esperada del estudiante en un examen.
- **Nivel de energía**: Un valor que representa la energía del estudiante.

## Tecnologías utilizadas

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Instalación

Para ejecutar este programa, asegúrate de tener instalado Python 3.x. Luego, puedes instalar las dependencias necesarias utilizando `pip`:

```bash
  pip install tensorflow numpy matplotlib
```

## Uso

1. **Ejecuta el código:** Simplemente ejecuta el archivo Python que contiene el código.
2. **Entrenamiento:** El modelo se entrenará durante 1000 épocas. Durante este proceso, podrás ver cómo se reduce la magnitud de pérdida.
3. **Predicción:** Una vez que el modelo ha sido entrenado, puedes hacer una predicción ingresando el número de horas estudiadas y horas dormidas. En el código proporcionado, se realiza una predicción para 5 horas de estudio y 3 horas de sueño:
   
   ```python
   nota, energia = modelo.predict([np.array([5, 3])])
   print("Nota:", nota)
   print("Energía:", energia)
   ```

## Resultados

El programa generará una gráfica que muestra la magnitud de pérdida a lo largo de las épocas. También imprimirá la nota y el nivel de energía estimados basados en las horas de estudio y sueño proporcionadas.

![image](https://github.com/user-attachments/assets/b129dd78-23e3-43a9-9149-c7980b73e1f2)

## Ejemplo de salida

```plainText
===========================
Comenzando entrenamiento...
===========================
=================
Modelo entrenado!
=================
Hagamos una predicción!
==================================================
Nota: [8.382353]
Energía: [5.923173]
==================================================
```

## Guardar el modelo

Si deseas guardar el modelo entrenado para uso futuro, puedes descomentar la línea al final del código:

```python
modelo.save("modeloPrediccionSuenio.keras")
```


