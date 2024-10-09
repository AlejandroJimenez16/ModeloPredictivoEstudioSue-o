import tensorflow as tf
import numpy as np

#Valores
horas_estudiadas = np.array([0, 1, 2, 3, 4, 5], dtype=float)
horas_dormidas = np.array([8, 7, 6, 5, 4, 3], dtype=float)
notas_examen = np.array([2, 4, 5, 6, 7, 9], dtype=float)
niveles_energia = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.2], dtype=float)

#Capas
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, input_shape=[1], name='horas_estudiadas'),
    tf.keras.layers.Dense(units=5),
    tf.keras.layers.Dense(units=10, input_shape=[1], name='horas_dormidas'),
    tf.keras.layers.Dense(units=5),
    tf.keras.layers.Dense(units=1, name='notas_examen'),
    tf.keras.layers.Dense(units=1, name='niveles_energia')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#Entrenamiento
print("===========================")
print("Comenzando entrenamiento...")
print("===========================")

historial = modelo.fit([horas_estudiadas,horas_dormidas], [notas_examen, niveles_energia], epochs=1000, verbose=False)

print("=================")
print("Modelo entrenado!")
print("=================")

#Grafica
import matplotlib.pyplot as plt
plt.xlabel('# Epoca')
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()

# Prediccion
print("Hagamos una prediccion!")
nota, energia = modelo.predict([np.array([5, 3])])
print("==================================================")
print("Nota:", nota)
print("Energia:", energia)
print("==================================================")

#modelo.save("modeloPrediccionSuenio.keras")