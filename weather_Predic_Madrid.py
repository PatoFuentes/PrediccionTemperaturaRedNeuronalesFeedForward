import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('weather_features.csv')

# Filtrar datos para Madrid
madrid_data = data[data['city_name'] == 'Madrid']

# Seleccionar características y objetivo
X_madrid = madrid_data[['pressure', 'humidity', 'wind_speed', 'clouds_all']].values
y_madrid = madrid_data['temp'].values

# Escalar los datos
scaler_X = StandardScaler()
X_scaled_madrid = scaler_X.fit_transform(X_madrid)

scaler_y = StandardScaler()
y_scaled_madrid = scaler_y.fit_transform(y_madrid.reshape(-1, 1))

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled_madrid, y_scaled_madrid, test_size=0.2, random_state=42)

# Función para crear y compilar el modelo
def create_ffnn(layers, neurons_per_layer, activation='relu', learning_rate=0.001):
    model = Sequential()
    # Agregar capas ocultas
    model.add(Dense(neurons_per_layer[0], input_dim=X_train.shape[1], activation=activation))
    for neurons in neurons_per_layer[1:]:
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))  # Capa de salida
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Entrenar el modelo
def train_ffnn(model, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

# Configuración 1: Red simple (1 capa, 64 neuronas, ReLU, lr=0.001)
model1 = create_ffnn(layers=1, neurons_per_layer=[64], activation='relu', learning_rate=0.001)
history1 = train_ffnn(model1, epochs=50, batch_size=32)

# Configuración 2: Más capas y neuronas (3 capas, 128-64-32 neuronas, ReLU, lr=0.001)
model2 = create_ffnn(layers=3, neurons_per_layer=[128, 64, 32], activation='relu', learning_rate=0.001)
history2 = train_ffnn(model2, epochs=50, batch_size=32)

# Configuración 3: Función de activación Sigmoid (2 capas, 64-32 neuronas, Sigmoid, lr=0.001)
model3 = create_ffnn(layers=2, neurons_per_layer=[64, 32], activation='sigmoid', learning_rate=0.001)
history3 = train_ffnn(model3, epochs=50, batch_size=32)

# Configuración 4: Tasa de aprendizaje menor (2 capas, 64-32 neuronas, ReLU, lr=0.0001)
model4 = create_ffnn(layers=2, neurons_per_layer=[64, 32], activation='relu', learning_rate=0.0001)
history4 = train_ffnn(model4, epochs=50, batch_size=32)

# Configuración 5: Más épocas (2 capas, 64-32 neuronas, ReLU, lr=0.001, 100 épocas)
model5 = create_ffnn(layers=2, neurons_per_layer=[64, 32], activation='relu', learning_rate=0.001)
history5 = train_ffnn(model5, epochs=100, batch_size=32)

# Función para graficar la pérdida
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.show()

# Graficar cada configuración
plot_loss(history1, 'Configuración 1: Red Simple (1 capa, 64 neuronas, ReLU)')
plot_loss(history2, 'Configuración 2: Más Capas (128-64-32, ReLU)')
plot_loss(history3, 'Configuración 3: Sigmoid (64-32, Sigmoid)')
plot_loss(history4, 'Configuración 4: Tasa de Aprendizaje Menor (64-32, ReLU, lr=0.0001)')
plot_loss(history5, 'Configuración 5: Más Épocas (64-32, ReLU, 100 épocas)')
