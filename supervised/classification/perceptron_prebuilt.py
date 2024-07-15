import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NeuralNetworkPrebuilt:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_dim=input_size, activation='sigmoid'))
        self.model.add(Dense(output_size, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X, y, epochs=1000):
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)
