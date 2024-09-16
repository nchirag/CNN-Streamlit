import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import os

# MNIST Dataset Parameters
num_classes = 10
num_features = 784  # Flattened 28x28 images

# Network Parameters
n_hidden_1 = 128
n_hidden_2 = 256

# Define the path to the MNIST dataset
mnist_path = 'mnist.npz'

# Define path for saving and loading the model
model_save_path = 'mnist_model.h5'

# Load MNIST Data with Error Handling
try:
    with np.load(mnist_path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
except Exception as e:
    st.error(f"An error occurred while loading the MNIST dataset: {e}")
    st.stop()

# Ensure x_train and x_test are correctly reshaped and normalized
x_train = np.array(x_train, np.float32).reshape([-1, num_features]) / 255.
x_test = np.array(x_test, np.float32).reshape([-1, num_features]) / 255.

# Update train_data with new batch_size in training settings
def create_train_data(batch_size):
    return tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(batch_size).prefetch(1)

# Define the Neural Network Model as a Keras Subclassed Model
class NeuralNet(tf.keras.Model):
    def __init__(self, activation):
        super(NeuralNet, self).__init__()
        self.activation = activation
        self.fc1 = tf.keras.layers.Dense(n_hidden_1, activation=activation)
        self.fc2 = tf.keras.layers.Dense(n_hidden_2, activation=activation)
        self.out = tf.keras.layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

    def get_config(self):
        config = super(NeuralNet, self).get_config()
        config.update({
            'activation': self.activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        activation = config.pop('activation', None)
        return cls(activation=activation)

# Loss Function
def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

# Accuracy Metric
def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Optimization Process
def run_optimization(x, y, neural_net, optimizer):
    with tf.GradientTape() as g:
        pred = neural_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)
    trainable_variables = neural_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# Save Model Function
def save_model(neural_net):
    neural_net.save(model_save_path, save_format='h5')

# Load Model Function
def load_model():
    try:
        return tf.keras.models.load_model(model_save_path, custom_objects={'NeuralNet': NeuralNet})
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# Streamlit App
st.title('MNIST Digit Classifier')

# Activation Functions Mapping
activation_functions = {
    'relu': 'relu',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
    'None': None
}

# Menu
menu = st.sidebar.radio("Menu", ["Training", "Prediction"])

if menu == "Training":
    st.sidebar.header('Training Settings')

    # User Inputs
    learning_rate = st.sidebar.slider('Learning Rate', 0.001, 1.0, 0.1)
    training_steps = st.sidebar.slider('Training Steps', 100, 1000, 500)
    batch_size = st.sidebar.slider('Batch Size', 32, 512, 256)
    activation_choice = st.sidebar.selectbox('Activation Function', ['relu', 'sigmoid', 'tanh', 'None'])

    activation = activation_functions.get(activation_choice, None)
    train_data = create_train_data(batch_size)

    neural_net = NeuralNet(activation)
    optimizer = tf.optimizers.SGD(learning_rate)

    if st.sidebar.button('Start Training'):
        step = 0
        for batch_x, batch_y in train_data:
            run_optimization(batch_x, batch_y, neural_net, optimizer)
            step += 1
            if step % 100 == 0 or step == training_steps:
                pred = neural_net(batch_x, is_training=True)
                loss = cross_entropy_loss(pred, batch_y)
                acc = accuracy(pred, batch_y)
                st.write(f"Step: {step}, Loss: {loss.numpy()}, Accuracy: {acc.numpy()}")
            if step >= training_steps:
                break

        # Evaluate on test data
        pred = neural_net(x_test, is_training=False)
        test_acc = accuracy(pred, y_test).numpy()
        st.write(f"Test Accuracy: {test_acc}")
        st.write("Training finished!")
        
        # Save the trained model
        save_model(neural_net)
        st.write("Model saved to disk!")

elif menu == "Prediction":
    st.header('Upload an Image for Prediction')
    uploaded_image = st.file_uploader("Choose an image...", type="png")

    if uploaded_image:
        image = Image.open(uploaded_image).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image).reshape([-1, num_features]) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Load the trained model
        if os.path.exists(model_save_path):
            neural_net = load_model()
        else:
            st.error("No trained model found. Please train the model first.")
            st.stop()
        
        prediction = neural_net(image_array, is_training=False)
        predicted_digit = np.argmax(prediction.numpy())

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f'Prediction: {predicted_digit}')
