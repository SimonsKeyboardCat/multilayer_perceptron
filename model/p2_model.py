import nnfs
import numpy as np
from load_dataset import *
from nnfs.datasets import sine_data, spiral_data
from p2_functions import *


# Model class
class Model:
   def __init__(self):
      # Create a list of network objects
      self.layers = []
      self.softmax_classifier_output = None
      
   # Add objects to the model
   def add(self, layer):
      self.layers.append(layer)

   # Set loss and optimizer
   def set(self, *, loss, optimizer, accuracy):
      self.loss = loss
      self.optimizer = optimizer
      self.accuracy = accuracy
   
   # Finalize the model
   def finalize(self):
      # Create and set the input layer
      self.input_layer = Layer_Input()
      # Count all the objects
      layer_count = len(self.layers)

      # Initialize a list containing trainable layers:
      self.trainable_layers = []

      # Iterate the objects
      for i in range(layer_count):
         # If it's the first layer,
         # the previous layer object is the input layer
         if i == 0:
            self.layers[i].prev = self.input_layer
            self.layers[i].next = self.layers[i+1]
         # All layers except for the first and the last
         elif i < layer_count - 1:
            self.layers[i].prev = self.layers[i-1]
            self.layers[i].next = self.layers[i+1]
         # The last layer - the next object is the loss
         else: 
            self.layers[i].prev = self.layers[i-1]
            self.layers[i].next = self.loss
            self.output_layer_activation = self.layers[i]
         
         if hasattr(self.layers[i], 'weights'):
            self.trainable_layers.append(self.layers[i])

         # Update loss object with trainable layers
         self.loss.remember_trainable_layers(
         self.trainable_layers)

      # If output activation is Softmax and
      # loss function is Categorical Cross-Entropy
      # create an object of combined activation
      # and loss function containing
      # faster gradient calculation
      if isinstance(self.layers[-1], Activation_Softmax) and \
         isinstance(self.loss, Loss_CategoricalCrossentropy):
         # Create an object of combined activation
         # and loss functions
         self.softmax_classifier_output = \
            Activation_Softmax_Loss_CategoricalCrossentropy()


   # Train the model
   def train(self, X, y, *, epochs=1, print_every=1, validation_data=None, batch_size=None):
      
      self.accuracy.init(y)

      # Default value if batch size is not being set
      train_steps = 1

      # If there is validation data passed,
      # set default number of steps for validation as well
      if validation_data is not None:
         validation_steps = 1

      # For better readability
      X_val, y_val = validation_data

      # Calculate number of steps
      if batch_size is not None:
         train_steps = len(X) // batch_size
         # Dividing rounds down. If there are some remaining
         # data, but not a full batch, this won't include it
         # Add `1` to include this not full batch
         if train_steps * batch_size < len(X):
            train_steps += 1
         if validation_data is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but nor full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
               validation_steps += 1

      # Main training loop
      for epoch in range(1, epochs+1):
         # Print epoch number
         print(f'epoch: {epoch}')
         # Reset accumulated values in loss and accuracy objects
         self.loss.new_pass()
         self.accuracy.new_pass()
         # Iterate over steps
         for step in range(train_steps):
            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
               batch_X = X
               batch_y = y
            # Otherwise slice a batch
            else:
               batch_X = X[step*batch_size:(step+1)*batch_size]
               batch_y = y[step*batch_size:(step+1)*batch_size]
            # Perform the forward pass
            output = self.forward(batch_X, training=True)
            # Calculate loss
            data_loss, regularization_loss = \
                                       self.loss.calculate(output, batch_y, include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
            # Perform backward pass
            self.backward(output, batch_y)
            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
               self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            # Print a summary
            if not step % print_every or step == train_steps - 1:
               print(f'step: {step}, ' +
                     f'acc: {accuracy:.3f}, ' +
                     f'loss: {loss:.3f} (' +
                     f'data_loss: {data_loss:.3f}, ' +
                     f'reg_loss: {regularization_loss:.3f}), ' +
                     f'lr: {self.optimizer.current_learning_rate}')

         # Get and print epoch loss and accuracy
         epoch_data_loss, epoch_regularization_loss = \
            self.loss.calculate_accumulated(include_regularization=True)
         epoch_loss = epoch_data_loss + epoch_regularization_loss
         epoch_accuracy = self.accuracy.calculate_accumulated()

         print(f'training, ' +
               f'acc: {epoch_accuracy:.3f}, ' +
               f'loss: {epoch_loss:.3f} (' +
               f'data_loss: {epoch_data_loss:.3f}, ' +
               f'reg_loss: {epoch_regularization_loss:.3f}), ' +
               f'lr: {self.optimizer.current_learning_rate}')
         
         # If there is the validation data
         if validation_data is not None:
            # Reset accumulated values in loss
            # and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(validation_steps):
               # If batch size is not set -
               # train using one step and full dataset
               if batch_size is None:
                  batch_X = X_val
                  batch_y = y_val
               # Otherwise slice a batch
               else:
                  batch_X = X_val[
                  step*batch_size:(step+1)*batch_size
                  ]
                  batch_y = y_val[
                  step*batch_size:(step+1)*batch_size
                  ]
               # Perform the forward pass
               output = self.forward(batch_X, training=False)
               # Calculate the loss
               self.loss.calculate(output, batch_y)
               # Get predictions and calculate an accuracy
               predictions = self.output_layer_activation.predictions(
               output)
               self.accuracy.calculate(predictions, batch_y)
               # Get and print validation loss and accuracy
            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()

            # Print a summary
            print(f'validation, ' +
            f'acc: {validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f}')


   # Performs forward pass
   def forward(self, X, training):
      # Call forward method on the input layer
      # this will set the output property that
      # the first layer in "prev" object is expecting
      self.input_layer.forward(X, training)
      # Call forward method of every object in a chain
      # Pass output of the previous object as a parameter
      for layer in self.layers:
         layer.forward(layer.prev.output, training)
      # "layer" is now the last object from the list,
      # return its output
      return layer.output
   
   # Performs backward pass
   def backward(self, output, y):
      # If softmax classifier
      if self.softmax_classifier_output is not None:
         # First call backward method
         # on the combined activation/loss
         # this will set dinputs property
         self.softmax_classifier_output.backward(output, y)
         # Since we'll not call backward method of the last layer
         # which is Softmax activation
         # as we used combined activation/loss
         # object, let's set dinputs in this object
         self.layers[-1].dinputs = \
            self.softmax_classifier_output.dinputs
         # Call backward method going through
         # all the objects but last
         # in reversed order passing dinputs as a parameter
         for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)
         return
      # First call backward method on the loss
      # this will set dinputs property that the last
      # layer will try to access shortly
      self.loss.backward(output, y)
      # Call backward method going through all the objects
      # in reversed order passing dinputs as a parameter
      for layer in reversed(self.layers):
         layer.backward(layer.next.dinputs)


# Common accuracy class
class Accuracy:
   # Calculates an accuracy
   # given predictions and ground truth values
   def calculate(self, predictions, y):
      # Get comparison results
      comparisons = self.compare(predictions, y)
      # Calculate an accuracy
      accuracy = np.mean(comparisons)
      # Add accumulated sum of matching values and sample count
      self.accumulated_sum += np.sum(comparisons)
      self.accumulated_count += len(comparisons)

      # Return accuracy
      return accuracy

   # Calculates accumulated accuracy
   def calculate_accumulated(self):
      # Calculate an accuracy
      accuracy = self.accumulated_sum / self.accumulated_count
      # Return the data and regularization losses
      return accuracy
   
   # Reset variables for accumulated accuracy
   def new_pass(self):
      self.accumulated_sum = 0
      self.accumulated_count = 0

# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
   def __init__(self):
      # Create precision property
      self.precision = None
   # Calculates precision value
   # based on passed in ground truth
   def init(self, y, reinit=False):
      if self.precision is None or reinit:
         self.precision = np.std(y) / 250
   # Compares predictions to the ground truth values
   def compare(self, predictions, y):
      return np.absolute(predictions - y) < self.precision

# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
   # No initialization is needed
   def init(self, y):
      pass
   # Compares predictions to the ground truth values
   def compare(self, predictions, y):
      if len(y.shape) == 2:
         y = np.argmax(y, axis=1)
      return predictions == y

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]
# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
127.5) / 127.5

# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss and optimizer objects
model.set(
   loss=Loss_CategoricalCrossentropy(),
   optimizer=Optimizer_Adam(decay=1e-3),
   accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)


