import nnfs
import numpy as np
from nnfs.datasets import sine_data, spiral_data
from p2_functions import *


# Model class
class Model:
   def __init__(self):
      # Create a list of network objects
      self.layers = []
      
   # Add objects to the model
   def add(self, layer):
      self.layers.append(layer)

   # Set loss and optimizer
   def set(self, *, loss, optimizer):
      self.loss = loss
      self.optimizer = optimizer
   
   # Finalize the model
   def finalize(self):
      # Create and set the input layer
      self.input_layer = Layer_Input()
      # Count all the objects
      layer_count = len(self.layers)
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
         
         if hasattr(self.layers[i], 'weights'):
            self.trainable_layers.append(self.layers[i])

      //TODO: Tu skończyłem
         # The last layer - the next object is the loss
         # Also let's save aside the reference to the last object
         # whose output is the model's output
         else:
            self.layers[i].prev = self.layers[i-1]
            self.layers[i].next = self.loss
            self.output_layer_activation = self.layers[i]

   # Train the model
   def train(self, X, y, *, epochs=1, print_every=1):
      # Main training loop
      for epoch in range(1, epochs+1):
         # Perform the forward pass
         output = self.forward(X)
         # Temporary
         print(output)
         exit()

   # Performs forward pass
   def forward(self, X):
      # Call forward method on the input layer
      # this will set the output property that
      # the first layer in "prev" object is expecting
      self.input_layer.forward(X)
      # Call forward method of every object in a chain
      # Pass output of the previous object as a parameter
      for layer in self.layers:
         layer.forward(layer.prev.output)
         # "layer" is now the last object from the list,
      # return its output
      return layer.output


# Create dataset
X, y = sine_data()

# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# Set loss and optimizer objects
model.set(
   loss=Loss_MeanSquaredError(),
   optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
)

model.finalize()

model.train(X, y, epochs=10000, print_every=100)

print(model.layers)

