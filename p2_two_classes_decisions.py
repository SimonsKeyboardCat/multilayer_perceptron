import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    # Layer initialization
   def __init__(self, n_inputs, n_neurons,
               weight_regularizer_l1=0, weight_regularizer_l2=0,
               bias_regularizer_l1=0, bias_regularizer_l2=0):
      self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
      self.biases = np.zeros((1, n_neurons))
      # Set regularization strength
      self.weight_regularizer_l1 = weight_regularizer_l1
      self.weight_regularizer_l2 = weight_regularizer_l2
      self.bias_regularizer_l1 = bias_regularizer_l1
      self.bias_regularizer_l2 = bias_regularizer_l2
      
   def forward(self, inputs):
      self.inputs = inputs
      self.output = np.dot(inputs, self.weights) + self.biases

   def backward(self, dvalues):
      # Gradients on parameters
      self.dweights = np.dot(self.inputs.T, dvalues)
      self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
      # Gradients on regularization
      # L1 on weights
      if self.weight_regularizer_l1 > 0:
         dL1 = np.ones_like(self.weights)
         dL1[self.weights < 0] = -1
         self.dweights += self.weight_regularizer_l1 * dL1
      # L2 on weights
      if self.weight_regularizer_l2 > 0:
         self.dweights += 2 * self.weight_regularizer_l2 * \
                           self.weights
      # L1 on biases
      if self.bias_regularizer_l1 > 0:
         dL1 = np.ones_like(self.biases)
         dL1[self.biases < 0] = -1
         self.dbiases += self.bias_regularizer_l1 * dL1
      # L2 on biases
      if self.bias_regularizer_l2 > 0:
         self.dbiases += 2 * self.bias_regularizer_l2 * \
                           self.biases
      # Gradient on values
      self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
                                            np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                            single_dvalues)


class Loss:
   # Regularization loss calculation
   def regularization_loss(self, layer):
      # 0 by default
      regularization_loss = 0
      # L1 regularization - weights
      # calculate only when factor greater than 0
      if layer.weight_regularizer_l1 > 0:
         regularization_loss += layer.weight_regularizer_l1 * \
                                 np.sum(np.abs(layer.weights))
      # L2 regularization - weights
      if layer.weight_regularizer_l2 > 0:
         regularization_loss += layer.weight_regularizer_l2 * \
                                 np.sum(layer.weights * \
                                 layer.weights)

      # L1 regularization - biases
      # calculate only when factor greater than 0
      if layer.bias_regularizer_l1 > 0:
         regularization_loss += layer.bias_regularizer_l1 * \
                                 np.sum(np.abs(layer.biases))
      # L2 regularization - biases
      if layer.bias_regularizer_l2 > 0:
         regularization_loss += layer.bias_regularizer_l2 * \
                                 np.sum(layer.biases * \
                                 layer.biases)
      return regularization_loss

   def calculate(self, output, y):
      sample_losses = self.forward(output, y)
      data_loss = np.mean(sample_losses)
      return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
    # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Update parameters
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):    
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * \
                                layer.dweights
            bias_updates = -self.current_learning_rate * \
                                layer.dbiases
            
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
   # Initialize optimizer - set settings
   def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
   beta_1=0.9, beta_2=0.999):
      self.learning_rate = learning_rate
      self.current_learning_rate = learning_rate
      self.decay = decay
      self.iterations = 0
      self.epsilon = epsilon
      self.beta_1 = beta_1
      self.beta_2 = beta_2
      
   # Call once before any parameter updates
   def pre_update_params(self):
      if self.decay:
         self.current_learning_rate = self.learning_rate * \
                                       (1. / (1. + self.decay * self.iterations))
   # Update parameters
   def update_params(self, layer):
      # If layer does not contain cache arrays,
      # create them filled with zeros
      if not hasattr(layer, 'weight_cache'):
         layer.weight_momentums = np.zeros_like(layer.weights)
         layer.weight_cache = np.zeros_like(layer.weights)
         layer.bias_momentums = np.zeros_like(layer.biases)
         layer.bias_cache = np.zeros_like(layer.biases)

      # Update momentum with current gradients
      layer.weight_momentums = self.beta_1 * \
                              layer.weight_momentums + \
                              (1 - self.beta_1) * layer.dweights
      layer.bias_momentums = self.beta_1 * \
                              layer.bias_momentums + \
                              (1 - self.beta_1) * layer.dbiases
      
      # Get corrected momentum
      # self.iteration is 0 at first pass
      # and we need to start with 1 here
      weight_momentums_corrected = layer.weight_momentums / \
                                    (1 - self.beta_1 ** (self.iterations + 1))
      bias_momentums_corrected = layer.bias_momentums / \
                                    (1 - self.beta_1 ** (self.iterations + 1))
      # Update cache with squared current gradients
      layer.weight_cache = self.beta_2 * layer.weight_cache + \
                                    (1 - self.beta_2) * layer.dweights**2
      layer.bias_cache = self.beta_2 * layer.bias_cache + \
                                    (1 - self.beta_2) * layer.dbiases**2

      # Get corrected cache
      weight_cache_corrected = layer.weight_cache / \
                              (1 - self.beta_2 ** (self.iterations + 1))
      bias_cache_corrected = layer.bias_cache / \
                              (1 - self.beta_2 ** (self.iterations + 1))
      # Vanilla SGD parameter update + normalization
      # with square rooted cache
      layer.weights += -self.current_learning_rate * \
                        weight_momentums_corrected / \
                        (np.sqrt(weight_cache_corrected) +
                        self.epsilon)
      layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) +
                        self.epsilon)

   # Call once after any parameter updates
   def post_update_params(self):
      self.iterations += 1

# Dropout
class Layer_Dropout:
    # Init
    def __init__(self, rate):
    # Store rate, we invert it as for example for dropout
    # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs):
    # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
        size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
    # Gradient on values
        self.dinputs = dvalues * self.binary_mask

# Sigmoid activation
class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

class Loss_BinaryCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
            (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Create dataset
X, y = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Dropout layer
# dropout1 = Layer_Dropout(0.1)
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 1)
# Create Softmax classifier's combined loss and activation
activation2 = Activation_Sigmoid()
# Perform a forward pass of our training data through this layer
loss_function = Loss_BinaryCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)


for epoch in range(10001):
   dense1.forward(X)
   # Perform a forward pass through activation function
   # takes the output of first dense layer here
   activation1.forward(dense1.output)
   # Perform a forward pass through Dropout layer
   dense2.forward(activation1.output)
   # Perform a forward pass through second Dense layer
   # takes outputs of activation function of first layer as inputs
   activation2.forward(dense2.output)
   # Perform a forward pass through the activation/loss function
   # takes the output of second dense layer here and returns loss
   data_loss = loss_function.calculate(dense2.output, y)
   # Calculate regularization penalty
   regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)
   # Calculate overall loss
   loss = data_loss + regularization_loss
   # Calculate accuracy from output of activation2 and targets
   # calculate values along first axis
   predictions = (activation2.output > 0.5) * 1
   accuracy = np.mean(predictions==y)  

   if not epoch % 100:
      print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'data_loss: {data_loss:.3f}, ' +
            f'reg_loss: {regularization_loss:.3f}), ' +
            f'lr: {optimizer.current_learning_rate}')

   # Backward pass
   loss_function.backward(activation2.output, y)
   activation2.backward(loss_function.dinputs)
   dense2.backward(activation2.dinputs)
   activation1.backward(dense2.dinputs)
   dense1.backward(activation1.dinputs)

   # Update weights and biases
   optimizer.pre_update_params()
   optimizer.update_params(dense1)
   optimizer.update_params(dense2)
   optimizer.post_update_params()


# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)
# Perform a forward pass of our testing data through this layer
y_test = y_test.reshape(-1, 1)
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y)  

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')