import nnfs
import numpy as np
from nnfs.datasets import sine_data, spiral_data

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
      
   def forward(self, inputs, training):
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
   def forward(self, inputs, training):
      self.inputs = inputs
      self.output = np.maximum(0, inputs)

   def backward(self, dvalues):
      # Since we need to modify the original variable,
      # let's make a copy of the values first
      self.dinputs = dvalues.copy()
      # Zero gradient where input values were negative
      self.dinputs[self.inputs <= 0] = 0

   # Calculate predictions for outputs
   def predictions(self, outputs):
      return outputs


class Activation_Softmax:
   def forward(self, inputs, training):
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
            
   # Calculate predictions for outputs
   def predictions(self, outputs):
      return np.argmax(outputs, axis=1)


class Loss:
    # Regularization loss calculation
    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0

        for layer in self.trainable_layers:
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
   
    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    
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
    def forward(self, inputs, training):
    # Save input values
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
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
   def forward(self, inputs, training):
      # Save input and calculate/save output
      # of the sigmoid function
      self.inputs = inputs
      self.output = 1 / (1 + np.exp(-inputs))

   # Backward pass
   def backward(self, dvalues):
      # Derivative - calculates from output of the sigmoid function
      self.dinputs = dvalues * (1 - self.output) * self.output

   # Calculate predictions for outputs
   def predictions(self, outputs):
      return (outputs > 0.5) * 1


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

# Linear activation
class Activation_Linear:
   # Forward pass
   def forward(self, inputs, training):
      # Just remember values
      self.inputs = inputs
      self.output = inputs
   # Backward pass
   def backward(self, dvalues):
      # derivative is 1, 1 * dvalues = dvalues - the chain rule
      self.dinputs = dvalues.copy()

   # Calculate predictions for outputs
   def predictions(self, outputs):
      return outputs
        
# Mean Squared Error loss
class Loss_MeanSquaredError(Loss): # L2 loss
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss): # L1 loss
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Input "layer"
class Layer_Input:
   # Forward pass
   def forward(self, inputs, training):
      self.output = inputs
