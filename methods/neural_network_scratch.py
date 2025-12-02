'''Feed-Forward Neural Network regressor - from scratch'''
#We will use the ReLU as the activation function
#Preferably 2 hidden layers (selected features (43)-64-32-1)
#Loss function: MSE

# ...
import numpy as np
import pandas as pd
import numpy as np



class NeuralNetworkScratch:

    def initialize_weights_normal(self, mean, std, shape):  
    #np.random.seed(random_state)  
        return np.random.normal(mean, std, shape)

    def relu(self, x): 
        return np.maximum(0, x)

    def relu_derivative(self, x): 
        return np.where(x > 0, 1.0, 0.0)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def __init__(self, input_dim, hidden1_dim=64, hidden2_dim=32, output_dim=1, learning_rate=0.001, epochs=100, number_of_batches=100, random_state=42, patience=20, learning_shrink=False):
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.min_learning_rate = 1e-5
        self.window_size = 3
        self.factor = 0.9
        self.min_delta = 1e-4
        self.epochs = epochs
        self.number_of_batches = number_of_batches
        self.random_state = random_state
        self.params = {
        # 1st hidden layer weights and biases
        "wh1": self.initialize_weights_normal(0, np.sqrt(2/self.input_dim), (self.input_dim, self.hidden1_dim)),  # input nodes, hidden nodes
        "bh1": np.zeros((1, self.hidden1_dim)),  # bias term for hidden layer
        # 2nd hidden layer weights and biases
        "wh2": self.initialize_weights_normal(0, np.sqrt(2/self.hidden1_dim), (self.hidden1_dim, self.hidden2_dim)),  # hidden nodes, hidden nodes
        "bh2": np.zeros((1, self.hidden2_dim)),  # bias term for hidden layer
        # output layer weights and biases
        "wo": self.initialize_weights_normal(0, np.sqrt(2/self.hidden2_dim), (self.hidden2_dim, self.output_dim)),  # hidden nodes, output node
        "bo": np.zeros((1, self.output_dim))   # bias for output layer
        }
        self.validation_loss_history = []
        self.training_loss_history = []
        self.best_validation_loss = float('inf')
        self.patience = patience
        self.learning_shrink = learning_shrink

    def forward_pass(self, X):
        # 1st hidden layer
        z1 = np.dot(X, self.params["wh1"]) + self.params["bh1"]  # Linear transformation
        a1 = self.relu(z1)  # Activation function

        # 2nd hidden layer
        z2 = np.dot(a1, self.params["wh2"]) + self.params["bh2"]  # Linear transformation
        a2 = self.relu(z2)  # Activation function

        # Output layer
        z3 = np.dot(a2, self.params["wo"]) + self.params["bo"]  # Linear transformation
        output = z3  # Linear output for regression 

        cache = {
            "X": X,
            "z1": z1, "a1": a1,
            "z2": z2, "a2": a2,
            "z3": z3, "output": output,
        }
        return output, cache
    
    def backward_pass(self, y_true, cache):
        m = y_true.shape[0]  # number of samples

        #Loss gradient
        dZ3 = 2*(cache["output"] - y_true) / m  # Derivative of loss w.r.t z3
        # Output layer gradients
        dWo = np.dot(cache["a2"].T, dZ3)  # Gradient w.r.t weights of output layer
        dBo = np.sum(dZ3, axis=0, keepdims=True)  # Gradient w.r.t bias of output layer

        # 2nd hidden layer gradients
        dA2 = np.dot(dZ3, self.params["wo"].T)  # Derivative of loss w.r.t a2
        dZ2 = dA2 * self.relu_derivative(cache["z2"])  # Derivative of loss w.r.t z2
        dWh2 = np.dot(cache["a1"].T, dZ2)  # Gradient w.r.t weights of 2nd hidden layer
        dBh2 = np.sum(dZ2, axis=0, keepdims=True)  # Gradient w.r.t bias of 2nd hidden layer

        # 1st hidden layer gradients
        dA1 = np.dot(dZ2, self.params["wh2"].T)  # Derivative of loss w.r.t a1
        dZ1 = dA1 * self.relu_derivative(cache["z1"])  # Derivative of loss w.r.t z1
        dWh1 = np.dot(cache["X"].T, dZ1)  # Gradient w.r.t weights of 1st hidden layer
        dBh1 = np.sum(dZ1, axis=0, keepdims=True)  # Gradient w.r.t bias of 1st hidden layer

        grads = {
            "dWo": dWo, "dBo": dBo,
            "dWh2": dWh2, "dBh2": dBh2,
            "dWh1": dWh1, "dBh1": dBh1
        }
        return grads
    
    def update_parameters(self, grads):
        # Update weights and biases using gradient descent
        self.params["wo"] -= self.learning_rate * grads["dWo"]
        self.params["bo"] -= self.learning_rate * grads["dBo"]
        self.params["wh2"] -= self.learning_rate * grads["dWh2"]
        self.params["bh2"] -= self.learning_rate * grads["dBh2"]
        self.params["wh1"] -= self.learning_rate * grads["dWh1"]
        self.params["bh1"] -= self.learning_rate * grads["dBh1"]

    def train(self, X, y):
        # Shuffle and split the data into training and validation sets
        np.random.seed(self.random_state)
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        X_shuffled_training = X_shuffled[:int(X.shape[0] * 0.8)]
        y_shuffled_training = y_shuffled[:int(X.shape[0] * 0.8)]
        X_validation = X_shuffled[int(X.shape[0] * 0.8):]
        y_validation = y_shuffled[int(X.shape[0] * 0.8):]
        no_improve_count = 0
        lr_counter = 0
        
        # Training loop
        for epoch in range(self.epochs):
            permutation_epoch = np.random.permutation(X_shuffled_training.shape[0])
            X_epoch_shuffled = X_shuffled_training[permutation_epoch]
            y_epoch_shuffled = y_shuffled_training[permutation_epoch]
            
            # Mini-batch training
            for batch in range(self.number_of_batches):
                start = batch * (X_epoch_shuffled.shape[0] // self.number_of_batches)
                end = start + (X_epoch_shuffled.shape[0] // self.number_of_batches)
                X_batch = X_epoch_shuffled[start:end]
                y_batch = y_epoch_shuffled[start:end]

                # Forward pass
                output, cache = self.forward_pass(X_batch)

                # Backward pass
                grads = self.backward_pass(y_batch, cache)

                # Update parameters
                self.update_parameters(grads)

            # Calculate and print loss for training and validation sets
            train_output, _ = self.forward_pass(X_shuffled_training)
            train_loss = self.mse(y_shuffled_training, train_output)
            print(f'Epoch {epoch}, Loss: {train_loss}')
            self.training_loss_history.append([epoch,train_loss])

            validation_output, _ = self.forward_pass(X_validation)
            val_loss = self.mse(y_validation, validation_output)
            print(f'Validation Loss: {val_loss}')
            self.validation_loss_history.append([epoch,val_loss])

            # Learning rate scheduling
            if self.learning_shrink and len(self.validation_loss_history) >= self.window_size:
                improvement = self.validation_loss_history[-self.window_size][1] - self.validation_loss_history[-1][1]
                if improvement < self.min_delta:
                    new_lr = max(self.learning_rate * self.factor, self.min_learning_rate)
                    lr_counter += 1
                    if new_lr < self.learning_rate and lr_counter == self.window_size:
                        print(f"Reducing learning rate from {self.learning_rate} to {new_lr}")
                        self.learning_rate = new_lr
                        lr_counter = 0

            # Early stopping check (training)
            if val_loss < self.best_validation_loss - 1e-5:
                self.best_validation_loss = val_loss
                best_params = self.params.copy()
                no_improve_count = 0
            else:  
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        # Restore best parameters
        print("Best validation loss:", self.best_validation_loss)
        self.params = best_params

    def predict(self, X):
        output, _ = self.forward_pass(X)
        return output