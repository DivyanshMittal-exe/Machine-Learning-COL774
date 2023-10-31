# %%
import numpy as np
from abc import ABC, abstractmethod
import numpy as np 
import sys
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import precision_recall_fscore_support


# %%
def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y

def get_metric(y_true, y_pred):
    '''
    Args:
        y_true: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
        y_pred: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
                
    '''
    results = classification_report(y_pred, y_true)
    print(results)

# %%
class BaseLayer(ABC):
    def __init__(self, input_size, output_size):
        self.weights =  np.random.normal(0, 0.01, (input_size + 1, output_size)) # Includes bias
        self.input = None
        self.output = None
        # self.bias = np.random.rand(output_size, 1)
    
    @abstractmethod
    def activation(self, input):
        pass
    
    def forward(self, input):
        input_data_with_bias = np.hstack((np.ones((input.shape[0],1)), input))
        self.input = input_data_with_bias
        
        z = np.dot(input_data_with_bias, self.weights)
        
        self.output = self.activation(z)
        
        # activation = 1/(1+np.exp(-(np.dot(input_data_with_bias, self.weights))))
        return self.output

    @abstractmethod
    def backward(self, grad_output, learning_rate):
        pass

# %%
class SigmoidLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
    
    def activation(self, input_data):
        return 1/(1 + np.exp(-input_data))
    
    def backward(self, grad_output, learning_rate):
        grad_input = grad_output * self.output * (1 - self.output)
        grad_weights = np.dot(self.input.T, grad_input)
        self.weights -= learning_rate * grad_weights
        return grad_input
        

# %%
class ReluLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
    
    def activation(self, input_data):
        return np.maximum(0.0, input_data)
    
    def backward(self, grad_output, learning_rate):
        grad_relu = (self.output > 0).astype(float)
        grad_input = grad_output * grad_relu
        grad_w = np.dot(self.input.T, grad_input)

        # Update weights
        self.weights -= learning_rate * grad_w

        return grad_input
        

# %%
class SoftPlusLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
    
    def activation(self, input_data):
        return np.log(1+np.exp(input_data))
    
    def backward(self, grad_output, learning_rate=0.1):
        exp_z = np.exp(np.dot(self.input, self.weights))
        grad_input = grad_output * exp_z / (1 + exp_z)
        grad_w = np.dot(self.input.T, grad_input)

        self.weights -= learning_rate * grad_w

        return grad_input

# %%
class SoftMaxLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
    
    def activation(self, input_data):
        exp_input = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))  # Numerical stability
        return exp_input / np.sum(exp_input, axis=1, keepdims=True)
    
    def backward(self, grad_output, learning_rate=0.1):

        grad_input = grad_output * self.output * (1 - self.output)

        grad_w = np.dot(self.input.T, grad_input)

        self.weights -= learning_rate * grad_w

        weight_with_bias = self.weights[1:, :]
        w_temp = np.dot(grad_input, weight_with_bias.T)

        return w_temp

# %%
class NN:
    def __init__(self, layers) -> None:
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad , learning_rate=0.1):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

    def cross_entropy_loss(self, output, target):
        # epsilon = 1e-15  # Avoid log(0)
        # output = np.clip(output, epsilon, 1 - epsilon)
        return -np.sum(target * np.log(output)) / len(output)

    def grad_cross_entropy_loss(self, output, target):
        # epsilon = 1e-15
        # output = np.clip(output, epsilon, 1 - epsilon)
        return (output - target) # Len output divide check

    def train(self, X_train, y_train, learning_rate=0.1, epochs=100, batch_size=32):
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            print(f"Epoch {epoch}: ")
            
            for i in range(0, num_samples, batch_size):
                batch_X = X_train_shuffled[i:i + batch_size]
                batch_y = y_train_shuffled[i:i + batch_size]
                
                output = self.forward(batch_X)
                grad = self.grad_cross_entropy_loss(output, batch_y)

                self.backward(grad, learning_rate)
                
            output = self.forward(X_train)
            total_loss = self.cross_entropy_loss(output, y_train)

            print(f"Epoch {epoch}: Loss {total_loss}")
                
            

        
        
        # for epoch in range(epochs):
            # output = self.forward(X)
            # loss = self.cross_entropy_loss(output, Y)
            # grad = self.grad_cross_entropy_loss(output, Y)
            
            # self.backward(grad, learning_rate)
            
            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}: loss {loss:.3f}")
    
    def __call__(self, X):
        return self.forward(X)

    

# %%

x_train_path = 'part_b/x_train.npy'
y_train_path = 'part_b/y_train.npy'

X_train, y_train = get_data(x_train_path, y_train_path)

x_test_path = 'part_b/x_test.npy'
y_test_path = 'part_b/y_test.npy'

X_test, y_test = get_data(x_test_path, y_test_path)

#you might need one hot encoded y in part a,b,c,d,e
label_encoder = OneHotEncoder(sparse_output = False)
label_encoder.fit(np.expand_dims(y_train, axis = -1))

y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))

# %%
hidden_layer = 10

nn = NN([
    SigmoidLayer(1024, hidden_layer),
    SoftMaxLayer(hidden_layer, 5)
])

# %%
nn.train(X_train, y_train_onehot, 0.01, 20, 32)

predictions = nn(X_test)

predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test_onehot, axis=1)

print(X_test.shape)
print(predictions.shape)

print(predicted_classes.shape)
print(actual_classes.shape)
# Compute precision, recall, and F1 scores for each class
precision, recall, f1, _ = precision_recall_fscore_support(actual_classes, predicted_classes, average='macro')

print(precision, recall, f1)
# f1_scores.append(f1)


