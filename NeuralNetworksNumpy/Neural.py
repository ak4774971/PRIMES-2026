import numpy as np

class Generic(): 
    def __init__(self):
        self.nb_params = 0 # Number of parameters in the layer
        self.save_X = None # Saved layer input (set in forward)
        
    def set_params(self, params): 
        # Set the layer parameters; input is a vector of length self.nb_params
        pass
        
    def get_params(self): 
        # Returns a vector of length self.nb_params containing the layer parameters
        return None
        
    def forward(self, X): 
        # Forward pass; X is the input data
        self.save_X = np.copy(X)
        return None
        
    def backward(self, grad_sortie):  
        # Backpropagation through the layer.
        grad_local = None
        grad_entree = None
        return grad_local, grad_entree
    

class Arctan(Generic): 
    def __init__(self):
        super().__init__()
        self.nb_params = 0 
        
    def forward(self, X): 
        self.save_X = np.copy(X)
        return np.arctan(X)
        
    def backward(self, grad_sortie):  
        grad_local = None
        grad_entree = grad_sortie / (1 + self.save_X**2)
        return grad_local, grad_entree


class Sigmoid(Generic):
    def __init__(self):
        super().__init__()
        self.nb_params = 0
        
    def forward(self, X):
        self.save_X = np.copy(X)
        return 1.0 / (1.0 + np.exp(-X))
        
    def backward(self, grad_sortie):
        grad_local = None
        S = 1.0 / (1.0 + np.exp(-self.save_X))
        grad_entree = grad_sortie * S * (1.0 - S)
        return grad_local, grad_entree


class ReLU(Generic):
    def __init__(self):
        super().__init__()
        self.nb_params = 0
        
    def forward(self, X):
        self.save_X = np.copy(X)
        return np.maximum(X, 0)
        
    def backward(self, grad_sortie):
        grad_local = None
        grad_entree = grad_sortie * (self.save_X > 0)
        return grad_local, grad_entree


class ABS(Generic):
    def __init__(self):
        super().__init__()
        self.nb_params = 0
        
    def forward(self, X):
        self.save_X = np.copy(X)
        return np.abs(X)
        
    def backward(self, grad_sortie):
        grad_local = None
        grad_entree = grad_sortie * np.sign(self.save_X)
        return grad_local, grad_entree


class Dense(Generic):
    def __init__(self, nb_entree, nb_output):
        super().__init__()
        self.n_entree = nb_entree
        self.n_sortie = nb_output
        self.nb_params = (nb_entree * nb_output) + nb_output
     
        self.A = np.random.randn(self.n_sortie, self.n_entree)
        self.b = np.random.randn(self.n_sortie)
        
    def set_params(self, params):
        split_idx = self.n_entree * self.n_sortie
        self.A = params[:split_idx].reshape(self.n_sortie, self.n_entree)
        self.b = params[split_idx:]
        
    def get_params(self):
        return np.concatenate((self.A.ravel(), self.b.ravel()))
        
    def forward(self, X):
        self.save_X = np.copy(X)
        return self.A.dot(X) + np.outer(self.b, np.ones(X.shape[1]))
        
    def backward(self, grad_sortie):
        g_A = grad_sortie.dot(self.save_X.T)
        g_b = np.sum(grad_sortie, axis=1)
        
        grad_local = np.concatenate((g_A.ravel(), g_b.ravel()))
        
        grad_entree = self.A.T.dot(grad_sortie)
        
        return grad_local, grad_entree


class Loss_L2(Generic):
    def __init__(self, D):
        super().__init__()
        self.nb_params = 0
        self.D = D
        
    def forward(self, X):
        self.save_X = np.copy(X)
        return 0.5 * np.sum((X - self.D)**2)
        
    def backward(self, grad_sortie):
        grad_local = None
        grad_entree = self.save_X - self.D
        return grad_local, grad_entree


class Network(Generic):
    def __init__(self, list_layers):
        super().__init__()
        self.list_layers = list_layers
        self.nb_params = sum(layer.nb_params for layer in self.list_layers)
        
    def set_params(self, params):
        idx = 0
        for layer in self.list_layers:
            if layer.nb_params > 0:
                layer.set_params(params[idx : idx + layer.nb_params])
                idx += layer.nb_params
                
    def get_params(self):
        params_list = []
        for layer in self.list_layers:
            p = layer.get_params()
            if p is not None:
                params_list.append(p)
                
        if len(params_list) > 0:
            return np.concatenate(params_list)
        return None
        
    def forward(self, X):
        self.save_X = np.copy(X)
        Z = np.copy(X)
        for layer in self.list_layers:
            Z = layer.forward(Z)
        return Z
        
    def backward(self, grad_sortie):
        grad = grad_sortie
        grad_local_list = []
        
        for layer in reversed(self.list_layers):
            g_local, grad = layer.backward(grad)
            if g_local is not None:
                grad_local_list.append(g_local)
                
        if len(grad_local_list) > 0:
            # made a mistake but make sure to reverse the gathered local gradients to maintain original sequence order 
            grad_local = np.concatenate(grad_local_list[::-1])
        else:
            grad_local = None
            
        return grad_local, grad
