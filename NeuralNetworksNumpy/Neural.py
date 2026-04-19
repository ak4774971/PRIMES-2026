import numpy as np

class Generic(): 
    def __init__(self):
        self.nbParams = None
        self.saveX = None

    def setParams(self, params): 
        pass

    def getParams(self): 
        return None

    def forward(self, X): 
        self.saveX = np.copy(X)
        return None

    def backward(self, gradSortie):  
        gradLocal = None
        gradEntree = None
        return gradLocal, gradEntree

class Arctan(Generic): 
    def __init__(self):
        self.nbParams = 0
        self.saveX = None

    def setParams(self, params): 
        pass

    def getParams(self): 
        pass

    def forward(self, X): 
        self.saveX = np.copy(X)
        return np.arctan(self.saveX)

    def backward(self, gradSortie):  
        gradLocal = None
        if self.saveX is None:
            return gradLocal, None
        gradEntree = gradSortie / (1 + self.saveX**2)
        return gradLocal, gradEntree

class Sigmoid(Generic):
    def __init__(self):
        self.nbParams = 0
        self.saveX = None

    def setParams(self, params):
        pass

    def getParams(self):
        return None

    def forward(self, X):
        self.saveX = np.copy(X)
        return 1.0 / (1.0 + np.exp(-self.saveX))

    def backward(self, gradSortie):
        gradLocal = None
        if self.saveX is None:
            return gradLocal, None
        s = 1.0 / (1.0 + np.exp(-self.saveX))
        gradEntree = gradSortie * s * (1.0 - s)
        return gradLocal, gradEntree
    
class Dense(Generic): 
    def __init__(self, nbEntree, nbOutput): 
        self.nEntree = nbEntree
        self.nSortie = nbOutput
        self.nbParams = self.nEntree * self.nSortie + self.nSortie
        self.A = np.random.randn(self.nSortie, self.nEntree)
        self.b = np.random.randn(self.nSortie)

    def setParams(self, params): 
        self.A = params[:self.nEntree * self.nSortie].reshape(self.nSortie, self.nEntree)
        self.b = params[self.nEntree * self.nSortie:]

    def getParams(self): 
        return np.concatenate([self.A.ravel(), self.b.ravel()])

    def forward(self, X): 
        self.saveX = np.copy(X)
        return self.A.dot(X) + self.b.reshape(-1, 1)

    def backward(self, gradSortie):  
        gA = gradSortie.dot(self.saveX.T)
        gb = np.sum(gradSortie, axis=1)
        gradLocal = np.concatenate([gA.ravel(), gb.ravel()])
        gradEntree = self.A.T.dot(gradSortie)
        return gradLocal, gradEntree
    
class LossL2(Generic): 
    def __init__(self, D): 
        self.nbParams = 0
        self.saveD = D
        self.saveX = None

    def setParams(self): 
        pass
        
    def getParams(self): 
        pass

    def forward(self, X): 
        self.saveX = np.copy(X)
        return 0.5 * np.linalg.norm(X - self.saveD)**2
    
    def backward(self, gradSortie): 
        gradLocal = None
        return gradLocal, self.saveX - self.saveD
    
class Network(Generic): 
    def __init__(self, listLayers): 
        self.listLayers = listLayers 
        self.nbParams = sum(layer.nbParams for layer in self.listLayers if layer.nbParams is not None and layer.nbParams > 0)
        
    def setParams(self, params): 
        offset = 0
        for layer in self.listLayers:
            if layer.nbParams is not None and layer.nbParams > 0:
                layer.setParams(params[offset:offset + layer.nbParams])
                offset += layer.nbParams
            
    def getParams(self): 
        parts = []
        for layer in self.listLayers: 
            p = layer.getParams()
            if p is not None: 
                parts.append(p)
        return np.concatenate(parts)

    def forward(self, X):
        Z = np.copy(X)
        for layer in self.listLayers:
            Z = layer.forward(Z)
        return Z

    def backward(self, gradSortie):
        gradLocals = []
        g = gradSortie
        for layer in reversed(self.listLayers):
            gl, g = layer.backward(g)
            if gl is not None:
                gradLocals.append(gl)
        return np.concatenate(gradLocals), g

class IlogitAndKL(Generic): 
    def __init__(self, D): 
        self.nbParams = 0
        self.saveD = D
        self.saveX = None

    def setParams(self, params): 
        pass
        
    def getParams(self): 
        return None

    def forward(self, X): 
        self.saveX = np.copy(X)
        logSumExp = np.log(np.sum(np.exp(self.saveX), axis=0))
        return np.sum(logSumExp - np.sum(self.saveX * self.saveD, axis=0))
    
    def backward(self, gradSortie): 
        gradLocal = None
        yTilde = np.exp(self.saveX) / np.sum(np.exp(self.saveX), axis=0)
        return gradLocal, yTilde - self.saveD
