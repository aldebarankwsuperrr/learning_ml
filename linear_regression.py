import numpy as np

class linear_regression:
    def __init__(self, learning_rate = 0.01, iteration = 1000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.bias = 0
    
    def fit(self, features, labels):
        self.m, self.n = features.shape
        self.weights = np.zeros(self.n)
        
        for i in range(self.iteration):
            y_predict = self.predict(features)
            cost = (1/(2*self.m)) * np.sum((y_predict - labels)**2)
            derevative_weight = (1/self.m) * np.dot(features.T, y_predict-labels )
            derevative_bias =  (1/self.m) * np.sum(y_predict-labels)
            
            self.weights = self.weights - (self.learning_rate * derevative_weight)
            self.bias = self.bias - (self.learning_rate * derevative_bias)
            
            print("{iteration} : {error}".format(iteration = i, error = cost))
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights ) + self.bias
    
            
def main():
    x = np.array([[1], [3], [5], [7]])
    y = np.array([5, 11, 17, 23])
    
    model = linear_regression()
    model.fit(x,y)
    
    predict = model.predict(11)
    
    print(predict)
    
if __name__ == '__main__':
    main()
    
    
    
    
        