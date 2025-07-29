import numpy as np


def gradient_descent(x,y, lr=0.1 , epochs=1000):
    
    m,b = 0.0, 0.0
    
    
    '''
    Mean Squared Error (MSE)
    MSE = (1/n) * Σ_{i=1}^n (y_i - ŷ_i)²
    where ŷ_i = m * x_i + b

    Gradient w.r.t. m:
    ∂MSE/∂m = -(2/n) * Σ_{i=1}^n [ x_i * (y_i - ŷ_i) ]
            =  (2/n) * Σ_{i=1}^n [ (ŷ_i - y_i) * x_i ]

    Gradient w.r.t. b:
    ∂MSE/∂b = -(2/n) * Σ_{i=1}^n (y_i - ŷ_i)
            =  (2/n) * Σ_{i=1}^n (ŷ_i - y_i)
    '''

    
    for epoch in range(epochs):
        y_pred = m*x + b
        error = y- y_pred
        cost = np.mean(error**2)
        
        dm = -2*np.mean(error*x)
        db = -2*np.mean(error)
        
        
        b -= db*lr
        m -= dm*lr
        print(f"m={m} , b={b} ,Epoch {epoch}: Cost= {cost}")


if __name__ == "__main__":
    
    x = np.array([1,2,3,4,5])
    y = np.array([5,7,9,11,13])
    
    gradient_descent(x,y)