import numpy as np
import matplotlib.pyplot as plt
from normalize import * 
import random
from sklearn.model_selection import train_test_split

def linearFit(X, y, lr=0.0001,tolerance = 0.007):  
    '''
    This function is the gradient descent for our linear regression.
    @param X: input series of dataframe
    @param y: output series of dataframe
    @param learning_rate: how quickly the gradient descent changes
    @param tolorance: helps determine when the gradient descent is finished
    @returns the desired betas for a linear model
    https://iq.opengenus.org/stochastic-gradient-descent-sgd/#:~:text=Implemen
    tation%20of%20Stochastic%20Gradient%20Descent%20Basics%20of%20Gradient,con
    vex%20function%20by%20means%20of%20using%20iteration.%20
    '''
    
    betas = np.array([0.0,0.0])
    old_rss = 99999999999
    new_rss = 9999999
    xPi = np.array(X)
    yPi = np.array(y)
    while (abs(old_rss - new_rss)) > tolerance:
        
        indexes = np.random.randint(0, len(X), 1) 
        
        Xs = np.take(xPi, indexes)
        ys = np.take(yPi, indexes)
        N = len(Xs)
        
        f = ys - (betas[1]*Xs + betas[0])
    
        betas[1] -= lr * (-2 * Xs.dot(f).sum() / N)
        betas[0] -= lr * (-2 * f.sum() / N)

        old_rss = new_rss
        new_rss = np.sum(betas[0] - betas[1] * X - y ** 2)
    
    return betas

def linearPred(betas,X):
    """
    This function gives a prediction for each of the
    @param X: input series of dataframe
    @param y: output series of dataframe
    @returns returns a series of predictions
    """

    return betas[0] + X*betas[1]


def linearRegression(X,y,lr=.001):
    """
    Creates a linear regression and plots it on a grapth
    @param X: input series of dataframe
    @param y: output series of dataframe
    @param learning_rate: how quickly the gradient descent changes
    """

    betas = linearFit(X=X, y=y, lr=0.001)
    x1 = np.linspace(-5,5,100)
    y1 = betas[0] + betas[1]*x1

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.plot(X, y,'.')
    plt.plot(x1, y1, '-r', label='y=2x+1')
    plt.title('Graph of y=x+1')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

def trainAndTest(df,x,y,iterations = 10):
    """
    Performs data validation over a random sample of 10 models, returning the residual 
    sum of squares
    @param df: dataframe
    @param iterations: how many random samples of the data to run a regression
    @return the residual sum of squares, as calculated by averaged bets numbers
    """
    avgBetas = np.array([0.0,0.0])
    for i in range(5):
        print(i)
        train, test = train_test_split(df, test_size=0.2,random_state = random.randint(0,100))

        betas = linearFit(X=train[x], y=train[y], lr=0.0001,tolerance = 0.001)
        avgBetas += betas
    
    train, test = train_test_split(df, test_size=0.2,random_state = random.randint(0,100))
    avgBetas = avgBetas/iterations
    predicted = linearPred(betas=avgBetas, X = test[x])
    print(calcError(y_pred=predicted, y=test[y]))
    return calcError(y_pred=predicted, y=test[y])