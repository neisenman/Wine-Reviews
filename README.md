# Wine-Reviews

Objective Description
I looked at two data sets concerning red wine and white wine with the goal of running a regression to predict a particular wine's quality as rated by sommeliers, based on various factors – including but not limited to alcohol, acidity, and sugar content. 

Results
I ended up implimenting two different types of models, a linear and multivariate model. The following data is the residual sum of squares for each of the top three univariate models in red and white wines, respectively, when the variable is:

Linear Models: 
Red Wine
    • Alcohol: 0.792
    • Volatile Acidity: 0.868
    • Density: 0.944
White Wine
    • Sulfates: .84
    • Alcohol .89
    • Volatile_acidity: 0.77
      
Analysis
The linear models were moderately effective in predicting a wine's quality. Citric acid was the strongest predictor of any feature. Given my results, I made multivariate models with the strongest predictors in the single-dimensional linear case. I got the following results:

Multivariable Models:
    • Alcohol, Density: 0.232   
    • Alcohol, Volatile Acidity: 0.317
    • Density, Volatile Acidity: 0.180
White Wine
    • alcohol, volatile_acidity 0.27
    • sulfates, volatile_acidity 0.08
    • alcohol, sulfates 0.19

Analysis
	The multivariate regression was much more effective in predicting a wine's quality for red and white wines. As seen in the data, the most effective variables for predicting the quality of red wine were Density and Volatile Acidity. The most effective variables for predicting the quality of white wine were Sulfates and Volatile acidity. Based on these results, I proceeded to combine each of the most effective variables in hopes of achieving greater accuracy through multivariate regressions. 

Data exploration
	Keeping the data in mind, I built the requisite linear and polynomial regression models by hand and used the pre-built python packages for multivariate-linear regression. Then I tested my intuition against the data – I tested the accuracy of each model, calculating the average of the residual sum of squares. 

Method
	Further exploration of the data required building requisite models to test my hypotheses. Thus, I built my own models to test our hypotheses. I started by implementing gradient descent for a linear model. My model acted erratically, but I resolved these issues by normalizing my data. I proceeded to implement gradient descent on my linear model. Building on my linear model, I then implemented my quadratic regression with only a few changes to my original code. Finally, I saw an opportunity to generalize my stochastic descent method for any polynomial. The regression model worked as intended; however, it exhibited more erratic behavior. This problem was especially true when we tested the regression model on higher-ordered polynomials. For example, cubic functions became increasingly challenging for the computer. Not only did the regression take exponentially more time, but the line would also frequently change, as the stochastic gradient descent would output the wrong betas. Implementing our three-dimensional multivariate regression model, we used the statsmodels python package. I also created some three-dimensional figures using matplotlib. 
Pre-processing
	Building the models also required processing our data and fine-tuning my hyper-parameters. In pre-processing, I replaced the spaces in the column names with underscores. The data needed to be normalized before being put into our model. I created my own model that normalized the data in each column before placing it into my models. I found that tolerance and learning rate significantly affected my models concerning my hyper-parameters. In particular, a tolerance of .01 resulted in extremely quick but inaccurate models. In most cases, I used a tolerance of 0.0001, which yielded relatively quick and accurate models. Concerning the learning rate, I generally used 0.001. Similar to tolerance, this learning rate yielded quick and accurate models. To be sure, these parameters struggled with a cubic model, as it took fundamentally more computing power. 
