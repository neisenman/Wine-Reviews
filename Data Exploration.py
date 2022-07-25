import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

red = pd.read_csv("https://raw.githubusercontent.com/csc371-machinelearning/project1-jordan_cameron/master/data/winequality-red.csv?token=GHSAT0AAAAAABRLBQ6XODTPLN5WWSGCYOCWYQ5PE7Q",delimiter=";")
white = pd.read_csv("https://raw.githubusercontent.com/csc371-machinelearning/project1-jordan_cameron/master/data/winequality-white.csv?token=GHSAT0AAAAAABRLBQ6XQIS5WLR67GCSQHSQYQ5PFNQ",delimiter=";")

red.columns

#    10 - sulphates
#    Output variable (based on sensory data): 
#    12 - quality (score between 0 and 10)



plt.plot(red['alcohol'], red['quality'],'.')
plt.xlabel('alcohol', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("Red Wine")
plt.show()

plt.plot(white['alcohol'], white['quality'],'.')
plt.xlabel('alcohol', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("White Wine")
plt.show()


plt.plot(red['volatile acidity'], red['quality'],'.')
plt.xlabel('volatile acidity', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("Red Wine")
plt.show()

plt.plot(white['volatile acidity'], white['quality'],'.')
plt.xlabel('volatile acidity', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("White Wine")
plt.show()


plt.plot(red['density'], red['quality'],'.')
plt.xlabel('density', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("Red Wine")

plt.show()

plt.plot(white['density'], white['quality'],'.')
plt.xlabel('density', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("White Wine")

plt.show()

plt.plot(red['sulphates'], red['quality'],'.')
plt.xlabel('sulphates', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("Red Wine")

plt.show()

plt.plot(white['sulphates'], white['quality'],'.')
plt.xlabel('sulphates', color='#1C2833')
plt.ylabel('quality', color='#1C2833')
plt.title("White Wine")
plt.show()

