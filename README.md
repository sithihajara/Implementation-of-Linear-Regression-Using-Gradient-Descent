# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Read the txt file using read_csv
3. Use numpy to find theta,x,y values
4. To visualize the data use plt.plot

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sithi hajara I
RegisterNumber:  212221230102
*/
```
```
#import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("ex1.txt",header=None)

plt.scatter(df[0],df[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Take in a np array X,y,theta and generate the cost function of using theta as parameter in a linear regression model
"""
def computeCost(X,y,theta):
    m=len(y) #length of the training data
    h=X.dot(theta) #hypothesis
    square_err=(h-y)**2
    
    return 1/(2*m)*np.sum(square_err) #returning J

df_n=df.values
m=df_n[:,0].size
X=np.append(np.ones((m,1)),df_n[:,0].reshape(m,1),axis=1)
y=df_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta) #call the function

"""
Take in np array X,y and theta and update theta by taking num_iters gradient steps with learning rate of alpha 
return theta and the list of the cost of theta during each iteration
"""
def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(),(predictions -y))
        descent = alpha*(1/m )*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(df[0],df[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

"""
Takes in numpy array of x and theta and return the predicted value of y based on theta
"""
def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
```
## Output:
![4](https://user-images.githubusercontent.com/94219582/193459033-f370f153-2df8-4bf0-b850-ccfd64ab047b.png)
![6](https://user-images.githubusercontent.com/94219582/193459052-b0e1c47e-972b-4be8-b606-3ea0e83ebea4.png)
![7](https://user-images.githubusercontent.com/94219582/193459056-a2783369-bf02-4e18-8d0b-18d5eed45d28.png)
![9](https://user-images.githubusercontent.com/94219582/193459110-f04ebeeb-10d8-4b1b-9072-1c276a991b7c.png)
![10](https://user-images.githubusercontent.com/94219582/193459122-97316650-1b6d-4bc0-93c2-09c901710581.png)
![12](https://user-images.githubusercontent.com/94219582/193459128-26eb3619-7fe6-4472-a9fe-137cea1f6dc8.png)
![14](https://user-images.githubusercontent.com/94219582/193459143-af1ef265-feaa-40dd-bf38-2123930b4323.png)
![15](https://user-images.githubusercontent.com/94219582/193459160-93da5608-9cd6-4591-8df7-7a8e7f8fbee0.png)
![16](https://user-images.githubusercontent.com/94219582/193459168-3f3d36c6-aaee-468e-9f70-cb0f33974aa2.png)
![17](https://user-images.githubusercontent.com/94219582/193459177-9c49c746-9be6-4f9a-82d9-550bbe75db1b.png)
![18](https://user-images.githubusercontent.com/94219582/193459185-74e9314b-a159-47f3-8cf0-bc839f2a0b43.png)
![19](https://user-images.githubusercontent.com/94219582/193459191-a6d47bde-ecd8-4a08-9505-aa779c865ec6.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
