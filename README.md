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
![1](https://user-images.githubusercontent.com/94219582/193458585-98ae2e50-63f3-4f19-92d5-0c85429a3e3b.png)
![2](https://user-images.githubusercontent.com/94219582/193458589-ef53c10b-e8a8-452c-92af-f57![3](https://user-images.githubusercontent.com/94219582/193458594-4ea2c1a8-65d3-4dd2-8e96-b8ba8a05cc4e.png)
843407218.png)
![4](https://user-images.githubusercontent.com/94219582/193458605-23cedf78-9bcc-4d1e-9ab2-2d3fb5ba8f7b.png)
![5](https://user-images.githubusercontent.com/94219582/193458609-5e255293-365f-49c3-b1e3-47cbae6c75e3.png)
![6](https://user-images.githubusercontent.com/94219582/193458617-45d39fcc-864b-479c-83fc-b2b57fa3ad6b.png)
![7](https://user-images.githubusercontent.com/94219582/193458625-ed4472b5-b6d6-415d-8e51-27d2dce2fe55.png)
![8](https://user-images.githubusercontent.com/94219582/193458631-8ac62eff-7e3e-47be-a0d1-8502fb797f38.png)
![9](https://user-images.githubusercontent.com/94219582/193458636-4bba6e4d-324e-4c99-b396-7c67d98f7a7f.png)
![10](https://user-images.githubusercontent.com/94219582/193458646-fe8837c5-89e2-4f46-bb02-ba038a866557.png)
![11](https://user-images.githubusercontent.com/94219582/193458648-762f928b-3181-4fc2-bb7e-6a0f65cb8ef7.png)
![12](https://user-images.githubusercontent.com/94219582/193458655-f2957519-12a3-457f-8fa1-ade8![14](https://user-images.githubusercontent.com/94219582/193458660-4207e1d9-ab6d-4712-88f4-08aabd21c68c.png)
d312f32f.png)
![15](https://user-images.githubusercontent.com/94219582/193458670-f03ba487-0bd7-44a5-bffb-3c0ca995b51a.png)
![16](https://user-images.githubusercontent.com/94219582/193458674-b8c130a3-599c-40d0-ac79-a350819dda0d.png)
![17](https://user-images.githubusercontent.com/94219582/193458677-c612c2a9-298d-4f6b-8b03-47fe3fb003db.png)
![18](https://user-images.githubusercontent.com/94219582/193458682-f8facda2-c529-4054-9c29-a0110a46de27.png)
![19](https://user-images.githubusercontent.com/94219582/193458690-fd9c3d81-9d4d-4d0a-b0e8-51fecba83db8.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
