#Project1: House Price Prediction (Regression- scikit-learn)
#We'll Load sample data , Train a linear regression model ,Predict house prices , visualize the results

#import basic libraries
import pandas as pd #for handling tabular data
import numpy as np #for numerical operations
import matplotlib.pyplot as plt #for plotting graphs

#import ml tools
from sklearn.linear_model import LinearRegression  #the regression model
from sklearn.model_selection import train_test_split  #to split data into train and test
from sklearn.metrics import mean_squared_error #to measure prediction error
import joblib #for saving and loading the model

#creating sample housing data
data={
    'Area' :[1000, 1500, 2000, 2500, 3000], #house sizes
    'Price' :[150000, 200000, 250000, 300000, 350000] #corresponding prices
}

#Load data into a dataFrame
df = pd.DataFrame(data)
#print(df) #show the dataset

#Split data into features (x) and target (y)
X= df[['Area']] #independent variable (ID) and 2D input array 
y = df['Price'] #dependent variable and 1D output array

#Split data:- 80% for trainning, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42) #test_size=0.2 means 20% of data will be used for testing

#Create a Linear Regression model
model = LinearRegression() 

#Trai (fit) the model on training data
model.fit(X_train,y_train)

#Save the trained model to a file so we caan use it later without retaining
joblib.dump(model,'house_price_model.pkl')
print("Model trained and saved as 'house_price_model.pkl'")

#Take ser input for prediction
try:
    #Ask user to enter house size
    user_input = float(input("Enter house size in square feet: "))
    #Load the previously saved model
    model = joblib.load('house_price_model.pkl')
    #Predict the price for the entered size
    input_df = pd.DataFrame([[user_input]], columns=["Area"])
    predicted_price = model.predict(input_df)[0]  #get the first element of the prediction array
    #Show the predictd price
    print(f"Predcted price: ${int(predicted_price.item())}")
except Exception as e:
    #In case of any error , show an error message
    print("Error:Please ener a valid number. Details: ", e)

#Make predictions on the test set
#y_pred = model.predict(X_test)

#Show actual vs predicted values
#print("Actual Prices: ", list(y_test))
#print("Predicted Prices: ", list(y_pred))

#Print Mean Squared Error to evaluate prediction accuracy
#print("Mean Squared Error: ",mean_squared_error(y_test, y_pred))
'''
OPTIONAL STEP TO SHOW GRAPH

#Visualize the regression line on the original data

plt.scatter(X, y, color='blue', label='Actual Data') #actual data points
plt.plot(X, model.predict(X), color='red', label='Regression line') #model's predicted line
#Add labels and title
plt.xlabel('Size (Sqft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction')
plt.legend()
#Show the plot
plt.show()
'''

'''
Later Load the model for use anywhere

import joblib
# Load the trained model
model= joblib.load('house_price_model.pkl)
#Predict using loaded model
size=[[2200]]
price = model.predict(size)
print(f"Predicted price for 2200 sqft: ${int(price[0])}")
'''