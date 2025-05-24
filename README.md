# 🏠 House Price Prediction using Linear Regression

This project builds a simple machine learning model using **Linear Regression** to predict house prices based on the **size of the house in square feet**.

---

## ✅ What This Model Does

The model is trained to predict the **price of a house** based on one input feature:

- 🏡 **Area (square feet)**

It learns from a small dataset of house areas and their corresponding prices and then makes predictions for new, unseen house sizes.

### 📦 Example:

If a user inputs:
```
Enter house size in square feet: 2200
```
The model might predict:
```
Predicted price: $270000
```

This is based on a learned pattern from training data such as:

| Area (sqft) | Price ($) |
|-------------|------------|
| 1000        | 100000     |
| 1500        | 150000     |
| 2000        | 200000     |

---

## 🧠 How the Model Works (Explained Simply)

This is a **Linear Regression model**, which tries to fit the best straight line through the data.

### Step-by-Step:

1. **Training Data**  
   You provide historical data with known `area` and `price`.

2. **Model Learning**  
   The model finds a formula:
      ```
      price = a * area + b
      ```
where:
- `a` is the slope (how much the price increases per extra square foot)
- `b` is the intercept (base price)

3. **Prediction**  
When you give it a new `area`, it plugs it into the formula and returns a predicted price.

4. **Evaluation**  
The model's accuracy is tested using metrics like **Mean Squared Error** to see how close predictions are to actual prices.

---

## ⚙️ Tools Used

- Python 3.10+
- Pandas (Data handling)
- NumPy (Numerical operations)
- Scikit-learn (LinearRegression, metrics)
- Joblib (Saving/loading model)
- Matplotlib (Plotting the regression line)

---

## 🚀 How to Run

1. Install required libraries:
```
pip install pandas numpy matplotlib scikit-learn joblib
```

2. Run the code:
```
python code_1.py
```

3. Input the house size when prompted.

---
## 📁 Model Saving
After training, the model is saved to disk as:
```
house_price_model.pkl
```
It can be loaded later to make predictions without retraining.

---
## Future Ideas
- ➕ Add More Features

  Include additional features like:

  - Location

  - Number of bedrooms

  - Age of the house

  - Type of property (flat, villa, etc.)

- 📄 Use Real Dataset from CSV

      Load actual housing market data from a `.csv` file and train the model on a larger, more realistic dataset.

- 🖥️ Build a GUI or Web App
  
     Use tools like Tkinter, Streamlit, or Flask to create an interactive interface where users can enter details and see predictions easily.

- 🛠️ Add Better Error Handling & Visualization

  - Improve how input errors are caught and handled.

  - Visualize not just the regression line, but also data distributions and feature correlations.

