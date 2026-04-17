# House_Price_Prediction


This is my first Machine Learning project where I built a complete pipeline to predict house prices and deployed it as an interactive web application using Streamlit.


## 🚀 Project Overview

The goal of this project is to predict the sale price of a house based on various features such as:

- Lot Area
- Year Built & Year Remodelled
- Overall Condition
- Basement Area
- Zoning & Property Type
- Exterior Material

The model is trained using **Linear Regression** and deployed using **Streamlit** for real-time predictions.


## 🧠 Machine Learning Workflow

1. Data Cleaning
   - Handled missing values
   - Removed unnecessary columns (e.g., Id)

2. Feature Engineering
   - One-hot encoding for categorical variables
   - Feature selection

3. Model Training
   - Linear Regression model trained on processed dataset

4. Model Saving
   - Saved trained model using `pickle`
   - Saved feature columns to ensure consistency during prediction

5. Web App Development
   - Built an interactive UI using Streamlit
   - Users can input house details and get predictions instantly


## 🖥️ App Features

- User-friendly interface
- Real-time price prediction
- Handles categorical inputs using one-hot encoding
- Displays input summary and estimated price


## 📂 Project Structure

app.py   # Streamlit app
model.pkl  # Trained ML model
model_columns.pkl   # Feature columns used during training
house_price_prediction.ipynb   # Model training notebook
requirements.txt   # Dependencies


## ⚙️ Installation & Setup

1. Clone the repository:
git clone <your-repo-link>
cd house_prediction

2. Install dependencies:
pip install -r requirements.txt

3.Run the app:
streamlit run app.py
