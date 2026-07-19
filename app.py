# Webpage using Python: streamlit module
# external module: pip install streamlit


# MODULES LOAD
import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pickle
import random
import gdown
import joblib
import os

# DATA LOAD AND X | Y CREATE
try:
    data = fetch_california_housing()
    df = pd.DataFrame(data['data'], columns = data['feature_names'])
    df['target'] = data['target']
except:
    pd.read_csv("House.csv")
cols = ["MedInc"	,"HouseAge"	,"AveRooms"	,"AveBedrms"	,"Population"	,"AveOccup"]
X = df[cols]
y = df['target']

# ==============================PROJECT START===============================

st.title("California Housing Price Prediction Using Multiple Machine Learning Models")

# Image url
url = "https://imagecdn.99acres.com/media1/35372/12/707452821M-1770031066558.jpg"
st.image(url)

st.write("""Developed a machine learning project to predict California housing prices by comparing the performance of multiple regression algorithms. The project used the California Housing dataset from Scikit-learn and involved data preprocessing, feature selection, model training, evaluation, and best-model selection.""")
st.markdown("""## Key Features

- Loaded and analyzed the California Housing dataset using **Scikit-learn**.
- Selected key numerical features:
  - Median Income
  - House Age
  - Average Rooms
  - Average Bedrooms
  - Population
  - Average Occupancy
- Built and trained multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors (KNN) Regressor
- Compared model performance using:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- Selected the best-performing model based on prediction accuracy.
- Used the best model for California housing price prediction.
- Visualized and analyzed model performance for comparison and selection.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn""")


# =======================Show dataframe==================
st.write(df.sample(5))

# ====================Left Side bar====================
st.sidebar.title("Select House features🏠")
st.sidebar.image(url)

user_input = []

for i,j in enumerate(cols):
    min_col = X[j].min()
    max_col = X[j].max()

    ans = float(st.sidebar.slider(f"Select {j} value: ", min_value = min_col, max_value = max_col))
    user_input.append(ans)


st.markdown("""---""")
st.sidebar.markdown("""👨‍💻 About""")
st.sidebar.write("""Hi, I'm a passionate Data Science and Machine Learning enthusiast who enjoys building intelligent, data-driven applications. I love exploring datasets, developing predictive models, and deploying interactive web apps using Streamlit. My interests include Machine Learning, Data Analytics, Python, and AI, with a focus on creating practical solutions to real-world problems.""")


st.sidebar.markdown("""
---
<div style="text-align:center; padding:10px;">

<a href="https://www.linkedin.com/in/ankitmishra97/" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://static.licdn.com/aero-v1/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="20" alt="LinkedIn"/>
</a>

<a href="https://github.com/axisgras-hash" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://github.com/favicon.ico" width="20" alt="GitHub"/>
</a>

<a href="https://www.instagram.com/your-instagram/" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://cdn.simpleicons.org/instagram/E4405F" width="20" alt="Instagram"/>
</a>

<a href="https://your-app.streamlit.app/" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://cdn.simpleicons.org/streamlit/FF4B4B" width="20" alt="Streamlit"/>
</a>

<a href="https://www.youtube.com/@your-channel" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://cdn.simpleicons.org/youtube/FF0000" width="20" alt="YouTube"/>
</a>

<br><br>

<span style="color:gray; font-size:14px;">
Made with ❤️ using Streamlit
</span>

</div>
""", unsafe_allow_html=True)







st.markdown("""---""")
st.markdown("""
### Selected House features by user: """)

temp_df = pd.DataFrame([user_input], columns = cols)
st.write(temp_df)

# ==========Pickle Model Load===============

with st.spinner("Model Loading.."):
    try:
        with open("house_price_gpt.pkl", 'rb') as f:
            model = pickle.load(f)
    except:
        import gdown
        import joblib
        import os
        MODEL_PATH = "house_price_gpt.pkl"
        FILE_ID = "1wydu6X1fn4Esmi5VtC9mJ3QvEovAiLof"
        @st.cache_resource
        def load_model():
            if not os.path.exists(MODEL_PATH):
                gdown.download(
                    id=FILE_ID,
                    output=MODEL_PATH,
                    quiet=False
                )
        
            model = joblib.load(MODEL_PATH)
            return model
        
        model = load_model()
    

    time.sleep(1)

    st.info("Model Loaded Successfully!!")

    
st.markdown("""---""")

if st.button("Predict House Price: "):
    with st.spinner("Predicting.."):
        ans = model.predict(temp_df)[0]
        time.sleep(2)

    img_no = str(random.randint(1,4))
    for i in ['jpg','jpeg','png']:
        try:
            img =  img_no + '.' +i
            st.image(img)
        except Exception as e:
            # st.write(e)
            pass
    final_ans = f"Predicted House Price is $ {round(ans*100000)}"
    st.success(final_ans)



# ====================ST FOOTER==================

st.markdown("""
---
<div style="text-align:center; padding:10px;">

<a href="https://www.linkedin.com/in/ankitmishra97/" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://static.licdn.com/aero-v1/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="30" alt="LinkedIn"/>
</a>

<a href="https://github.com/axisgras-hash" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://github.com/favicon.ico" width="30" alt="GitHub"/>
</a>

<a href="https://www.instagram.com/your-instagram/" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://cdn.simpleicons.org/instagram/E4405F" width="30" alt="Instagram"/>
</a>

<a href="https://your-app.streamlit.app/" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://cdn.simpleicons.org/streamlit/FF4B4B" width="30" alt="Streamlit"/>
</a>

<a href="https://www.youtube.com/@your-channel" target="_blank" style="text-decoration:none; margin:0 12px;">
    <img src="https://cdn.simpleicons.org/youtube/FF0000" width="30" alt="YouTube"/>
</a>

<br><br>

<span style="color:gray; font-size:14px;">
Made with ❤️ using Streamlit
</span>

</div>
""", unsafe_allow_html=True)


