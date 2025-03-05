import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error

# Streamlit UI Config
st.set_page_config(page_title="Data Viz", layout="wide")
st.title("📊 Data Viz")
st.markdown("Upload your dataset, choose an algorithm, and explore insights!")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of the Dataset")
    st.dataframe(df.head())
    
    # Data Visualization
    st.sidebar.header("Visualization Options")
    chart_type = st.sidebar.selectbox("Choose a chart type", ["Scatter Plot", "Line Chart", "Histogram", "Correlation Matrix"])
    
    if chart_type == "Scatter Plot":
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot of {x_axis} vs {y_axis}")
        st.plotly_chart(fig)
    elif chart_type == "Line Chart":
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        fig = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart of {x_axis} vs {y_axis}")
        st.plotly_chart(fig)
    elif chart_type == "Histogram":
        column = st.sidebar.selectbox("Select a column", df.columns)
        fig = px.histogram(df, x=column, title=f"Histogram of {column}")
        st.plotly_chart(fig)
    elif chart_type == "Correlation Matrix":
        plt.figure(figsize=(10, 5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # Machine Learning Section
    st.sidebar.header("Machine Learning")
    task = st.sidebar.selectbox("Choose a task", ["Classification", "Regression", "Clustering"])
    target_column = st.sidebar.selectbox("Select target column", df.columns)
    features = st.sidebar.multiselect("Select feature columns", df.columns)

    if len(features) > 0 and target_column:
        X = df[features]
        y = df[target_column]

        if task == "Classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### Classification Accuracy: {accuracy:.2f}")
        
        elif task == "Regression":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"### Mean Absolute Error: {mae:.2f}")
        
        elif task == "Clustering":
            n_clusters = st.sidebar.slider("Select number of clusters", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters)
            df["Cluster"] = model.fit_predict(X)
            fig = px.scatter(df, x=features[0], y=features[1], color=df["Cluster"].astype(str), title="Cluster Visualization")
            st.plotly_chart(fig)

st.sidebar.markdown("Developed with ❤️ using Streamlit by @pr1ncegupta")
