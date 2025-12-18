import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="KNN Social Network Ads", layout="wide")
st.title("Social Network Ads – KNN Classifier (Enhanced App)")

# Sidebar controls
st.sidebar.header("Model Settings")

# 1. Upload data
uploaded = st.sidebar.file_uploader(
    "Upload Social_Network_Ads.csv", type=["csv"]
)

if uploaded is not None:
    data = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # 2. Select features & target
    st.subheader("Feature Selection")
    features = st.multiselect(
        "Select feature columns",
        options=data.columns.tolist(),
        default=["Age", "EstimatedSalary"],
    )
    target = st.selectbox("Select target column", data.columns, index=data.columns.get_loc("Purchased"))

    if len(features) != 2:
        st.warning("Please select exactly TWO features for visualization.")
        st.stop()

    X = data[features]
    y = data[target]

    # 3. Train / test split
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.25, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. KNN model
    k = st.sidebar.slider("Number of neighbors (k)", 1, 30, 5)
    metric = st.sidebar.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan"])

    model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    model.fit(X_train_scaled, y_train)

    # 6. Accuracy & evaluation
    acc = model.score(X_test_scaled, y_test)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy", f"{acc:.3f}")

    with col2:
        st.write("Confusion Matrix")
        cm = confusion_matrix(y_test, model.predict(X_test_scaled))
        st.write(cm)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, model.predict(X_test_scaled)))

    # 7. User input prediction
    st.subheader("Predict for a New User")

    input_vals = []
    for col in features:
        val = st.number_input(
            f"{col}",
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean()),
        )
        input_vals.append(val)

    if st.button("Predict"):
        new_point = scaler.transform([input_vals])
        pred = model.predict(new_point)[0]
        st.success(f"Prediction: {pred}")

    # 8. Visualization
    st.subheader("Decision Boundary Visualization")

    show_plot = st.checkbox("Show decision boundary", value=True)
    if show_plot:
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 5000, X.iloc[:, 1].max() + 5000

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300),
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid)
        Z = model.predict(grid_scaled).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

        for cls in np.unique(y):
            ax.scatter(
                X.iloc[:, 0][y == cls],
                X.iloc[:, 1][y == cls],
                label=f"Class {cls}",
                s=20,
            )

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("KNN Decision Boundary")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

else:
    st.info("Upload the CSV file from the sidebar to start.")
