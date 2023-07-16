import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight

%matplotlib inline
wine_data = pd.read_csv(r'C:/Users/ravik/Downloads/Wine-Quality-Dataset.csv')
wine_data.head(10)
wine_data = pd.read_csv(r'C:/Users/ravik/Downloads/Wine-Quality-Dataset.csv')
from sklearn.preprocessing import MinMaxScaler

df = wine_data.copy()
df["quality"] = df["quality"] - 3

# Perform feature scaling on the dataset
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Create a new DataFrame with scaled features
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Display the modified DataFrame
df_scaled.head()
df.info()
df.isnull().sum()
# Perform feature scaling on the dataset
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Create a new DataFrame with scaled features
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Display the modified DataFrame with a gradient background
plt.figure(figsize=(10, 6))
sns.heatmap(df_scaled.describe()[1:], cmap="viridis", annot=True, fmt=".2f", linewidths=0.5)
plt.title("Descriptive Statistics of Scaled Wine Data")
plt.show()
df.describe()[1:].style.background_gradient()
#unique values of the data
df.nunique().sort_values(ascending=True)
wine_data.hist(figsize=(20,15))
plt.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

float_cols = df.select_dtypes(include=["float64"]).columns.tolist()
float_cols = [col for col in float_cols if col != "quality"]

subplot_titles = [f"Distribution - {col}" for col in float_cols] + [f"Boxplot - {col}" for col in float_cols]
fig = make_subplots(rows=len(float_cols), cols=2, subplot_titles=subplot_titles, shared_yaxes=True)

for index, col in enumerate(float_cols):
    row = index + 1
    fig.add_trace(go.Histogram(x=df[col], nbinsx=30, name="Distribution"), row=row, col=1)

for index, col in enumerate(float_cols):
    row = index + 1
    fig.add_trace(go.Box(x=df[col], name="Boxplot"), row=row, col=2)

# Update layout
fig.update_layout(
    height=1600,
    width=900,
    title="Visualizing Columns",
    title_font=dict(size=40),
    showlegend=False,
)

# Update subplot titles
for i in range(1, len(float_cols)+1):
    fig.update_xaxes(title_text="Values", row=i, col=1)
    fig.update_xaxes(title_text="Values", row=i, col=2)
    fig.update_yaxes(title_text="Frequency", row=i, col=1)
    fig.update_yaxes(title_text="Frequency", row=i, col=2)

fig.show()
#Stripplot of chlorides values at diffrent quality ratings
plt.figure(figsize=(16,6))
sns.stripplot(y='chlorides',x='quality',data=wine_data)
plt.title('Strip plot of chlorides feature at different quality ratings ',fontsize=16,c='k')
plt.xlabel('quality',fontsize=16,c='k')
plt.ylabel('chlorides',fontsize=16,c='k')
plt.show()
sns.countplot(data=df, x="quality")
fig = px.imshow(df.corr(), color_continuous_scale="Greys")
fig.update_layout(height=600)
fig.show()
fig = go.Figure()

for x in range(6):
    fig.add_trace(
        go.Box(
            x=df[df.quality == x]["volatile acidity"],
            y=df[df.quality == x].quality,
            name="Quality " + str(x),
        )
    )

fig.update_layout(yaxis_title="quality", xaxis_title="volatile acidity")
fig.update_traces(orientation="h")
fig.show()
fig = px.scatter(
    df,
    x="total sulfur dioxide",
    y="free sulfur dioxide",
    color=df.quality,
    color_continuous_scale="Blues",
)
fig.update_layout(legend_title_text="Quality")
fig = go.Figure()

for x in range(6):
    fig.add_trace(
        go.Box(
            x=df[df.quality == x]["citric acid"],
            y=df[df.quality == x].quality,
            name="Quality " + str(x),
        )
    )

fig.update_layout(yaxis_title="quality", xaxis_title="citric acid")
fig.update_traces(orientation="h")
fig.show()
fig = px.scatter(
    df,
    x="fixed acidity",
    y="density",
    color=df.quality,
    color_continuous_scale="Blues",
)
fig.update_layout(legend_title_text="Quality")
fig = go.Figure()

for x in range(6):
    fig.add_trace(
        go.Box(
            x=df[df.quality == x].sulphates,
            y=df[df.quality == x].quality,
            name="Quality " + str(x),
        )
    )

fig.update_layout(yaxis_title="quality", xaxis_title="sulphates")
fig.update_traces(orientation="h")
fig.show()
fig = px.scatter(
    df,
    x="citric acid",
    y="volatile acidity",
    color=df.quality,
    color_continuous_scale="Blues",
)
fig.update_layout(legend_title_text="Quality")
import plotly.graph_objects as go
import plotly.io as pio
import random

X = df.drop(["quality"], axis=1)
y = df["quality"]
fs = SelectKBest(score_func=f_classif, k="all")
fs.fit(X, y)

feature_contribution = (fs.scores_ / sum(fs.scores_)) * 100
feature_names = X.columns

# Generate random colors for each bar
colors = [f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})" for _ in range(len(feature_names))]

# Create a bar plot with customized colors
fig = go.Figure(data=go.Bar(x=feature_names, y=fs.scores_, marker_color=colors))

# Set the title and axis labels
fig.update_layout(
    title="Feature Contribution to Quality",
    xaxis_title="Features",
    yaxis_title="Score",
    title_font=dict(size=24),
    xaxis=dict(title_font=dict(size=14)),
    yaxis=dict(title_font=dict(size=14))
)
for i, j in enumerate(X.columns):
    print(f"{j} : {feature_contribution[i]:.2f}%")

# Enable hover information
fig.update_traces(hovertemplate="Feature: %{x}<br>Score: %{y:.2f}")

# Show the interactive graph
pio.show(fig)
X_fs = X[
    [
        "volatile acidity",
        "citric acid",
        "chlorides",
        "total sulfur dioxide",
        "sulphates",
        "alcohol",
    ]
]
X_train, X_test, y_train, y_test = train_test_split(
    X_fs, y, stratify=y, test_size=0.25, random_state=0
)
oversample = RandomOverSampler(random_state=0)
X_train, y_train = oversample.fit_resample(X_train, y_train)
sns.countplot(data=X_train, x=y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
models = [
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    KNeighborsClassifier(),
    SVC(random_state=0),
    LogisticRegression(random_state=0),
]

models_comparison = {}

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5)
    
    models_comparison[str(model)] = [
        accuracy_score(y_pred, y_test),
        f1_score(y_pred, y_test, average="macro"),
        precision_score(y_pred, y_test, average="macro"),
        recall_score(y_pred, y_test, average="macro"),
        accuracies.mean(),
    ]

for model, scores in models_comparison.items():
    print(f"Model: {model}\n")
    print(classification_report(y_test, y_pred))
    print("-" * 30, "\n")
models_com_df = pd.DataFrame(models_comparison).T
models_com_df.columns = [
    "Model Accuracy",
    "Model F1-Score",
    "Precision",
    "Recall",
    "CV Accuracy",
]
models_com_df = models_com_df.sort_values(by="Model F1-Score", ascending=False)
models_com_df.style.format("{:.2%}").background_gradient(cmap="Blues")
import plotly.express as px

# Create an interactive graph of model comparison
fig = px.bar(models_com_df, x=models_com_df.index, y="Model F1-Score", color="Model F1-Score", title="Model Comparison - F1 Score")
fig.update_layout(xaxis_title="Model", yaxis_title="F1 Score")
fig.show()

# Create an interactive heatmap of correlation matrix
fig = px.imshow(df.corr(), color_continuous_scale="Blues")
fig.update_layout(title="Correlation Matrix", title_font=dict(size=24))
fig.show()

# Create an interactive table of model comparison results
fig = go.Figure(data=[go.Table(
    header=dict(values=list(models_com_df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[models_com_df.index] + [models_com_df[col] for col in models_com_df.columns],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(title="Model Comparison Results", title_font=dict(size=24))
fig.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the model performance
plt.figure(figsize=(10, 6))
sns.barplot(
    y=models_com_df.index,
    x="Model F1-Score",
    data=models_com_df,
    palette="Blues",
    orient="h"
)
plt.title("Comparison of Model F1-Scores")
plt.xlabel("F1-Score")
plt.ylabel("Models")
plt.xlim(0, 40)

# Add value labels to the bars
for i, score in enumerate(models_com_df["Model F1-Score"]):
    plt.text(score + 1, i, f"{score:.2f}%", va="center")

plt.show()

# Select the best model
best_model = models_com_df.iloc[0]
best_model_name = best_model.name
best_model_accuracy = best_model["Model Accuracy"]
best_model_f1_score = best_model["Model F1-Score"]
best_model_precision = best_model["Precision"]
best_model_recall = best_model["Recall"]
best_model_cv_accuracy = best_model["CV Accuracy"]

