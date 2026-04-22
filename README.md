---
title: Energy Forecasting App
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---
# ⚡ Smart Energy Consumption Forecasting Dashboard

A professional, end-to-end machine learning application built with **Streamlit** to predict appliance energy consumption in residential buildings. This project leverages the **UCI Appliances Energy Prediction Dataset** to drive smarter, data-backed building management.

## 🚀 Key Features
- **Interactive EDA**: 5 dynamic visualisation modules (Distribution, Time Series, Periods, Scatter, Heatmap).
- **Multi-Model Engine**: Compare 5 Regression architectures (Linear, Ridge, Lasso, Random Forest, Gradient Boosting).
- **Live Prediction Form**: Input real-time environmental data to receive an instant energy forecast.
- **XAI (Explainable AI)**: Deep-dive into model decision-making using **SHAP** values.
- **MLOps Integration**: Track hyperparameter tuning experiments live on **Weights & Biases**.

## 🛠️ Technology Stack
- **Framework**: Streamlit
- **Processing**: NumPy, Pandas
- **Modeling**: Scikit-Learn
- **Explainability**: SHAP
- **Tracking**: Weights & Biases (W&B)
- **Visualisation**: Plotly, Seaborn, Matplotlib

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ds_final
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets:**
   Create a `.env` file in the root directory (already gitignored):
   ```env
   WANDB_API_KEY=your_key_here
   WANDB_PROJECT=energy-forecasting
   ```

4. **Launch the App:**
   ```bash
   python3 -m streamlit run app.py
   ```

## 🌐 Deployment
- **HuggingFace Space**: [Link to your space]
- **Streamlit Cloud**: [Link to your app]

## 👥 Team
- **Hassan Raza** & Team

---
*Developed for the Data Science Final Project - April 2026*
