# 🏀 NBA Player Performance Predictor

A machine learning project that predicts **NBA player stats** (e.g., points, rebounds, assists) using historical and contextual game data.  
Built with **PyTorch**, **FastAPI**, and **XGBoost**, this project explores data-driven approaches to model player performance across seasons.

---

## 🚀 Project Overview

This project trains both **a simple linear neural network** and an **XGBoost regression model** to predict player performance metrics based on:
- Recent game statistics
- Rolling averages
- Minutes played
- Team and opponent context

The goal is to create a lightweight and explainable prediction system that can later be deployed as a **REST API** using FastAPI.

---

## 🧠 Model Architecture

### 1. Linear Neural Network
A simple feedforward model trained in PyTorch:
- Input layer → Hidden layers → Output (e.g., points)
- Loss: Mean Absolute Error (MAE)
- Optimizer: Adam
- Activation: ReLU

Achieved consistent test MAE around **4.6 points**, showing solid generalization for a small, interpretable network.

### 2. XGBoost Regressor
Gradient boosting tree model trained for feature importance insights and ensemble comparison.

---

