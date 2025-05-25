# 🧮 Linear Regression Health Costs Calculator
A TensorFlow/Keras-based neural network model that predicts individual medical expenses based on demographic and lifestyle features. This project demonstrates a complete machine learning workflow, from data preparation and model building to evaluation and visualization.

## 📊 Solution Overview
This project solves the challenge of predicting healthcare costs using a feedforward neural network regression model. The solution involved:
- Data loading and preprocessing
- Exploratory data analysis and encoding
- Neural network construction and training
- Model evaluation and prediction visualization

## 📁 File Structure
```bash
📦 training_models/
├── p_h_c_w_lr.ipynb   # The main Jupyter notebook
└── README.md          # Project description
```

## 📌 Note
All code is implemented inside one Jupyter notebook. You can run each section independently to train and test the models.

## 🧹 Data Preparation
- Data Loading: Dataset loaded into a Pandas DataFrame.
- Exploratory Data Analysis: Used histograms and pairplots to visualize distributions.
- Categorical Encoding: One-hot encoding for `sex`, `smoker`, and `region` using pd.get_dummies.
- Train/Test Split: 80/20 random sampling.
- Feature/Label Separation: `expenses` column popped from the dataset and stored as labels.

## 🧠 Model Building and Training
- Normalization: A normalization layer was adapted to the training features.
- Model Architecture:
  - Input normalization layer
  - Two Dense layers with ReLU activation
  - One output Dense layer for regression (no activation)
- Compilation: Optimizer: Adam, Loss: Mean Absolute Error (MAE)
- Training: Trained for 200 epochs with a validation split of 20%

## ✅ Evaluation
- Performance Metric: Mean Absolute Error (MAE)
- Target: Ensure MAE is less than $3,500
- Visualization: Scatter plot of predicted vs actual expenses for visual performance check

## 🧮 Data Structures & Algorithms
| Data Structure / Algorithm | Description                                                                |
|----------------------------|----------------------------------------------------------------------------|
| `Pandas DataFrame`         | Used for loading, manipulating, encoding, and splitting the dataset.       |
| `NumPy Array`              | Used for efficient numerical operations and as input format to the model.  |
| `TensorFlow Tensor`        | Used internally by TensorFlow for model training and inference.            |
| Neural Network (Regression)| The main algorithm used to predict continuous healthcare cost values.      |

## 📌 Summary
This project demonstrates how a feedforward neural network can effectively predict medical expenses based on structured data. By preprocessing the dataset, encoding categorical features, normalizing inputs, and carefully training a regression model, we achieved accurate predictions within an acceptable error range.

## 🏷️ Project Origin
This project is a solution to a [freeCodeCamp](https://www.freecodecamp.org/) deep learning project challenge.

## 📎 License
This project is licensed under the [MIT License](https://mit-license.org/).