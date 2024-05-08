import optuna
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_curve, auc, precision_score, recall_score, confusion_matrix
import time
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class IC50Predictor:
   def __init__(self, data, labels):
       self.data = data
       self.labels = labels
       self.model = None
       self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

   def create_model(self, units1=128, units2=64, dropout=0.2):
       model = Sequential([
           Dense(units1, activation='relu', input_shape=(self.train_data.shape[1],)),
           Dropout(dropout),
           Dense(units2, activation='relu'),
           Dense(1, activation='sigmoid')
       ])
       model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
       return model

   def objective(self, trial):
       params = {
           'units1': trial.suggest_int('units1', 32, 1024),
           'units2': trial.suggest_int('units2', 32, 1024),
           'dropout': trial.suggest_float('dropout', 0.1, 0.5)
       }
       model = self.create_model(**params)
       
       kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
       scores = []
       for train_index, val_index in kf.split(self.train_data, self.train_labels):
           x_train, x_val = self.train_data[train_index], self.train_data[val_index]
           y_train, y_val = self.train_labels[train_index], self.train_labels[val_index]
           model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), verbose=0)
           y_pred = model.predict(x_val)
           y_pred_classes = (y_pred > 0.5).astype(int)
           score = precision_score(y_val, y_pred_classes)
           scores.append(score)
       
       return np.mean(scores)

   def optuna_optimize(self, n_trials=100):
       start_time = time.time()
       study = optuna.create_study(direction='maximize')
       study.optimize(self.objective, n_trials=n_trials)
       optuna_time = time.time() - start_time
       print(f"Optuna best parameters: {study.best_params}")
       print(f"Optuna best score: {study.best_value}")
       print(f"Optuna optimization time: {optuna_time:.2f} seconds")
       return study.best_params

   def grid_search_optimize(self):
       param_grid = {
           'units1': [32, 64, 128, 256, 512, 1024],
           'units2': [32, 64, 128, 256, 512, 1024],
           'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
       }
       model = KerasRegressor(build_fn=self.create_model, verbose=0)
       start_time = time.time()
       grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='precision')
       grid_result = grid_search.fit(self.train_data, self.train_labels)
       grid_search_time = time.time() - start_time
       print(f"Grid search best parameters: {grid_search.best_params_}")
       print(f"Grid search best score: {grid_search.best_score_}")
       print(f"Grid search optimization time: {grid_search_time:.2f} seconds")
       return grid_search.best_params_

   def train_and_evaluate(self, best_params):
       self.model = self.create_model(**best_params)
       history = self.model.fit(self.train_data, self.train_labels, epochs=100, batch_size=32, validation_split=0.1)

       # Plot training & validation loss values
       plt.plot(history.history['loss'])
       plt.plot(history.history['val_loss'])
       plt.title('Model loss')
       plt.ylabel('Loss')
       plt.xlabel('Epoch')
       plt.legend(['Train', 'Validation'], loc='upper left')
       plt.show()

       # Evaluate the model on the test data
       y_pred_prob = self.model.predict(self.test_data)
       y_pred_classes = (y_pred_prob > 0.5).astype(int)
       
       # ROC AUC
       fpr, tpr, _ = roc_curve(self.test_labels, y_pred_prob)
       roc_auc = auc(fpr, tpr)
       plt.figure()
       plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
       plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('Receiver Operating Characteristic')
       plt.legend(loc="lower right")
       plt.show()

       # Precision, Recall, Confusion Matrix
       precision = precision_score(self.test_labels, y_pred_classes)
       recall = recall_score(self.test_labels, y_pred_classes)
       cm = confusion_matrix(self.test_labels, y_pred_classes)

       print(f"Precision: {precision:.2f}")
       print(f"Recall: {recall:.2f}")
       print("Confusion Matrix:")
       print(cm)

       # Chaos Plot
       plt.figure()
       plt.scatter(self.test_labels, y_pred_prob, alpha=0.5)
       plt.xlabel('Actual Labels')
       plt.ylabel('Predicted Probabilities')
       plt.title('Chaos Plot')
       plt.show()

       # Save the model
       self.model.save("learning_data_01.h5")

   def predict(self, input_data):
       if self.model is None:
           raise ValueError("Model is not trained. Please train the model first.")
       
       return self.model.predict(input_data)

# Usage example
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data for binary classification
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# Create an instance of IC50Predictor
predictor = IC50Predictor(X, y)

# Hyperparameter optimization using Optuna
best_params_optuna = predictor.optuna_optimize(n_trials=100)

# Hyperparameter optimization using Grid Search
best_params_grid = predictor.grid_search_optimize()

# Train and evaluate the model using the best parameters from Optuna
predictor.train_and_evaluate(best_params_optuna)

# Predict IC50 for new compounds
new_data, _ = make_classification(n_samples=100, n_classes=2, n_features=20, n_informative=10, n_redundant=5, random_state=42)
predicted_ic50 = predictor.predict(new_data)
print("Predicted IC50 for new compounds:")
print(predicted_ic50)
