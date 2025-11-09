"""
Regression Models for Price Target Prediction

Models:
- XGBoost Regressor
- Neural Network Regressor
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class XGBoostPricePredictor:
    """
    XGBoost regression model for price target prediction
    """

    def __init__(self, params=None):
        self.params = params or self._default_params()
        self.model = None

    def _default_params(self):
        return {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist'
        }

    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost regression model"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")

        print("Training XGBoost Regressor...")

        self.model = xgb.XGBRegressor(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        print(f"✓ XGBoost Regressor trained")

        return self.model

    def predict(self, X):
        """Predict future return"""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate regression performance"""
        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Directional accuracy
        direction_correct = (np.sign(y_pred) == np.sign(y_test)).mean()

        print("\n" + "="*60)
        print("XGBoost Regressor - Evaluation Results")
        print("="*60)
        print(f"MSE:                     {mse:.6f}")
        print(f"RMSE:                    {np.sqrt(mse):.6f}")
        print(f"MAE:                     {mae:.6f}")
        print(f"R² Score:                {r2:.4f}")
        print(f"Directional Accuracy:    {direction_correct:.4f}")
        print("="*60)

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'directional_accuracy': direction_correct
        }


class NeuralNetRegressor:
    """
    Deep Neural Network for price prediction
    """

    def __init__(self, input_dim, hidden_dims=None):
        """
        Initialize Neural Network

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.model = None
        self.device = None

    def _build_model(self):
        """Build PyTorch model"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")

        class NNModel(nn.Module):
            def __init__(self, input_dim, hidden_dims):
                super(NNModel, self).__init__()

                layers = []
                prev_dim = input_dim

                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.3))
                    prev_dim = hidden_dim

                # Output layer
                layers.append(nn.Linear(prev_dim, 1))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        return NNModel(self.input_dim, self.hidden_dims)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
        """Train neural network"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")

        print("Training Neural Network Regressor...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.FloatTensor(y_train.values).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val.values),
            torch.FloatTensor(y_val.values).unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 20

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_nn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_nn_model.pth'))

        print(f"✓ Neural Network Regressor trained")

        return self.model

    def predict(self, X):
        """Predict future returns"""
        import torch

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions

    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        direction_correct = (np.sign(y_pred) == np.sign(y_test)).mean()

        print("\n" + "="*60)
        print("Neural Network Regressor - Evaluation Results")
        print("="*60)
        print(f"MSE:                     {mse:.6f}")
        print(f"RMSE:                    {np.sqrt(mse):.6f}")
        print(f"MAE:                     {mae:.6f}")
        print(f"R² Score:                {r2:.4f}")
        print(f"Directional Accuracy:    {direction_correct:.4f}")
        print("="*60)

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'directional_accuracy': direction_correct
        }
