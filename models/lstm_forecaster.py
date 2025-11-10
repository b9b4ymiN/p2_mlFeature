"""
LSTM Time-Series Forecaster

For forecasting OI and Price using sequential patterns
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class LSTMForecaster:
    """
    LSTM model for time-series forecasting of OI and Price
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, lookback=50):
        """
        Initialize LSTM Forecaster

        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Number of outputs
            lookback: Number of past timesteps to use
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lookback = lookback
        self.model = None
        self.device = None

    def _build_model(self):
        """Build LSTM model"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")

        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(LSTMModel, self).__init__()

                self.hidden_dim = hidden_dim
                self.num_layers = num_layers

                # LSTM layers
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0
                )

                # Fully connected layers
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, output_dim)
                )

            def forward(self, x):
                # x shape: (batch, seq_len, input_dim)
                lstm_out, _ = self.lstm(x)

                # Take last timestep output
                last_output = lstm_out[:, -1, :]

                # Pass through FC layers
                prediction = self.fc(last_output)

                return prediction

        return LSTMModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)

    def create_sequences(self, data, target):
        """
        Create sequences for LSTM

        For each sample, take previous 'lookback' timesteps as input
        """
        X, y = [], []

        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """Train LSTM model"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")

        print("Training LSTM Forecaster...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

        # Create sequences
        print(f"Creating sequences with lookback={self.lookback}...")
        X_train_seq, y_train_seq = self.create_sequences(
            X_train.values, y_train.values
        )
        X_val_seq, y_val_seq = self.create_sequences(
            X_val.values, y_val.values
        )

        print(f"Training sequences: {len(X_train_seq)}, Val sequences: {len(X_val_seq)}")

        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq).unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for time-series!
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 15

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
                print(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))

        print(f"✓ LSTM Forecaster trained")

        return self.model

    def predict(self, X):
        """Predict future values"""
        import torch

        # Create sequences
        X_seq, _ = self.create_sequences(X.values, np.zeros(len(X)))

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        # Pad predictions to match original length
        predictions = np.concatenate([np.full(self.lookback, np.nan), predictions])

        return predictions

    def evaluate(self, X_test, y_test):
        """Evaluate LSTM model"""
        y_pred = self.predict(X_test)

        # Remove NaN values
        mask = ~np.isnan(y_pred)
        y_pred_valid = y_pred[mask]
        y_test_valid = y_test.values[mask]

        mse = mean_squared_error(y_test_valid, y_pred_valid)
        mae = mean_absolute_error(y_test_valid, y_pred_valid)

        # Directional accuracy
        direction_correct = (np.sign(y_pred_valid) == np.sign(y_test_valid)).mean()

        print("\n" + "="*60)
        print("LSTM Forecaster - Evaluation Results")
        print("="*60)
        print(f"MSE:                     {mse:.6f}")
        print(f"RMSE:                    {np.sqrt(mse):.6f}")
        print(f"MAE:                     {mae:.6f}")
        print(f"Directional Accuracy:    {direction_correct:.4f}")
        print("="*60)

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'directional_accuracy': direction_correct
        }
