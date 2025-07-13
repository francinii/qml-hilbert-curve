import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.datasets import fetch_openml
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from hilbert_curve import HilbertCurveProcessor
from PIL import Image
from preprocing_data import prepare_data_multiclass
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import PlotUtils

class QuantumHilbertClassifier:
    def __init__(self, n_qubits=8, pca_features=8, batch_size=16, epochs=20, lr=0.01, layers=3, seed=42, use_hilbert=True):
        # Store hyperparameters for the experiment
        self.n_qubits = n_qubits
        self.pca_features = pca_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.layers = layers
        self.seed = seed
        self.use_hilbert = use_hilbert
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Prepare data and build the model
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        X, y = prepare_data_multiclass()
        X = X.reshape((X.shape[0], -1)) / 255.0  # flatten and normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=self.pca_features)
        X_pca = pca.fit_transform(X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=self.seed)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)

    def _build_model(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        def circuit(inputs, weights):
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))
        weight_shapes = {"weights": (self.layers, self.n_qubits, 3)}
        qlayer = qml.qnn.TorchLayer(qml.qnode(dev)(circuit), weight_shapes)
        class HybridModel(nn.Module):
            def __init__(self, qlayer, n_classes=4):
                super().__init__()
                self.qlayer = qlayer
                self.fc = nn.Linear(1, n_classes)
            def forward(self, x):
                out = []
                for i in range(x.shape[0]):
                    out.append(self.qlayer(x[i]))
                x = torch.stack(out)
                x = x.view(-1, 1)
                x = self.fc(x)
                return nn.functional.softmax(x, dim=1)
        self.model = HybridModel(qlayer, n_classes=4)

    def train_and_evaluate(self):
        X_train_t = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_t = torch.tensor(self.y_train, dtype=torch.long)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        loss_history = []
        start = time.time()
        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_history.append(loss.item())
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        end = time.time()
        with torch.no_grad():
            X_test_t = torch.tensor(self.X_test, dtype=torch.float32)
            y_test_t = torch.tensor(self.y_test, dtype=torch.long)
            preds = self.model(X_test_t)
            preds_cls = torch.argmax(preds, dim=1)
            acc = (preds_cls == y_test_t).float().mean().item()
            cm = confusion_matrix(y_test_t.cpu().numpy(), preds_cls.cpu().numpy())
            recall = recall_score(y_test_t.cpu().numpy(), preds_cls.cpu().numpy(), average='macro')
            f1 = f1_score(y_test_t.cpu().numpy(), preds_cls.cpu().numpy(), average='macro')
        # Graficar función de pérdida y matriz de confusión
        class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        PlotUtils.plot_loss(loss_history, save_path='results/graphics/loss_function_hilbert.png')
        PlotUtils.plot_confusion_matrix(cm, class_names=class_names, save_path='results/graphics/confusion_matrix_hilbert.png')
        results = {
            'epochs': self.epochs,
            'learning_rate': self.lr,
            'features': self.pca_features,
            'layers': self.layers,
            'batch_size': self.batch_size,
            'loss': float(loss.item()),
            'accuracy': acc,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'execution_time': end - start
        }
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print(f"Recall: {recall:.4f}")
        print(f"F1 score: {f1:.4f}")
        return results

if __name__ == "__main__":
    import csv
    import os
    from datetime import datetime
    import argparse
    # Parse command-line arguments for experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--features', type=int, default=8)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_hilbert', type=bool, default=True)
    args = parser.parse_args()
    # Instantiate and run the quantum Hilbert classifier
    qhc = QuantumHilbertClassifier(
        n_qubits=args.features,
        pca_features=args.features,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        layers=args.layers,
        seed=args.seed,
        use_hilbert=args.use_hilbert
    )
    results = qhc.train_and_evaluate()
    # Save results to CSV for experiment tracking
    csv_file = 'results_hilbert_log.csv'
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = [
        'date', 'execution_time', 'epochs', 'learning_rate', 'features', 'layers', 'batch_size',
        'loss', 'accuracy', 'recall', 'f1_score'
    ]
    row = [
        date, f"{results['execution_time']:.2f}", results['epochs'], results['learning_rate'], results['features'],
        results['layers'], results['batch_size'], results['loss'], results['accuracy'], results['recall'], results['f1_score']
    ]
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    print('Results saved in', csv_file)
    print('Run summary:')
    print(row)
