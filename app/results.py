import csv
import os
import time
from datetime import datetime
from pen import QuantumClassifier
from pen_hilbert import QuantumHilbertClassifier

# =====================
# CONFIGURATION CONSTANTS
# =====================
MODE = 'quantum'  # Options: 'quantum', 'quantum_hilbert', 'both'
EPOCHS = 20
LEARNING_RATE = 0.01
FEATURES = 10
LAYERS = 40
BATCH_SIZE = 32
SEED = 42
USE_HILBERT = True  # Only relevant for quantum

# =====================
# EXPERIMENT RUNNER CLASSES
# =====================
class ExperimentRunner:
    def __init__(self, epochs, lr, features, layers, batch_size, seed):
        self.epochs = epochs
        self.lr = lr
        self.features = features
        self.layers = layers
        self.batch_size = batch_size
        self.seed = seed

    def run_and_log(self, results, csv_file):
        duration = results.get('execution_time', None)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = [
            'date', 'execution_time', 'epochs', 'learning_rate', 'features', 'layers', 'batch_size',
            'loss', 'accuracy', 'recall', 'f1_score'
        ]
        row = [
            date, f'{duration:.2f}', results['epochs'], results['learning_rate'], results['features'],
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

class QuantumRunner(ExperimentRunner):
    def run(self):
        print("\n--- Running QUANTUM QuantumClassifier ---")
        qc = QuantumClassifier(
            n_qubits=self.features,
            pca_features=self.features,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            layers=self.layers,
            seed=self.seed
        )
        start_time = time.time()
        results = qc.train_and_evaluate()
        end_time = time.time()
        self.duration = end_time - start_time
        results['execution_time'] = self.duration
        self.run_and_log(results, 'results_log.csv')

class QuantumHilbertRunner(ExperimentRunner):
    def __init__(self, epochs, lr, features, layers, batch_size, seed, use_hilbert=True):
        super().__init__(epochs, lr, features, layers, batch_size, seed)
        self.use_hilbert = use_hilbert
    def run(self):
        print("\n--- Running QUANTUM QuantumHilbertClassifier ---")
        qhc = QuantumHilbertClassifier(
            n_qubits=self.features,
            pca_features=self.features,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            layers=self.layers,
            seed=self.seed,
            use_hilbert=self.use_hilbert
        )
        start_time = time.time()
        results = qhc.train_and_evaluate()
        end_time = time.time()
        self.duration = end_time - start_time
        results['execution_time'] = self.duration
        csv_file = 'results_hilbert_log.csv' if self.use_hilbert else 'results_log.csv'
        self.run_and_log(results, csv_file)

# =====================
# MAIN EXECUTION
# =====================
def main():
    if MODE == 'quantum':
        QuantumRunner(EPOCHS, LEARNING_RATE, FEATURES, LAYERS, BATCH_SIZE, SEED).run()
    elif MODE == 'quantum_hilbert':
        QuantumHilbertRunner(EPOCHS, LEARNING_RATE, FEATURES, LAYERS, BATCH_SIZE, SEED, USE_HILBERT).run()
    elif MODE == 'both':
        QuantumRunner(EPOCHS, LEARNING_RATE, FEATURES, LAYERS, BATCH_SIZE, SEED).run()
        QuantumHilbertRunner(EPOCHS, LEARNING_RATE, FEATURES, LAYERS, BATCH_SIZE, SEED, USE_HILBERT).run()
    else:
        raise ValueError("Invalid MODE. Choose from 'classic', 'quantum', or 'both'.")

if __name__ == "__main__":
    main() 