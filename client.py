# Modified client.py to record training metrics for each epoch
import flwr as fl
import tensorflow as tf
import numpy as np
from model import create_model
from data_utils import load_client_data
import os
import argparse
import time
from datetime import datetime

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define the results directory
RESULTS_DIR = r"D:\research work\brain_tumor_fl"

class BrainTumorClient(fl.client.NumPyClient):
    def __init__(self, client_id, zip_path=None):
        self.client_id = client_id
        self.model = create_model()
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        # Load data for this client
        try:
            self.x_train, self.y_train = load_client_data(client_id, zip_path)
            print(f"Client {client_id} loaded {len(self.x_train)} training samples")
        except Exception as e:
            print(f"Error loading data for client {client_id}: {e}")
            print("Using a small dummy dataset for testing")
            # Create a small dummy dataset for testing
            self.x_train = np.random.rand(10, 150, 150, 3).astype(np.float32)
            self.y_train = np.random.randint(0, 4, size=10).astype(np.int32)
        
        # Create log file for this client
        self.log_file = os.path.join(RESULTS_DIR, f"client_{client_id}_training_log.txt")
        
        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write(f"Brain Tumor Classification - Client {client_id} Training Log\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of training samples: {len(self.x_train)}\n\n")
            f.write("Round,Epoch,Loss,Accuracy,Timestamp\n")
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set model weights
        self.model.set_weights(parameters)
        
        # Get training config
        batch_size = config.get("batch_size", 32)
        epochs = config.get("local_epochs", 1)
        round_number = config.get("round_number", 0)
        
        # Custom callback to log metrics after each epoch
        class LoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self, client_instance, round_number):
                super().__init__()
                self.client = client_instance
                self.round = round_number
                
            def on_epoch_end(self, epoch, logs=None):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_line = f"{self.round},{epoch+1},{logs['loss']:.6f},{logs['accuracy']:.6f},{timestamp}\n"
                
                with open(self.client.log_file, 'a') as f:
                    f.write(log_line)
                
                print(f"Client {self.client.client_id} - Round {self.round}, Epoch {epoch+1}: loss={logs['loss']:.6f}, accuracy={logs['accuracy']:.6f}")
        
        # Train the model with the logging callback
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[LoggingCallback(self, round_number)]
        )
        
        # Return updated model parameters and results
        return self.model.get_weights(), len(self.x_train), {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1]
        }

    def evaluate(self, parameters, config):
        # Set model weights
        self.model.set_weights(parameters)
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        
        # Log evaluation results
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        round_number = config.get("round_number", 0)
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nEvaluation - Round {round_number}: loss={loss:.6f}, accuracy={accuracy:.6f}, timestamp={timestamp}\n")
        
        # Return metrics
        return loss, len(self.x_train), {"accuracy": accuracy}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (0, 1, 2, etc.)")
    parser.add_argument("--zip-path", type=str, help="Path to the dataset zip file (optional)")
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    
    # Start Flower client
    client = BrainTumorClient(client_id=args.client_id, zip_path=args.zip_path)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()