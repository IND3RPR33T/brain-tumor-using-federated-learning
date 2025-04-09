import flwr as fl
import tensorflow as tf
from typing import Dict, Optional, Tuple, List
import numpy as np
from model import create_model
import os
import argparse

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define strategy
def get_on_fit_config_fn():
    """Return function that returns training configuration."""
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "batch_size": 32,
            "local_epochs": 1,
            "round_number": server_round
        }
        return config
    return fit_config

def get_evaluate_fn(model, zip_path=None):
    """Return an evaluation function for server-side evaluation."""
    # Load test data here to avoid loading it on each client
    from data_utils import load_test_data
    x_test, y_test = load_test_data(zip_path)
    
    # The evaluation function
    def evaluate(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Make predictions for additional metrics
        y_pred = np.argmax(model.predict(x_test), axis=1)
        
        # Calculate additional metrics (simplified here)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    return evaluate

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--zip-path", type=str, help="Path to the dataset zip file (optional)")
    args = parser.parse_args()
    
    # Initialize model
    model = create_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Get the evaluation function
    evaluate_fn = get_evaluate_fn(model, args.zip_path)
    
    # Define strategy
    # For newer versions of Flower (>= 1.0.0)
    try:
        # First try the newer API
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=2,  # Wait until at least 2 clients are available
            on_fit_config=get_on_fit_config_fn(),
            evaluate_fn=evaluate_fn,  # Pass the evaluation function
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        )
    except TypeError:
        # Fall back to older API if needed
        print("Using older Flower API")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            on_fit_config_fn=get_on_fit_config_fn(),
            evaluate_fn=evaluate_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

    # Save the final model after federated learning completes
    print("Federated learning complete. Saving the final model...")
    model_save_path = "brain_tumor_model.h5"
    model.save_weights(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()

