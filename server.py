# server.py
import flwr as fl
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Flower Server for Brain Tumor Classification')
parser.add_argument('--num_rounds', type=int, default=5, help='Number of federated learning rounds')
parser.add_argument('--min_clients', type=int, default=2, help='Minimum number of clients')
parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=args.min_clients,  # Minimum number of clients to be sampled for training
    min_evaluate_clients=args.min_clients,  # Minimum number of clients to be sampled for evaluation
    min_available_clients=args.min_clients,  # Minimum number of clients that need to be connected to the server
)

# Start server
def main():
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    main()