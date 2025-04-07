# run_simulation.py
import subprocess
import sys
import threading
import time
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Flower Simulation for Brain Tumor Classification')
parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
parser.add_argument('--num_rounds', type=int, default=5, help='Number of federated learning rounds')
parser.add_argument('--zip_path', type=str, required=True, help='Path to dataset zip file')
parser.add_argument('--data_dir', type=str, default='brain_tumor_data', help='Directory to extract dataset')
parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

def run_server():
    print("Starting Flower server...")
    subprocess.run([
        sys.executable, "server.py",
        "--num_rounds", str(args.num_rounds),
        "--min_clients", str(args.num_clients),
        "--output_dir", args.output_dir
    ])

def run_client(client_id):
    print(f"Starting Flower client {client_id}...")
    subprocess.run([
        sys.executable, "brain_tumor_flower.py",
        "--client_id", str(client_id),
        "--num_clients", str(args.num_clients),
        "--zip_path", args.zip_path,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir
    ])

def main():
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(5)
    
    # Start clients
    client_threads = []
    for client_id in range(args.num_clients):
        client_thread = threading.Thread(target=run_client, args=(client_id,))
        client_threads.append(client_thread)
        client_thread.start()
        time.sleep(1)  # Stagger client starts
    
    # Wait for clients to finish
    for client_thread in client_threads:
        client_thread.join()
    
    # Wait for server to finish
    server_thread.join()
    
    print("Federated learning complete!")

if __name__ == "__main__":
    main()