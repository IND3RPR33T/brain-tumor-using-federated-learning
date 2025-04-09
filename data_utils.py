import numpy as np
import os
import cv2
import zipfile
import shutil
from sklearn.model_selection import train_test_split
import tempfile

# Constants
data_dir = "brain_tumor_data"
# Update these based on your actual dataset structure
classes = ["glioma", "meningioma", "pituitary", "notumor"]
# Add the subdirectories that contain the class folders
subdirs = ["Training", "Testing"]

# Directory for saving numpy arrays
save_dir = "data_arrays"

def get_zip_path():
    """Prompt the user for the zip file path if not already provided."""
    zip_path = input("Enter the path to your dataset zip file: ")
    if not os.path.exists(zip_path):
        print(f"Error: File '{zip_path}' not found.")
        return get_zip_path()  # Recursively ask until a valid path is provided
    return zip_path

def is_dataset_valid():
    """Check if the dataset directory exists and contains the expected structure."""
    if not os.path.exists(data_dir):
        return False
    
    # Check if the subdirectories exist
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        # Check if at least one of the expected class directories exists in this subdir
        for class_name in classes:
            class_dir = os.path.join(subdir_path, class_name)
            if os.path.exists(class_dir) and os.path.isdir(class_dir):
                # Check if directory contains at least one image
                files = os.listdir(class_dir)
                if len(files) > 0:
                    return True
    
    return False

def extract_dataset(zip_path=None, force_extract=False):
    """Extract the dataset from zip file if not already extracted.
    
    Args:
        zip_path (str, optional): Path to the zip file. If None, user will be prompted.
        force_extract (bool): If True, re-extract even if directory exists.
    """
    # Check if we need to extract
    if force_extract and os.path.exists(data_dir):
        print(f"Removing existing dataset directory '{data_dir}'...")
        shutil.rmtree(data_dir)
    
    if not os.path.exists(data_dir) or force_extract:
        if zip_path is None:
            zip_path = get_zip_path()
            
        print(f"Extracting dataset from '{zip_path}' to '{data_dir}'...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete!")
            
            # Check if extraction created the expected structure
            if not is_dataset_valid():
                print("Warning: Extracted dataset doesn't have the expected structure.")
                print("The zip file might have a different internal structure.")
                
                # Try to find the actual structure
                if os.path.exists(data_dir):
                    print("Searching for image directories...")
                    detect_and_print_structure(data_dir)
                
        except zipfile.BadZipFile:
            print(f"Error: '{zip_path}' is not a valid zip file.")
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)  # Clean up the created directory
            return extract_dataset(None)  # Try again with a new path
        except Exception as e:
            print(f"Error extracting zip file: {e}")
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)  # Clean up the created directory
            return extract_dataset(None)  # Try again with a new path
    else:
        # Directory exists, check if it's valid
        if not is_dataset_valid():
            print(f"Dataset directory '{data_dir}' exists but doesn't contain the expected structure.")
            print("Detecting actual dataset structure...")
            detect_and_print_structure(data_dir)
            
            print("Would you like to re-extract the dataset? (y/n)")
            response = input().strip().lower()
            if response == 'y':
                return extract_dataset(zip_path, force_extract=True)
            else:
                print("Continuing with the existing directory structure.")
        else:
            print(f"Dataset directory '{data_dir}' already exists and contains valid data.")

def detect_and_print_structure(directory, level=0, max_depth=3):
    """Recursively detect and print the directory structure."""
    if level > max_depth:
        return
    
    items = os.listdir(directory)
    
    # Count image files at this level
    image_count = sum(1 for item in items if item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')))
    
    if image_count > 0:
        print(f"{'  ' * level}Directory: {os.path.basename(directory)} - Contains {image_count} images")
    
    # Process subdirectories
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            detect_and_print_structure(item_path, level + 1, max_depth)

def load_data(zip_path=None):
    """Load all data from the dataset with nested structure.
    
    Args:
        zip_path (str, optional): Path to the zip file. If None and extraction is needed,
                                 user will be prompted.
    """
    # Extract dataset if needed
    extract_dataset(zip_path)
    
    # Check if dataset is valid after extraction
    if not is_dataset_valid():
        print("Error: Dataset is not valid. Cannot load data.")
        print("Please provide a valid dataset zip file or fix the directory structure.")
        return load_data(get_zip_path())
    
    images, labels = [], []
    
    # Load from both Training and Testing subdirectories
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Warning: Subdirectory {subdir_path} does not exist!")
            continue
            
        print(f"Loading data from {subdir} directory...")
        
        for label, category in enumerate(classes):
            class_path = os.path.join(subdir_path, category)
            print(f"  Loading {category} images from {class_path}")
            
            if not os.path.exists(class_path):
                print(f"  Warning: Path {class_path} does not exist!")
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if not os.path.isfile(img_path):
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Warning: Could not read image {img_path}")
                    continue
                img = cv2.resize(img, (150, 150))  # Resize to standard size
                # Convert BGR to RGB (OpenCV loads as BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
    
    if len(images) == 0:
        print("Error: No images were loaded. Please check the dataset structure.")
        print("Would you like to try with a different zip file? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            return load_data(get_zip_path())
        else:
            raise ValueError("Cannot proceed without valid image data.")
    
    print(f"Successfully loaded {len(images)} images.")
    return np.array(images), np.array(labels)

def preprocess_data(images):
    """Normalize pixel values to [0, 1]."""
    return images.astype(np.float32) / 255.0

def ensure_save_dir():
    """Ensure the directory for saving numpy arrays exists."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def safe_save(array, filename):
    """Safely save a numpy array to disk with error handling."""
    directory = ensure_save_dir()
    filepath = os.path.join(directory, filename)
    
    try:
        # First try to save directly
        np.save(filepath, array)
        print(f"Successfully saved {filepath}")
    except (OSError, IOError) as e:
        print(f"Error saving {filepath}: {e}")
        print("Trying alternative saving method...")
        
        try:
            # Try saving in smaller chunks
            with open(filepath, 'wb') as f:
                np.save(f, array)
            print(f"Successfully saved {filepath} using file handle")
        except (OSError, IOError) as e:
            print(f"Error with alternative saving method: {e}")
            print("Trying to save with reduced precision...")
            
            try:
                # Try saving with reduced precision if it's a float array
                if array.dtype.kind == 'f':
                    reduced_array = array.astype(np.float16)
                    np.save(filepath, reduced_array)
                    print(f"Successfully saved {filepath} with reduced precision")
                else:
                    print("Cannot reduce precision for non-float array")
                    raise e
            except (OSError, IOError) as e:
                print(f"All saving methods failed: {e}")
                print("Please check disk space and permissions")
                raise

def split_data_for_clients(num_clients=3, zip_path=None):
    """Split the data for multiple clients.
    
    Args:
        num_clients (int): Number of clients to split data for.
        zip_path (str, optional): Path to the zip file. If None and extraction is needed,
                                 user will be prompted.
    """
    # Load and preprocess all data
    X, y = load_data(zip_path)
    X = preprocess_data(X)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save test data for central evaluation
    safe_save(X_test, "X_test.npy")
    safe_save(y_test, "y_test.npy")
    
    # Split training data for clients
    client_data = []
    data_per_client = len(X_train) // num_clients
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else len(X_train)
        
        client_X = X_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        
        # Save client data
        safe_save(client_X, f"client_{i}_X.npy")
        safe_save(client_y, f"client_{i}_y.npy")
        
        client_data.append((client_X, client_y))
        print(f"Client {i}: {len(client_X)} samples")
    
    return client_data

def load_client_data(client_id, zip_path=None):
    """Load data for a specific client.
    
    Args:
        client_id (int): ID of the client to load data for.
        zip_path (str, optional): Path to the zip file. If None and data splitting is needed,
                                 user will be prompted.
    """
    # Check if client data files exist
    client_x_path = os.path.join(ensure_save_dir(), f"client_{client_id}_X.npy")
    client_y_path = os.path.join(ensure_save_dir(), f"client_{client_id}_y.npy")
    
    if not os.path.exists(client_x_path) or not os.path.exists(client_y_path):
        print(f"Client {client_id} data not found. Splitting data for clients...")
        split_data_for_clients(zip_path=zip_path)
    
    # Load client data
    try:
        X = np.load(client_x_path)
        y = np.load(client_y_path)
        print(f"Successfully loaded data for client {client_id}: {len(X)} samples")
        return X, y
    except (OSError, IOError) as e:
        print(f"Error loading client data: {e}")
        print("Re-generating client data...")
        split_data_for_clients(zip_path=zip_path)
        X = np.load(client_x_path)
        y = np.load(client_y_path)
        return X, y

def load_test_data(zip_path=None):
    """Load the test data for central evaluation.
    
    Args:
        zip_path (str, optional): Path to the zip file. If None and data splitting is needed,
                                 user will be prompted.
    """
    # Check if test data files exist
    test_x_path = os.path.join(ensure_save_dir(), "X_test.npy")
    test_y_path = os.path.join(ensure_save_dir(), "y_test.npy")
    
    if not os.path.exists(test_x_path) or not os.path.exists(test_y_path):
        print("Test data not found. Splitting data...")
        split_data_for_clients(zip_path=zip_path)
    
    # Load test data
    try:
        X_test = np.load(test_x_path)
        y_test = np.load(test_y_path)
        print(f"Successfully loaded test data: {len(X_test)} samples")
        return X_test, y_test
    except (OSError, IOError) as e:
        print(f"Error loading test data: {e}")
        print("Re-generating test data...")
        split_data_for_clients(zip_path=zip_path)
        X_test = np.load(test_x_path)
        y_test = np.load(test_y_path)
        return X_test, y_test

if __name__ == "__main__":
    # When run directly, prepare data for all clients
    print("Preparing data for federated learning...")
    
    # Ask for zip file path
    zip_path = get_zip_path()
    
    # Check if dataset directory exists but is invalid
    if os.path.exists(data_dir) and not is_dataset_valid():
        print(f"Dataset directory '{data_dir}' exists but doesn't contain the expected structure.")
        print("Detecting dataset structure...")
        detect_and_print_structure(data_dir)
        
        print("\nYour dataset appears to have a nested structure with 'Training' and 'Testing' subdirectories.")
        print("Would you like to re-extract the dataset? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            extract_dataset(zip_path, force_extract=True)
    
    # Split data for clients
    try:
        client_data = split_data_for_clients(num_clients=3, zip_path=zip_path)
        print(f"Data prepared for {len(client_data)} clients.")
    except Exception as e:
        print(f"Error preparing data: {e}")
        print("Please check your dataset structure and try again.")

