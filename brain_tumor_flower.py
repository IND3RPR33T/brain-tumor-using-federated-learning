# brain_tumor_flower.py

import tensorflow as tf
import numpy as np
import os
import cv2
import zipfile
import hashlib
import argparse
import flwr as fl
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Brain Tumor Classification with ResNet50 using Flower')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--client_id', type=int, default=0, help='Client ID (0, 1, or 2)')
parser.add_argument('--num_clients', type=int, default=3, help='Total number of clients')
parser.add_argument('--server', type=str, default='127.0.0.1:8080', help='Server address')
parser.add_argument('--zip_path', type=str, required=True, help='Path to dataset zip file')
parser.add_argument('--data_dir', type=str, default='brain_tumor_data', help='Directory to extract dataset')
parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)
print(f"Output directory created: {args.output_dir}")

# ------------------------------
# 📂 DATA ACQUISITION & PREPROCESSING
# ------------------------------
# Extract dataset if needed
if not os.path.exists(args.data_dir):
    print(f"Extracting dataset from {args.zip_path} to {args.data_dir}...")
    try:
        with zipfile.ZipFile(args.zip_path, 'r') as zip_ref:
            zip_ref.extractall(args.data_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        exit(1)

img_size = (150, 150)
classes = ["glioma", "meningioma", "pituitary", "notumor"]

def load_data():
    images, labels = [], []
    
    # Process both Training and Testing directories
    for split in ["Training", "Testing"]:
        split_dir = os.path.join(args.data_dir, split)
        
        # Skip if directory doesn't exist
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping.")
            continue
            
        # Process each class directory
        for label, category in enumerate(classes):
            class_dir = os.path.join(split_dir, category)
            
            # Skip if class directory doesn't exist
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist, skipping.")
                continue
                
            print(f"Loading images from {class_dir}")
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    if not images:
        raise ValueError("No images were loaded. Please check the dataset structure.")
        
    return np.array(images), np.array(labels)

# Load and split data
print("Loading data...")
X, y = load_data()
X = X / 255.0  # Normalize
print(f"Loaded {len(X)} images with shape {X.shape}")

# Partition data for federated learning
def partition_data(X, y, num_clients, client_id):
    # Determine indices for this client
    # Simple partitioning: each client gets a different subset
    client_indices = np.arange(client_id, len(X), num_clients)
    
    # Get data for this client
    X_client = X[client_indices]
    y_client = y[client_indices]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42
    )
    
    print(f"Client {client_id} data: {len(X_train)} training, {len(X_test)} testing images")
    return X_train, X_test, y_train, y_test

# Get data for this client
X_train, X_test, y_train, y_test = partition_data(X, y, args.num_clients, args.client_id)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# ------------------------------
# 🧠 MODEL DEFINITION: ResNet50
# ------------------------------
def create_resnet50_model():
    base_model = tf.keras.applications.ResNet50(
        weights=None, 
        input_shape=(150, 150, 3), 
        include_top=False
    )
    base_model.trainable = True
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ------------------------------
# 🌸 FLOWER CLIENT IMPLEMENTATION
# ------------------------------
class BrainTumorClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.datagen = datagen
    
    def get_parameters(self, config):
        return [np.array(w) for w in self.model.get_weights()]
    
    def fit(self, parameters, config):
        # Set model weights with the parameters received from the server
        self.model.set_weights([np.array(w) for w in parameters])
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            self.datagen.flow(self.x_train, self.y_train, batch_size=args.batch_size),
            epochs=args.epochs,
            validation_data=(self.x_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Return updated model parameters and training results
        return [np.array(w) for w in self.model.get_weights()], len(self.x_train), {
            "accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1])
        }
    
    def evaluate(self, parameters, config):
        # Set model weights with the parameters received from the server
        self.model.set_weights([np.array(w) for w in parameters])
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Get predictions for more detailed metrics
        y_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        
        # Calculate class-wise metrics
        report = classification_report(self.y_test, y_pred, target_names=classes, output_dict=True)
        
        # Return evaluation results
        return loss, len(self.x_test), {
            "accuracy": float(accuracy),
            "f1_score": float(np.mean([report[c]["f1-score"] for c in classes]))
        }

# ------------------------------
# 🚀 MAIN EXECUTION
# ------------------------------
def main():
    # Create model
    model = create_resnet50_model()
    model.summary()
    
    # Create Flower client
    client = BrainTumorClient(model, X_train, y_train, X_test, y_test)
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )
    
    # After federated training, save the model
    model_path = os.path.join(args.output_dir, f'brain_tumor_model_client_{args.client_id}.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate the model
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save classification report to file
    report_path = os.path.join(args.output_dir, f'classification_report_client_{args.client_id}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(args.output_dir, f'confusion_matrix_client_{args.client_id}.png')
    plt.savefig(cm_path)
    
    print(f"Evaluation complete for client {args.client_id}")

if __name__ == "__main__":
    main()