# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import cv2
import os

# Constants
classes = ["glioma", "meningioma", "pituitary", "notumor"]

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_pred_proba, save_path="roc_curve.png"):
    """Plot ROC curve for multi-class classification."""
    plt.figure(figsize=(10, 8))
    
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def plot_sample_predictions(X_test, y_test, y_pred, model, num_samples=5, save_dir="sample_predictions"):
    """Plot sample predictions with their true and predicted labels."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get random indices
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = classes[y_test[idx]]
        pred_label = classes[y_pred[idx]]
        
        # Get prediction probabilities
        probs = model.predict(np.expand_dims(img, axis=0))[0]
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPredicted: {pred_label}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        bars = plt.bar(classes, probs)
        plt.xticks(rotation=45)
        plt.title("Prediction Probabilities")
        
        # Highlight the predicted class
        bars[y_pred[idx]].set_color('red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
        plt.close()
    
    print(f"{num_samples} sample predictions saved to {save_dir}")

def apply_gradcam(model, img, layer_name="conv5_block3_out", save_path="gradcam.png"):
    """Apply Grad-CAM to visualize important regions in the image."""
    # Create a model that maps the input image to the activations
    # of the last conv layer and output predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Prepare image
    img_array = np.expand_dims(img, axis=0)
    
    # Compute gradient of the predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight output feature map with gradients
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Average over all channels
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # ReLU
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    img_rgb = (img * 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    
    # Save the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Grad-CAM visualization saved to {save_path}")
    
    return superimposed_img

def plot_training_history(history, save_path="training_history.png"):
    """Plot training history."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

if __name__ == "__main__":
    print("Visualization utilities loaded. Import this module to use the functions.")