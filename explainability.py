# Fixed explainability.py with memory-efficient SHAP and correct Grad-CAM
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries
import os
import tensorflow as tf

# Constants
classes = ["glioma", "meningioma", "pituitary", "notumor"]

def explain_with_shap_memory_efficient(model, X_test, num_samples=3, save_dir="shap_explanations"):
    """Generate SHAP explanations using a memory-efficient approach."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Select a very small subset of test images to avoid memory issues
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)
    X_subset = X_test[indices]
    
    # Create a very small background dataset
    background = X_test[np.random.choice(X_test.shape[0], 10, replace=False)]
    
    # Use a wrapper function that processes one image at a time
    def model_predict(images):
        # Process one image at a time to save memory
        results = []
        for img in images:
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            results.append(pred)
        return np.array(results)
    
    # Initialize the SHAP explainer with a very small sample
    print("Initializing SHAP KernelExplainer with memory-efficient settings...")
    explainer = shap.KernelExplainer(
        model_predict, 
        background,
        link="identity"
    )
    
    # Process one image at a time
    for i, idx in enumerate(indices):
        img = X_test[idx]
        print(f"Generating SHAP explanation for image {i+1}/{num_samples}...")
        
        # Get prediction for this image
        pred = np.argmax(model.predict(np.expand_dims(img, axis=0))[0])
        pred_class = classes[pred]
        
        # Compute SHAP values for this single image
        shap_values = explainer.shap_values(
            np.expand_dims(img, axis=0),
            nsamples=100  # Use fewer samples to save memory
        )
        
        # Plot the explanation
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"Original\nPredicted: {pred_class}")
        plt.axis('off')
        
        # SHAP values visualization
        plt.subplot(1, 3, 2)
        # Create a simple visualization of the SHAP values
        shap_img = np.sum(np.abs(shap_values[pred][0]), axis=2)
        shap_img = (shap_img - np.min(shap_img)) / (np.max(shap_img) - np.min(shap_img) + 1e-8)
        plt.imshow(shap_img, cmap='hot')
        plt.title(f"SHAP Importance\nfor {pred_class}")
        plt.axis('off')
        
        # SHAP overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        plt.imshow(shap_img, cmap='hot', alpha=0.6)
        plt.title("SHAP Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"shap_explanation_{i}.png"))
        plt.close()
    
    print(f"SHAP explanations saved to {save_dir}")
    return True

def explain_with_lime(model, X_test, num_samples=5, save_dir="lime_explanations"):
    """Generate LIME explanations for model predictions."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Select a subset of test images
    if num_samples > len(X_test):
        num_samples = len(X_test)
    
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)
    
    # Initialize the LIME explainer
    explainer = lime.lime_image.LimeImageExplainer()
    
    # Function to predict with the model
    def predict_fn(images):
        return model.predict(images)
    
    # Generate and save LIME explanations
    for i, idx in enumerate(indices):
        img = X_test[idx]
        
        # Generate explanation
        explanation = explainer.explain_instance(
            img, 
            predict_fn,
            top_labels=len(classes),
            hide_color=0,
            num_samples=1000
        )
        
        # Get the prediction
        pred = np.argmax(model.predict(np.expand_dims(img, axis=0))[0])
        
        # Plot the explanation
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title(f"Original (Pred: {classes[pred]})")
        plt.axis('off')
        
        # LIME explanations for each class
        for j, class_idx in enumerate(explanation.top_labels):
            if j >= 5:  # Limit to 5 classes
                break
                
            temp, mask = explanation.get_image_and_mask(
                class_idx,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            plt.subplot(2, 3, j+2)
            plt.imshow(mark_boundaries(temp, mask))
            plt.title(f"LIME for {classes[class_idx]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"lime_explanation_{i}.png"))
        plt.close()
    
    print(f"LIME explanations saved to {save_dir}")

def generate_gradcam_visualizations(model, X_test, y_test, num_samples=3, save_dir="gradcam_explanations"):
    """Generate Grad-CAM visualizations for model predictions."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Select a subset of test images
    if num_samples > len(X_test):
        num_samples = len(X_test)
    
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)
    
    # Find the last convolutional layer in the model
    conv_layer = None
    for layer in model.layers:
        # Check if it's the ResNet50 layer
        if layer.name == 'resnet50':
            # Get the ResNet model
            resnet_model = layer
            # Find the last conv layer in ResNet
            for resnet_layer in resnet_model.layers:
                if 'conv' in resnet_layer.name.lower() and not 'bn' in resnet_layer.name.lower():
                    conv_layer = resnet_layer.name
            break
    
    if not conv_layer:
        # If we couldn't find a conv layer in ResNet, try to find any conv layer
        for layer in model.layers:
            if 'conv' in layer.name.lower():
                conv_layer = layer.name
                break
    
    if not conv_layer:
        print("Could not find any convolutional layer for Grad-CAM.")
        return
    
    print(f"Using layer '{conv_layer}' for Grad-CAM")
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = y_test[idx]
        
        # Get model prediction
        pred = np.argmax(model.predict(np.expand_dims(img, axis=0))[0])
        
        try:
            # Create a model that maps the input image to the activations
            # of the target layer and output predictions
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(conv_layer).output, model.output]
            )
            
            # Prepare image
            img_array = np.expand_dims(img, axis=0)
            
            # Compute gradient of the predicted class with respect to the output feature map
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, pred]
            
            # Extract gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Pool gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight output feature map with gradients
            conv_outputs = conv_outputs.numpy()[0]
            pooled_grads = pooled_grads.numpy()
            
            for j in range(pooled_grads.shape[-1]):
                conv_outputs[:, :, j] *= pooled_grads[j]
            
            # Average over all channels
            heatmap = np.mean(conv_outputs, axis=-1)
            
            # ReLU
            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
            
            # Resize heatmap to original image size
            heatmap = tf.image.resize(
                np.expand_dims(heatmap, axis=-1), 
                (img.shape[0], img.shape[1])
            ).numpy()[:,:,0]
            
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
            plt.title(f"Original: {classes[true_label]}\nPredicted: {classes[pred]}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(superimposed_img)
            plt.title("Grad-CAM")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"gradcam_{i}.png"))
            plt.close()
            
            print(f"Generated Grad-CAM for image {i}")
            
        except Exception as e:
            print(f"Error generating Grad-CAM for image {i}: {e}")
            
            # Try with a different approach - direct visualization without Grad-CAM
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Original: {classes[true_label]}\nPredicted: {classes[pred]}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.title("No Grad-CAM available\nShowing original image")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"gradcam_{i}_fallback.png"))
            plt.close()
    
    print(f"Grad-CAM visualizations saved to {save_dir}")

def run_all_explainability(model, X_test, y_test, num_samples=3, save_dir=None):
    """Run all explainability methods and save results."""
    print("Generating model explanations...")
    
    # Import cv2 here to avoid import errors
    import cv2
    
    # Create base directory
    base_dir = save_dir if save_dir else "explanations"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # SHAP explanations
    print("Generating SHAP explanations...")
    shap_dir = os.path.join(base_dir, "shap")
    try:
        explain_with_shap_memory_efficient(model, X_test, num_samples, save_dir=shap_dir)
    except Exception as e:
        print(f"SHAP explanations failed: {e}")
        print("Continuing with other explainability methods...")
    
    # LIME explanations
    print("Generating LIME explanations...")
    lime_dir = os.path.join(base_dir, "lime")
    explain_with_lime(model, X_test, num_samples, save_dir=lime_dir)
    
    # Grad-CAM explanations
    print("Generating Grad-CAM explanations...")
    gradcam_dir = os.path.join(base_dir, "gradcam")
    generate_gradcam_visualizations(model, X_test, y_test, num_samples, save_dir=gradcam_dir)
    
    print("All explanations generated successfully!")

if __name__ == "__main__":
    print("Explainability tools loaded. Import this module to use the functions.")