# evaluate.py
import tensorflow as tf
import numpy as np
import os
import argparse
import time
from datetime import datetime
from model import create_model
from data_utils import load_test_data
from visualization import plot_confusion_matrix, plot_roc_curve, plot_sample_predictions
from explainability import run_all_explainability
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define the results directory
RESULTS_DIR = r"D:\research work\brain_tumor_fl\results"

def load_model_safely(model_path):
    """Load a model with proper error handling and support for different formats."""
    print(f"Attempting to load model from: {model_path}")
    
    # Create base model architecture
    model = create_model()
    
    try:
        # First try loading as weights
        if model_path.endswith('.h5'):
            try:
                model.load_weights(model_path)
                print(f"Successfully loaded model weights from {model_path}")
                return model
            except Exception as e:
                print(f"Could not load as weights file: {e}")
                print("Trying to load as full model...")
        
        # Try loading as full model
        try:
            loaded_model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded full model from {model_path}")
            return loaded_model
        except Exception as e:
            print(f"Could not load as full model: {e}")
            raise ValueError(f"Failed to load model from {model_path}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_evaluation_results(results, filename="evaluation_results.txt"):
    """Save evaluation metrics to a text file."""
    with open(filename, 'w') as f:
        f.write(f"Brain Tumor Classification Model Evaluation\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {results['model_path']}\n\n")
        
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Test Loss: {results['loss']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        
        f.write("\nClass-wise Metrics:\n")
        for i, class_name in enumerate(results['class_names']):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {results['precision'][i]:.4f}\n")
            f.write(f"  Recall: {results['recall'][i]:.4f}\n")
            f.write(f"  F1-Score: {results['f1'][i]:.4f}\n")
            f.write(f"  Support: {results['support'][i]}\n\n")
        
        f.write(f"\nEvaluation completed in {results['eval_time']:.2f} seconds\n")
    
    print(f"Evaluation results saved to {filename}")

def create_lime_explanation(model, X_test, y_test, num_samples=3, save_dir=None):
    """Create LIME explanations as an alternative to SHAP."""
    try:
        # Try to import LIME
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        import matplotlib.pyplot as plt
        
        # Create explanations directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Get class names
        class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
        
        # Sample images to explain
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        for i, idx in enumerate(indices):
            image = X_test[idx]
            true_label = y_test[idx]
            
            # Generate explanation
            explanation = explainer.explain_instance(
                image, 
                model.predict, 
                top_labels=4,
                hide_color=0, 
                num_samples=1000
            )
            
            # Get the explanation for the true label
            temp, mask = explanation.get_image_and_mask(
                true_label, 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            
            # Plot the explanation
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f"Original: {class_names[true_label]}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mark_boundaries(temp, mask))
            plt.title("LIME Explanation")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"lime_explanation_{i}.png"))
            plt.close()
        
        print(f"LIME explanations saved to '{save_dir}' directory")
        return True
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate brain tumor model")
    parser.add_argument("--zip-path", type=str, help="Path to the dataset zip file (optional)")
    parser.add_argument("--model-path", type=str, 
                        help="Path to the trained model file (if not provided, you'll be prompted to enter it)")
    parser.add_argument("--skip-explainability", action="store_true", 
                        help="Skip the explainability analysis (useful if SHAP is causing errors)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results (default: results_dir/evaluation_results.txt)")
    args = parser.parse_args()
    
    # If model path is not provided as a command-line argument, ask for it
    model_path = args.model_path
    if model_path is None:
        model_path = input("Please enter the path to your trained model file: ")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    
    # Set up subdirectories for different result types
    vis_dir = os.path.join(RESULTS_DIR, "visualizations")
    samples_dir = os.path.join(vis_dir, "samples")
    explain_dir = os.path.join(RESULTS_DIR, "explanations")
    
    # Create subdirectories if they don't exist
    for directory in [vis_dir, samples_dir, explain_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Set default output file if not specified
    if args.output is None:
        args.output = os.path.join(RESULTS_DIR, "evaluation_results.txt")
    
    print("\n===== Brain Tumor Model Evaluation =====\n")
    
    # Load the model with improved error handling
    model = None
    while model is None:
        model = load_model_safely(model_path)
        if model is None:
            retry = input("Would you like to enter a different model path? (y/n): ")
            if retry.lower() == 'y':
                model_path = input("Please enter the path to your trained model file: ")
            else:
                print("Exiting evaluation.")
                return
    
    # Ensure model is compiled
    if not model.optimizer:
        print("Model needs to be compiled. Compiling with default settings...")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    # Load test data
    print("\nLoading test data...")
    try:
        X_test, y_test = load_test_data(args.zip_path)
        print(f"Loaded {len(X_test)} test samples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        retry = input("Would you like to try a different dataset path? (y/n): ")
        if retry.lower() == 'y':
            zip_path = input("Please enter the path to your dataset zip file: ")
            try:
                X_test, y_test = load_test_data(zip_path)
                print(f"Loaded {len(X_test)} test samples")
            except Exception as e:
                print(f"Error loading test data: {e}")
                print("Exiting evaluation.")
                return
        else:
            print("Exiting evaluation.")
            return
    
    # Start timing the evaluation
    start_time = time.time()
    
    # Evaluate the model
    print("\nEvaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    classification_rep = classification_report(y_test, y_pred, target_names=class_names)
    print(classification_rep)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    # Using the function signatures that match your implementation
    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(vis_dir, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_pred_proba, save_path=os.path.join(vis_dir, "roc_curve.png"))
    plot_sample_predictions(X_test, y_test, y_pred, model, num_samples=5, save_dir=samples_dir)
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    # Save evaluation results
    results = {
        'model_path': model_path,
        'accuracy': accuracy,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'class_names': class_names,
        'classification_report': classification_rep,
        'eval_time': eval_time
    }
    save_evaluation_results(results, args.output)
    
    # Generate model explanations (with error handling for SHAP)
    if not args.skip_explainability:
        print("\nGenerating model explanations...")
        try:
            # Try to modify the function call to include save_dir if the function supports it
            try:
                # First try with save_dir parameter
                run_all_explainability(model, X_test, y_test, num_samples=3, save_dir=explain_dir)
            except TypeError:
                # If that fails, try the original call and hope it uses the right directory
                # We'll need to temporarily change the current directory
                original_dir = os.getcwd()
                if not os.path.exists(explain_dir):
                    os.makedirs(explain_dir)
                os.chdir(explain_dir)
                run_all_explainability(model, X_test, y_test, num_samples=3)
                os.chdir(original_dir)
        except Exception as e:
            print(f"\nError during explainability analysis: {e}")
            print("\nSHAP error detected. This is likely due to the 'gradient registry has no entry for: shap_AddV2' error.")
            print("This is a known compatibility issue between SHAP and certain TensorFlow versions.")
            
            print("\nPossible solutions:")
            print("1. Update SHAP: pip install --upgrade shap")
            print("2. Try a different explainability method")
            print("3. Run without explainability: python evaluate.py --skip-explainability")
            
            # Offer to continue with alternative explainability
            alt_explain = input("\nWould you like to try an alternative explainability method? (y/n): ")
            if alt_explain.lower() == 'y':
                try:
                    print("\nTrying alternative explainability method (LIME)...")
                    # Use our custom LIME implementation
                    success = create_lime_explanation(model, X_test, y_test, num_samples=3, save_dir=explain_dir)
                    if success:
                        print("LIME explanations completed successfully!")
                    else:
                        print("LIME explanations failed. Try running without explainability.")
                except Exception as e2:
                    print(f"Alternative explainability also failed: {e2}")
                    
                    # Try Grad-CAM as a last resort
                    try:
                        print("\nTrying Grad-CAM as a last resort...")
                        from explainability import generate_gradcam_visualizations
                        
                        # Check if the function accepts a save_dir parameter
                        import inspect
                        sig = inspect.signature(generate_gradcam_visualizations)
                        
                        if 'save_dir' in sig.parameters:
                            generate_gradcam_visualizations(model, X_test, y_test, num_samples=3, save_dir=explain_dir)
                        else:
                            # If not, temporarily change directory
                            original_dir = os.getcwd()
                            os.chdir(explain_dir)
                            generate_gradcam_visualizations(model, X_test, y_test, num_samples=3)
                            os.chdir(original_dir)
                            
                        print("Grad-CAM visualizations completed successfully!")
                    except Exception as e3:
                        print(f"Grad-CAM also failed: {e3}")
    else:
        print("\nSkipping explainability analysis as requested.")
    
    print(f"\nEvaluation complete! Results saved to '{args.output}'")
    print(f"Visualizations saved to '{vis_dir}' directory")
    print(f"Sample predictions saved to '{samples_dir}' directory")
    if not args.skip_explainability:
        print(f"Explanations saved to '{explain_dir}' directory (if successful)")
    print(f"\nTotal evaluation time: {eval_time:.2f} seconds")

if __name__ == "__main__":
    main()