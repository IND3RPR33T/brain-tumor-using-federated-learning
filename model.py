# model.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

def create_model():
    """Create a ResNet50 model for brain tumor classification."""
    base_model = ResNet50(
        weights=None,  # No pre-trained weights
        input_shape=(150, 150, 3),
        include_top=False
    )
    base_model.trainable = True
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, pituitary, no_tumor
    ])
    
    return model

def get_model_summary():
    """Get a string representation of the model summary."""
    model = create_model()
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

if __name__ == "__main__":
    # Print model summary when run directly
    model = create_model()
    model.summary()