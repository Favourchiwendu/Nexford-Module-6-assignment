import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, utils
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class FashionMNISTClassifier:
    """
    A comprehensive CNN classifier for Fashion MNIST with enhanced performance analysis.
    Includes visualization tools and detailed performance metrics as suggested by supervisor.
    """

    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        """
        Initializes the classifier with default parameters.
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.x_train = self.y_train = None
        self.x_test = self.y_test = None
        self.y_test_original = None  # Store original labels before one-hot encoding
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        self.history = None  # Store training history

    def load_data(self):
        """
        Loads and prepares the Fashion MNIST dataset.
        """
        print("Loading Fashion MNIST dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test_original) = keras.datasets.fashion_mnist.load_data()
        self.y_test = self.y_test_original.copy()  # Keep original for evaluation
        
        print(f"Training data shape: {self.x_train.shape}")
        print(f"Test data shape: {self.x_test.shape}")

    def preprocess_data(self):
        """
        Preprocesses the data: scaling, reshaping and normalizing.
        Converts training labels to categorical format while preserving original test labels.
        """
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Preprocessing data...")
        # Scale and reshape images
        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)
        
        # One-hot encode only training labels (keep original test labels for metrics)
        self.y_train = utils.to_categorical(self.y_train, self.num_classes)
        
        print(f"Processed training shape: {self.x_train.shape}")
        print(f"Processed test shape: {self.x_test.shape}")

    def build_model(self):
        """
        Builds a 6-layer CNN model with enhanced architecture.
        """
        print("Building CNN model...")
        
        self.model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape,
                          kernel_initializer='he_normal'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),  # Added to reduce overfitting
            layers.Flatten(),
            layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            layers.Dropout(0.5),  # Added to reduce overfitting
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.summary()

    def compile_model(self):
        """
        Compiles the model with additional metrics (precision and recall) as suggested.
        """
        print("Compiling model with precision and recall metrics...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )

    def train_model(self, batch_size=128, epochs=15):
        """
        Trains the model and stores training history.
        """
        print(f"Training model for {epochs} epochs...")
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        print("Training completed.")

    def evaluate_model(self):
        """
        Evaluates model performance with comprehensive metrics.
        """
        print("\nEvaluating model performance...")
        
        # 1. Basic evaluation metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            self.x_test, utils.to_categorical(self.y_test_original), verbose=0
        )
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
        
        # 2. Generate predictions for full test set
        y_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        
        # 3. Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test_original, y_pred, target_names=self.class_names))
        
        # 4. Confusion matrix
        self.plot_confusion_matrix(self.y_test_original, y_pred)
        
        # 5. Training history visualization
        self.plot_training_history()

    def plot_training_history(self):
        """
        Plots training and validation accuracy/loss over epochs.
        """
        if self.history is None:
            print("No training history available.")
            return
            
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plots a confusion matrix for model predictions.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

    def predict_and_visualize(self, indices=None, num_samples=4):
        """
        Makes predictions and visualizes results for sample images.
        """
        if indices is None:
            indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(indices):
            img = self.x_test[idx]
            pred = self.model.predict(np.expand_dims(img, axis=0))[0]
            pred_label = np.argmax(pred)
            true_label = self.y_test_original[idx]
            
            plt.subplot(1, len(indices), i+1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"Pred: {self.class_names[pred_label]}\nTrue: {self.class_names[true_label]}",
                     color='green' if pred_label == true_label else 'red')
            plt.axis('off')
            
            print(f"\nImage {idx}:")
            print(f"True: {self.class_names[true_label]} ({true_label})")
            print(f"Pred: {self.class_names[pred_label]} ({pred_label})")
            print(f"Confidence: {pred[pred_label]:.2%}")
            print("Class probabilities:")
            for j, (name, prob) in enumerate(zip(self.class_names, pred)):
                print(f"  {name}: {prob:.2%}")
        
        plt.tight_layout()
        plt.show()

    def interpret_performance(self):
        """
        Provides interpretation of model performance as suggested by supervisor.
        """
        print("\n=== Model Performance Interpretation ===")
        print("1. Accuracy: Measures overall correctness of predictions.")
        print("2. Precision: For each class, what proportion of predicted positives were truly positive.")
        print("3. Recall: For each class, what proportion of actual positives were correctly identified.")
        print("4. F1-Score: Harmonic mean of precision and recall - good for imbalanced classes.")
        print("\nKey Observations:")
        print("- High accuracy with low precision/recall for a class indicates poor performance on that class.")
        print("- Comparing train vs validation metrics helps identify overfitting.")
        print("- Confusion matrix reveals which classes are most commonly confused.")

    def save_model(self, filepath):
        """Saves the trained model to specified filepath."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

# Main execution
if __name__ == "__main__":
    # Initialize classifier
    classifier = FashionMNISTClassifier()
    
    # Load and preprocess data
    classifier.load_data()
    classifier.preprocess_data()
    
    # Build, compile and train model
    classifier.build_model()
    classifier.compile_model()
    classifier.train_model(epochs=15)
    
    # Evaluate model with all suggested metrics
    classifier.evaluate_model()
    
    # Provide performance interpretation
    classifier.interpret_performance()
    
    # Make sample predictions
    print("\nMaking sample predictions...")
    classifier.predict_and_visualize([0, 12, 42, 100])  # Predict specific images
    
    # Save the trained model
    classifier.save_model("fashion_mnist_cnn_enhanced.keras")