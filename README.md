# Nexford-Module-6-assignment
# CNN fashion classifier

 **Data Loading and Preprocessing:**
   - The script automatically loads the Fashion MNIST dataset.
   - Data preprocessing steps include normalizing pixel values and converting labels to categorical format.

 **Model Training:**
    - The script automatically builds, compiles, and trains the model.
    - You can modify the number of epochs and batch size in the `train_model()` method if desired.

 **Model Evaluation and Visualization:**
    - After training, the script performs a comprehensive evaluation with metrics, classification report, confusion matrix, training curves and sample predictions.

 **Model Saving:** The trained model is saved as "fashion_mnist_cnn_enhanced.keras"


## Dependencies

* NumPy
* TensorFlow/Keras
* Matplotlib
* Scikit-learn
* Seaborn


## How to run the code
run Favour_fashion_mnist_classifier.py in your terminal
or run it directly in google colab using the ipynb version

# Model Performance Evaluation
The performance of the model was evaluated on the test dataset, and the following metrics were computed:

Test Accuracy: 92.90%

The model correctly classified 92.90% of the test data, indicating a high level of overall correctness.

Test Precision: 93.37%

The model's precision is 93.37%, meaning that 93.37% of the instances classified as positive were truly positive. This indicates that the model is effective at minimizing false positives.

Test Recall: 92.37%

The recall value of 92.37% shows that the model correctly identified 92.37% of the actual positive instances. This suggests that the model is also good at capturing most of the positive samples without missing many.

Test F1-Score: 92.87%

The F1-Score is 92.87%, which balances precision and recall, indicating that the model maintains a good trade-off between detecting positive instances and avoiding false positives.
