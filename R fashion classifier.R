#library(tensorflow)
#install_tensorflow(method = "virtualenv")
# Load necessary libraries
#library(keras)
#library(ggplot2)
#library(caret)
#library(reshape2)
#library(reticulate)
#library(dplyr)
#library(gridExtra)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% fashion_mnist

# Normalize pixel values
x_train <- x_train / 255
x_test <- x_test / 255

# Reshape data to include channel dimension
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# Convert class vectors to binary class matrices
y_train_cat <- to_categorical(y_train, 10)
y_test_cat <- to_categorical(y_test, 10)

# Define class names
class_names <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

#Build CNN Model

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(28, 28, 1), kernel_initializer = 'he_normal') %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu', kernel_initializer = 'he_normal') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

#Compile the model

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#Train the model
history <- model %>% fit(
  x_train, y_train_cat,
  batch_size = 128,
  epochs = 15,
  validation_split = 0.1,
  verbose = 1
)

# Evaluate on test data
scores <- model %>% evaluate(x_test, y_test_cat, verbose = 0)
cat(sprintf("Test Accuracy: %.4f\n", scores$accuracy))

# Generate predictions
pred_probs <- model %>% predict(x_test)
pred_classes <- apply(pred_probs, 1, which.max) - 1  # Adjust for zero-based indexing

# Confusion matrix
conf_mat <- table(Predicted = pred_classes, Actual = y_test)
print(conf_mat)

# Classification report
confusion <- confusionMatrix(as.factor(pred_classes), as.factor(y_test))
print(confusion)

# Convert history to data frame
df <- data.frame(
  epoch = 1:15,
  accuracy = history$metrics$accuracy,
  val_accuracy = history$metrics$val_accuracy,
  loss = history$metrics$loss,
  val_loss = history$metrics$val_loss
)

# Accuracy plot
acc_plot <- ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = accuracy, color = "Train Accuracy")) +
  geom_line(aes(y = val_accuracy, color = "Validation Accuracy")) +
  labs(title = "Model Accuracy", y = "Accuracy") +
  theme_minimal()

# Loss plot
loss_plot <- ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Train Loss")) +
  geom_line(aes(y = val_loss, color = "Validation Loss")) +
  labs(title = "Model Loss", y = "Loss") +
  theme_minimal()

# Display plots side by side
grid.arrange(acc_plot, loss_plot, ncol = 2)

# Function to plot sample predictions
plot_predictions <- function(indices) {
  par(mfrow = c(1, length(indices)))
  for (i in indices) {
    img <- x_test[i,,,1]
    true_label <- class_names[y_test[i] + 1]
    pred_label <- class_names[pred_classes[i] + 1]
    col <- ifelse(true_label == pred_label, "green", "red")
    image(1:28, 1:28, t(apply(img, 2, rev)), col = gray.colors(256), axes = FALSE,
          main = paste0("Pred: ", pred_label, "\nTrue: ", true_label), col.main = col)
  }
  par(mfrow = c(1,1))
}

# Example usage
set.seed(123)
sample_indices <- sample(1:length(y_test), 4)
plot_predictions(sample_indices)


# Save the trained model
save_model_hdf5(model, "fashion_mnist_cnn_enhanced.h5")

