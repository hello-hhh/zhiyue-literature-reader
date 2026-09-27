# Machine Learning Concepts

## Supervised Learning
Supervised learning trains a model on labelled data where the correct output is known for each input.
The model learns to map inputs to outputs by minimising a loss function during training.
Common supervised learning algorithms include linear regression, logistic regression, decision trees, and neural networks.
Classification predicts a discrete category (e.g. spam or not spam).
Regression predicts a continuous value (e.g. house price in dollars).

## Unsupervised Learning
Unsupervised learning finds patterns in data without any labelled examples.
Clustering groups similar data points together — k-means is the most common algorithm.
Dimensionality reduction compresses data to fewer features — PCA (Principal Component Analysis) is the standard method.
Autoencoders are neural networks trained to compress and then reconstruct their input.

## Neural Networks
A neural network consists of layers of interconnected nodes called neurons.
Each connection has a weight that is adjusted during training via backpropagation.
The activation function introduces non-linearity — ReLU (Rectified Linear Unit) is most commonly used in hidden layers.
Deep learning refers to neural networks with many hidden layers.
Convolutional Neural Networks (CNNs) are designed for image data and use spatial filtering.
Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state across time steps.
Transformers use self-attention to process all tokens in a sequence in parallel, replacing RNNs for most NLP tasks.

## Training Concepts
Overfitting occurs when a model memorises the training data and performs poorly on new data.
Regularisation techniques like L1 (Lasso) and L2 (Ridge) add a penalty for large weights to reduce overfitting.
Dropout randomly deactivates neurons during training to prevent co-adaptation between neurons.
A validation set is used during training to monitor performance and tune hyperparameters.
The test set is held out completely until final evaluation — never used during training.
Cross-validation splits data into k folds and trains k models to get a more reliable accuracy estimate.
The learning rate controls how much weights change per update — too high causes divergence, too low causes slow training.
Batch size determines how many training samples are processed before updating the weights.
An epoch is one complete pass through the entire training dataset.

## Evaluation Metrics
Accuracy measures the fraction of correct predictions out of all predictions.
Precision measures how many of the predicted positives are actually positive.
Recall (also called sensitivity) measures how many actual positives were correctly identified.
F1 score is the harmonic mean of precision and recall — useful when classes are imbalanced.
AUC-ROC (Area Under the Curve — Receiver Operating Characteristic) measures a classifier's ability to distinguish between classes.
Mean Squared Error (MSE) is the average squared difference between predicted and actual values, used for regression.
Mean Absolute Error (MAE) is the average absolute difference — less sensitive to outliers than MSE.
