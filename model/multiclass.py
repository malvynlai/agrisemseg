# Takes in the pretrained ResNet50 class defined in res_class_model. py, and then uses it as the baseline model for
# MultiOutputClassifier to perform multilabel classification, after finetuning the classifier.

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.data_processor import processing
from model.res_class_model import ResNet50Classifier
from sklearn.metrics import classification_report
import joblib
from data.data_processor import processing, AgricultureConfigs

# Initialize the agriculture configs
configs = AgricultureConfigs()

# Process the data using the agriculture configs
X_train, X_test, y_train, y_test = processing(configs)

resnet_classifier = ResNet50Classifier(n_classes=1, epochs=10)
multi_target_classifier = MultiOutputClassifier(resnet_classifier, n_jobs=-1)
multi_target_classifier.fit(X_train, y_train)
y_pred = multi_target_classifier.predict(X_test)
accuracy_per_target = [accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
print(f"Accuracy per target: {accuracy_per_target}")
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {overall_accuracy}")


for i in range(y_test.shape[1]):
    print(f"Classification report for target {i}:")
    print(classification_report(y_test[:, i], y_pred[:, i]))

joblib.dump(multi_target_classifier, 'multi_target_classifier.pkl')
print("Model saved as multi_target_classifier.pkl")

