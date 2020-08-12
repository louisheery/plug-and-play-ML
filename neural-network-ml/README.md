# Plug and Play ML

[![Build Status](https://img.shields.io/badge/python-3-blue)](https://github.com/louisheery/plug-and-play-ML)
[![Build Status](https://img.shields.io/badge/build-v1.1-brightgreen)](https://github.com/louisheery/plug-and-play-ML)
[![Build Status](https://img.shields.io/badge/build_status-published-brightgreen)](https://github.com/louisheery/plug-and-play-ML)

### 5. Neural Network
A Neural Network Image Classifier for Predicting Car Insurance Prices using the PyTorch Python Library.

#### Contents
- [Neural Network Price Prediction Model](insurance-price-prediction-ml.py)

#### Codebase Readout
```
--- START ---
--- 1. Defining Neural Network ---
--- 2. Datadset Import & Loading & Pre-processing ---
(3722, 11)
--- 3. Training Neural Network Classifier ---
[epoch:    1] loss: 0.836
...
[epoch:  100] loss: 0.758
--- 4. Saving Trained Neural Network Classifier ---
--- 5. Testing Neural Network Classifier ---
--- 6. Evaluating Neural Network Classifier ---

Confusion Matrix
[[280 284]
 [136 417]]

Confusion Report: Accuracy, F1 Score and ROC Accuracy:
              precision    recall  f1-score   support

         0.0       0.67      0.50      0.57       564
         1.0       0.59      0.75      0.67       553

    accuracy                           0.62      1117
   macro avg       0.63      0.63      0.62      1117
weighted avg       0.63      0.62      0.62      1117



ROC Accuracy:
0.6252613084016263
Epoch 100  -- Accuracy:  0.6252613084016263
[epoch:    1] loss: 0.645
...
[epoch:  200] loss: 0.678


Confusion Matrix
[[290 274]
 [160 393]]


Confusion Report: Accuracy, F1 Score and ROC Accuracy:
              precision    recall  f1-score   support

         0.0       0.64      0.51      0.57       564
         1.0       0.59      0.71      0.64       553

    accuracy                           0.61      1117
   macro avg       0.62      0.61      0.61      1117
weighted avg       0.62      0.61      0.61      1117



ROC Accuracy:
0.612426737460403
--- 7. Finding Optimal Hyperparameters of Neural Network Classifier ---
[skipped]
```
