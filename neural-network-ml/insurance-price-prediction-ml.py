# PyTorch: Neural Network Car Insurance Pricing Prediction Model
# Author: Louis Heery

### Import required libraries ###
import numpy as np
import pickle
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

### 0. Device Setup ###
device = 'cpu'
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
Tensor = FloatTensor
seed = 323
np.random.seed(seed)
torch.manual_seed(seed)



###### Model Implementation ######
### 1. Neural Network ###
class Net(nn.Module):

    def __init__(self, neurons, num_features=9):
        super(Net, self).__init__()

        ## Define number of Neurons in each Layer
        # If Neuron is specified as 9, then used the optimised NN architecture
        if (neurons == 9):
            a = 9
            b = 8
            c = 4

        # Otherwise, calculate number of neurons in other layers based on specified
        # number of neurons in first layer
        else:
            a = neurons
            b = int(a*0.5)
            c = int(b*0.5)

        self.l1 = nn.Linear(num_features, a)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(a,b)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(b,c)
        self.tanh3 = nn.Tanh()
        self.l4 = nn.Linear(c,1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.tanh1(x)
        x = self.l2(x)
        x = self.tanh2(x)
        x = self.l3(x)
        x = self.tanh3(x)
        x = self.l4(x)
        x = self.sig(x)
        return x


### 2. Car Insurance Pricing Prediction Model ###
class ClaimClassifier():

    def __init__(self, epoch=100, batchsize=4, learnrate=0.0001, neurons=9, num_features=9):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """

        # Define NN Hyperparameters based on Class Constructor Parameters
        self.net = Net(neurons=neurons, num_features=num_features)
        self.EPOCH = epoch
        self.BATCHSIZE = batchsize
        self.LR = learnrate
        self.optimizer = optim.Adam(self.net.parameters(), lr=learnrate, betas=(0.9, 0.999))
        self.loss_func = nn.BCEWithLogitsLoss()

    # convert numpy -> tensor
    def xTensor(self, x):
        """Numpy to Tensor Conversion Function for x_data

        This function prepares converts x_data (values) from a Numpy array
        to a PyTorch Tensor variable

        Parameters
        ----------
        x : ndarray
            An array, this is the raw data

        Returns
        -------
        X_tensor: pytorch tensor
            A Tensor array.
        """
        # Check data is of type Numpy and float32
        x = np.array(x, dtype=np.float32)

        # Convert to Tensor
        X_tensor = Variable(torch.from_numpy(x))
        return X_tensor

    # convert numpy -> tensor
    def yTensor(self, y):
        """Numpy to Tensor Conversion Function for y_data

        This function prepares converts y_data (labels) from a Numpy array
        to a PyTorch Tensor variable

        Parameters
        ----------
        y : ndarray
            An array, this is the raw data

        Returns
        -------
        Y_tensor: pytorch tensor
            A Tensor array.
        """
        # Resize Numpy to a 1-D Array
        y = y.reshape((y.shape[0],1))

        # Convert to Tensor
        Y_tensor = Variable(torch.from_numpy(y)).type(torch.FloatTensor)
        return Y_tensor


    def _balance_dataset(self, X_y_raw):
        """Function to balance dataset used for training/validation/testing

        This function balances the dataset so it contains an equal number of
        Class 0 and Class 1 events

        Parameters
        ----------
        X_y_raw : ndarray
            An array, this is the raw data

        Returns
        -------
        X_y_balanced: ndarray
            An array, but balanced for each Class
        """
        # Seperate dataset into Class 0 and Class 1 events
        class_0 = X_y_raw[X_y_raw[:,-1] == 0]
        class_1 = X_y_raw[X_y_raw[:,-1] == 1]

        # Shuffle Class_0 events
        np.random.shuffle(class_0)

        # Take Subset of Class_0 events of equal size to Class 1 events
        class_1_size = class_1.shape[0]
        class_0_subset = class_0[:class_1_size,]
        X_y_balanced = np.vstack((class_0_subset,class_1))

        # Shuffle combined balanced dataset before returning
        np.random.shuffle(X_y_balanced)

        return X_y_balanced


    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # Scale dataset to range of [0,1] for all x-values
        standard_scaler = MinMaxScaler()
        standard_scaler.fit(X_raw)
        X_raw_scaled = standard_scaler.transform(X_raw)

        # Return balanced dataset
        self.dataset = X_raw_scaled
        return self.dataset

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # Apply preprocessing to dataset first
        X_clean = self._preprocessor(X_raw)

        if type(y_raw) is not np.ndarray:
            y_raw = y_raw.to_numpy()

        Y_clean = y_raw.reshape(-1, 1)

        # Combine x and y data together
        X_Y_clean = np.hstack((X_clean, Y_clean))

        # Split into training and validation dataset in ratio of 0.7:0.3
        train, val = np.split(X_Y_clean, [int(0.7 * len(X_Y_clean))])

        # Split x and y data for train and validation dataset
        trainX = train[:,:-1]
        trainY = train[:,-1:]
        valX = val[:,:-1]
        valY = val[:,-1:]

        all_losses = []

        # Convert data to Tensors
        X_tensor_train = self.xTensor(trainX)
        Y_tensor_train = self.yTensor(trainY)

        X_tensor_val = self.xTensor(valX)
        Y_tensor_val = self.yTensor(valY)

        losses_train = []
        acc_val = []

        # Loop through required number of epochs
        for epoch in range(self.EPOCH):

            # Create empty array to store loss values for batches
            batch_losses = []

            # Put model into training mode
            self.net.train()

            # Loop through required number of batches, based on self.BATCHSIZE
            for batch in range(0, X_tensor_train.size(0), self.BATCHSIZE):

                # Specify which are the input and target values for this batch
                inputs = X_tensor_train[batch:batch + self.BATCHSIZE, :]
                targets = Y_tensor_train[batch:batch + self.BATCHSIZE]

                # Apply zero_grad to optimiser
                self.optimizer.zero_grad()

                # Determine prediction and loss function of neural network
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, targets)

                # Propagate loss function backwards
                loss.backward()

                self.optimizer.step()
                batch_losses.append(loss.data.numpy())

            # Calculate average loss value for the completed batch
            average_batch_loss = sum(batch_losses)/len(batch_losses)
            losses_train.append(average_batch_loss)

            # Put model into evaluation mode
            self.net.eval()

            # Calculate output of network
            outputs = self.net(X_tensor_val)

            # Calculate accuracy of network
            target_y = Y_tensor_val.data.numpy()
            pred_y = outputs.data.numpy()
            roc_auc = roc_auc_score(target_y,pred_y)

            acc_val.append(loss)
            print(f'[epoch: {epoch + 1:4d}] loss: {loss:1.3f}')

        # GRAPH PLOT of training loss and validation accuracy
        plt.plot(losses_train, label="Training Loss (avg. over each Epoch")
        plt.plot(acc_val, label="Accuracy on Validation Set")
        plt.xlabel('Epoch Number')
        plt.ylabel('Binary Cross Entropy Loss / ROC Accuracy')
        name = 'nn_testing_plots/accuracyloss_epoch' + str(self.EPOCH) + '_batch' + str(self.BATCHSIZE) + '_lr' + str(self.LR) + '.pdf'
        plt.savefig(name)
        plt.close()

        # Return trained network
        return self


    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # Convert Pandas Dataframe to Numpy Array
        if type(X_raw) is not np.ndarray:
            X_raw = X_raw.to_numpy()

        # Preprocess dataset and convert to Tensor
        X_clean = self._preprocessor(X_raw)
        X_tensor_test = self.xTensor(X_clean)

        # Put model into evaluation mode
        self.net.eval()

        # Calculate outputs of Network
        outputs = self.net(X_tensor_test)

        # Convert outputs to Numpy and return
        pred_y = outputs.data.numpy()

        return pred_y

    # QUESTION 2.2
    def evaluate_architecture(self, X_raw, y_raw):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """

        # Put network into evaluation mode
        self.net.eval()

        # Preprocess evaluation dataset
        X_clean = self._preprocessor(X_raw)

        if type(y_raw) is not np.ndarray:
            y_raw = y_raw.to_numpy()

        y_raw = y_raw.reshape(-1, 1)

        # Convert evaluation dataset to Numpy
        X_tensor_test = self.xTensor(X_clean)
        Y_tensor_test = self.yTensor(y_raw)

        # Calculate output using a Cut Value of 0.1
        # -> Values in range [0, 0.1] labelled as Class 0
        # -> Values in range [0.1, 1.0] labelled as Class 1
        outputs_unbinned = self.net(X_tensor_test)
        outputs = (self.net(X_tensor_test) > 0.1)

        predicted_val = outputs.data.float()
        predicted_val_unbinned = outputs_unbinned.data.float()

        pred_y = predicted_val.numpy()
        pred_y_unbinned = predicted_val_unbinned.numpy()

        target_y = y_raw

        # GRAPH OF Unbinned Prediction Histogram
        result = np.hstack((target_y, pred_y_unbinned))
        result_actual1 = result[(1.0<=result[:,0])]
        result_actual0 = result[(0.0==result[:,0])]
        plt.hist([result_actual0[:,1],result_actual1[:,1]], stacked=True, label=['Actual Class = Class 0', 'Actual Class = Class 1'])
        plt.legend()
        plt.xlabel('Predicted Class')
        plt.ylabel('Number of Events')
        plt.xlim(0,1)
        name = 'nn_evaluation_plots/predictionhstogram_epoch' + str(self.EPOCH) + '_batch' + str(self.BATCHSIZE) + '_lr' + str(self.LR) + '.pdf'
        plt.legend()
        plt.savefig(name)
        plt.close()

        # Calculate Confusion Matrix
        print("\n\nConfusion Matrix")
        print(confusion_matrix(target_y,pred_y))

        # Calculate Confusion Report (Accuracy, F1 Score, Confusion Matrix)
        print("\n\nConfusion Report: Accuracy, F1 Score and ROC Accuracy:")
        print(classification_report(target_y,pred_y))

        # Calculate ROC-AUC, which will be returned by this function
        print ('\n\nROC Accuracy:')
        roc_auc = roc_auc_score(target_y,pred_y)
        print(roc_auc)

        # GRAPH OF Prediction Histogram [BINNED using Class Split at 0.1]
        result = np.hstack((target_y, pred_y))
        result_actual1 = result[(1.0<=result[:,0])]
        result_actual0 = result[(0.0==result[:,0])]
        plt.hist([result_actual0[:,1],result_actual1[:,1]], stacked=True, label=['Actual Class = Class 0', 'Actual Class = Class 1'])
        plt.legend()
        plt.xlabel('Predicted Class')
        plt.ylabel('Number of Events')
        plt.xlim(0,1)
        name = 'nn_evaluation_plots/BINNEDpredictionhstogram_epoch' + str(self.EPOCH) + '_batch' + str(self.BATCHSIZE) + '_lr' + str(self.LR) + '.pdf'
        plt.legend()
        plt.savefig(name)
        plt.close()

        # Return final ROC Accuracy
        return roc_auc


    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


    def load_model(self):
        # Please alter this section so that it works in tandem with the save_model method of your class
        with open('part2_claim_classifier.pickle', 'rb') as target:
            trained_model = pickle.load(target)
        return trained_model


### 3. Car Insurance Pricing Prediction Model Hyperparameter Optimisation ###
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """

    # Initialise dataset
    data = np.genfromtxt('car_insurance_training_data.csv', delimiter=",", skip_header=1)

    # Balance and Preprocess Dataset
    balanced_dataset = _balance_dataset_public(data)
    processed_dataset = _preprocessor_public(balanced_dataset)

    # Segment Training and validation Component of Dataset in ratio of 0.7:0.3
    train, test = np.split(processed_dataset, [int(0.7 * len(processed_dataset))])
    train_x = train[:,:-2]
    train_y = train[:,-1:]
    val_x = test[:,:-2]
    val_y = test[:,-1:]

    # Initialise Hyperparameter array
    best_hp = {'Epoch': None, 'BatchSize': None, 'LearningRate': None, 'ActFunc': None, 'Layers': None, 'Neurons': None}


    # OPTIMISATION: EPOCH
    best_accuracy = 0

    for epoch in [1,5,10,20,40,100,200,400,600]:

        classifier = ClaimClassifier(epoch=epoch)

        # Training of Classifier
        classifier.fit(train_x, train_y)
        accuracy = classifier.evaluate_architecture(val_x, val_y)
        print("Epoch", epoch, " -- Accuracy: ", accuracy)

        # Update Hyperparameter value, if accuracy > current best accuracy
        if (accuracy > best_accuracy):
            best_hp['Epoch'] = epoch

    # OPTIMISATION: BATCH SIZE
    best_accuracy = 0

    for batchsize in [2,4,8,16,32,64,128,256]:

        classifier = ClaimClassifier(epoch=best_hp['Epoch'], batchsize=batchsize)

        # Training of Classifier
        classifier.fit(train_x, train_y)
        accuracy = classifier.evaluate_architecture(val_x, val_y)
        print("BatchSize", batchsize, " -- Accuracy: ", accuracy)

        # Update Hyperparameter value, if accuracy > current best accuracy
        if (accuracy > best_accuracy):
            best_hp['BatchSize'] = batchsize

    # OPTIMISATION: LEARNING RATE
    best_accuracy = 0

    for learningrate in [0.000001, 0.00001, 0.00003, 0.00006, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]:

        classifier = ClaimClassifier(epoch=best_hp['Epoch'], batchsize=best_hp['BatchSize'], learnrate=learningrate)

        # Training of Classifier
        classifier.fit(train_x, train_y)
        accuracy = classifier.evaluate_architecture(val_x, val_y)
        print("LearningRate", learningrate, " -- Accuracy: ", accuracy)

        # Update Hyperparameter value, if accuracy > current best accuracy
        if (accuracy > best_accuracy):
            best_hp['LearningRate'] = learningrate

    # OPTIMISATION: NUMBER OF NEURONS
    best_accuracy = 0

    for neurons in [512, 256, 128, 64, 32, 18, 16, 12, 9, 8, 4]:

        classifier = ClaimClassifier(epoch=best_hp['Epoch'], batchsize=best_hp['BatchSize'], learnrate=best_hp['LearningRate'], neurons=neurons)

        # Training of Classifier
        classifier.fit(train_x, train_y)
        accuracy = classifier.evaluate_architecture(val_x, val_y)
        print("Neurons", neurons, " -- Accuracy: ", accuracy)

        # Update Hyperparameter value, if accuracy > current best accuracy
        if (accuracy > best_accuracy):
            best_hp['Neurons'] = neurons

    # RETURN OPTIMAL HYPERPARAMTERS
    return best_hp

### 5. Helper Function for Processing Dataset ###
def _balance_dataset_public(X_Y_raw):
        """Function to balance dataset used for training/validation/testing

        This function balances the dataset so it contains an equal number of
        Class 0 and Class 1 events

        Parameters
        ----------
        X_y_raw : ndarray
            An array, this is the raw data

        Returns
        -------
        X_y_balanced: ndarray
            An array, but balanced for each Class
        """

        # Seperate dataset into Class 0 and Class 1 events
        class_0 = X_Y_raw[X_Y_raw[:,-1] == 0]
        class_1 = X_Y_raw[X_Y_raw[:,-1] == 1]

        # Shuffle Class_0 events
        np.random.shuffle(class_0)

        # Take Subset of Class_0 events of equal size to Class 1 events
        class_1_size = class_1.shape[0]
        class_0_subset = class_0[:class_1_size,]
        X_raw_balanced = np.vstack((class_0_subset,class_1))

        # Shuffle combined balanced dataset before returning
        np.random.shuffle(X_raw_balanced)

        return X_raw_balanced

def _preprocessor_public(X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """

        # Scale dataset to range of [0,1] for all x-values
        standard_scaler = MinMaxScaler()
        standard_scaler.fit(X_raw)
        X_raw_scaled = standard_scaler.transform(X_raw)

        return X_raw_scaled



###### Model Usage ######
if __name__ == "__main__":
    print("--- START ---")
    print("--- 1. Defining Neural Network ---")
    classifier = ClaimClassifier()

    print("--- 2. Datadset Import & Loading & Pre-processing ---")
    data = np.genfromtxt('car_insurance_training_data.csv', delimiter=",", skip_header=1)

    balanced_dataset = classifier._balance_dataset(data)
    processed_dataset = classifier._preprocessor(balanced_dataset)
    print(processed_dataset.shape)

    train_x = processed_dataset[0:2600,:-2]
    train_y = processed_dataset[0:2600,-1:]

    val_x = processed_dataset[3100:3600,:-2]

    test_x = processed_dataset[2600:3100,:-2]
    test_y = processed_dataset[2600:3100,-1:]

    # Training of Classifier
    print("--- 3. Training Neural Network Classifier ---")
    classifier.fit(train_x, train_y)

    print("--- 4. Saving Trained Neural Network Classifier ---")
    classifier.save_model()

    # Prediction from Classifier
    print("--- 5. Testing Neural Network Classifier ---")
    classifier.predict(val_x)

    # Validation of Classifier
    print("--- 6. Evaluating Neural Network Classifier ---")
    classifier.evaluate_architecture(test_x, test_y)

    # Perform Search of Best Hyperparameters
    print("--- 7. Finding Optimal Hyperparameters of Neural Network Classifier ---")
    best_hp = ClaimClassifierHyperParameterSearch()
    epoch, batchsize, learnrate, layers, neurons = best_hp['Epoch'], best_hp['BatchSize'], best_hp['LearningRate'], best_hp['Layers'], best_hp['Neurons']
    print("--- Car Insurance Prediction Model Optimal Hyperparameters -- ")
    print("Epoch = " + str(epoch))
    print("Batch Size = " + str(batchsize))
    print("Learning Rate = " + str(learnrate))
    print("Number of Layers = " + str(layers))
    print("Number of Neurons = " + str(neurons))

    print("--- END ---")
