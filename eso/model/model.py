from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from .cnn import BaseCNN
import numpy as np
import torch
from copy import deepcopy
import os
from pathlib import Path
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress only UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)



# TODO MAKE THIS CONFIGURABLE

conv_layers = 1
fc_layers = 2
max_pooling_size = 4
dropout_rate = 0.5
conv_filters = 8
conv_kernel = 8
fc_units = 32
epochs = 2
batch_size = 3

# conv_layers = 1
# fc_layers = 1
# max_pooling_size = 4
# dropout_rate = 0.4
# conv_filters = 8
# conv_kernel = 16
# fc_units = 32
# epochs = 40
# batch_size = 8


CNN_ARCHITECTURE = {
    "conv_layers": conv_layers,
    "conv_filters": conv_filters,
    "dropout_rate": dropout_rate,
    "conv_kernel": conv_kernel,
    "max_pooling_size": max_pooling_size,
    "fc_units": fc_units,
    "fc_layers": fc_layers,
}


CHROMOSOME_CNN_ARCHITECTURE = CNN_ARCHITECTURE.copy()


class Model:
    """Model class."""

    def __init__(
        self,
        results_path,
        input_shape,
        optimizer_name: str,
        loss_function_name: str,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        metric: str,
        architecture_args: dict,
        shuffle: bool = True,
        logger =None,
        use_chromosome=False,  #keep in case we allow different architecture for chromosome
        patience=3, min_delta=0.005
    ):

        
        architecture = architecture_args
        self.cnn = BaseCNN(input_shape=input_shape, **architecture)
        architecture = architecture.copy()
        architecture["input_shape"] = input_shape
        self._architecture = architecture

        # self.logger.info("Initializing Model...")
        # Get Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #to save the model 
        self.results_path=results_path
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.loss_name = loss_function_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_epochs = num_epochs
        self.logger = logger
        self.metric = metric

        self._set_optimizer_and_loss()

        #earlystopping
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    @staticmethod
    def load_cnn(cnn_dict, device):
        """
        Load the model from a saved state dictionary of the CNN.

        Parameters
        ----------
        cnn_dict_path : str
            Path to the saved cnn model dictionary.

        Returns
        -------
        Model
            The loaded model.
        """
        # Check if its a path or a dictionary
        if type(cnn_dict) == dict:
            dictionary = cnn_dict
        else:
            if os.path.exists(cnn_dict):
                dictionary = torch.load(cnn_dict, map_location=device)
            else:
                raise FileNotFoundError(f"Model file {cnn_dict} not found")
        cnn = BaseCNN(**dictionary["architecture"])
        cnn.load_state_dict(dictionary["state_dict"])
        return cnn
    
    
    @staticmethod
    def load(self, model_dict):
        self.cnn = BaseCNN(**model_dict["architecture"])
        self.cnn.load_state_dict(model_dict["state_dict"])

  


    def get_model_dict(self):
        return {"state_dict": self.cnn.state_dict(), "architecture": self._architecture}

    def _set_optimizer_and_loss(self):
        """Set the optimizer and loss function"""
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.cnn.parameters(), lr=self.learning_rate
            )
        else:
            raise NotImplementedError("Only Adam optimizer is supported at the moment")

        if self.loss_name == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                "Only cross entropy loss is supported at the moment"
            )

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)

    def get_minimum_input_shape(self):
        return self.cnn.calculate_min_input_size()

    def _create_dataloader(
        self, X: np.array, Y: np.array
    ) -> torch.utils.data.DataLoader:
        """Create a dataloader from the given data

        Parameters
        ----------
        X : np.array
            Input data of shape (n_samples,height,width) or (n_samples,channels,height,width). Will add channel dimension if needed.
        Y : np.array
            Target data of shape (n_samples,).

        Returns
        -------
        loader : torch.utils.data.DataLoader
            Dataloader with the given data and batch size specified in the constructor.

        """
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).float()

        # Reshape X_tensor
        if len(X_tensor.shape) == 3:
            X_tensor = X_tensor.unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        return loader
    
    def _early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def _train_one_epoch(self, dataloader):
        self.cnn.train()
        self.cnn.to(self.device)
        running_loss = 0.0

        num_batches = len(dataloader)
        num_samples = len(dataloader.dataset)

        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(self.device
                ), batch_targets.to(self.device)
            # Reset gradients
            self.optimizer.zero_grad()
            # Forward pass
            batch_preds = self.cnn.forward(batch_inputs)
            # Compute loss
            loss = self.criterion(batch_preds, batch_targets)
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            # Calculate Loss
            running_loss += loss.item() * batch_inputs.size(0)
        
        epoch_loss = running_loss / num_samples
        return epoch_loss
    
    def _val_one_epoch(self, dataloader):
        self.cnn.eval()
        self.cnn.to(self.device)
        num_batches = len(dataloader)
        num_samples = len(dataloader.dataset)
    
        with torch.no_grad():
            running_loss = 0.0

            for batch_inputs, batch_targets in dataloader:
                batch_inputs, batch_targets = batch_inputs.to(
                    self.device
                ), batch_targets.to(self.device)

                # Clear the gradients
                self.optimizer.zero_grad()
                # Forward pass
                batch_preds = self.cnn.forward(batch_inputs)
                # Compute loss
                loss = self.criterion(batch_preds, batch_targets)
                # Calculate Loss
                running_loss += loss.item() * batch_inputs.size(0)
      
            epoch_loss = running_loss / num_samples

        return epoch_loss

    def train(self, X_train, Y_train, X_val, Y_val, save=True, model_name="baseline", save_path=None, verbose=False):
        # Create Dataloaders
        train_loader = self._create_dataloader(X_train, Y_train)
        val_loader = self._create_dataloader(X=X_val, Y=Y_val)
        val_losses=[]
        train_losses = []
        min_val_loss = torch.inf

        for epoch in range(self.n_epochs):
            epoch_train_loss=self._train_one_epoch(train_loader)
            train_losses.append(epoch_train_loss)
            
            epoch_val_loss = self._val_one_epoch(val_loader)
            val_losses.append(epoch_val_loss)

            if save: 
                if min_val_loss > epoch_val_loss:
                    min_val_loss = epoch_val_loss
                    # Saving State Dict
                    self.save_model(self.results_path, model_name)
            

            if self._early_stop(epoch_val_loss):             
                break
        
        
        
        return train_losses, val_losses

    def evaluate(self, X_val, Y_val, metric=None, threshold=None, print_report=False):
        if metric is None:
            metric = self.metric
        loader = self._create_dataloader(X=X_val, Y=Y_val)
        self.cnn.eval()

        with torch.no_grad():
            total_loss = 0
            targets = []
            predictions = []
            for batch_inputs, batch_targets in loader:
                batch_inputs, batch_targets = batch_inputs.to(
                    self.device
                ), batch_targets.to(self.device)
                # print("targets: ", batch_targets)
                batch_preds = self.cnn.forward(batch_inputs)
                total_loss += self.criterion(batch_preds, batch_targets).item()
                
                #Predict true label if probability is greater than 0.7
                if threshold is not None:
                    prediction = (batch_preds > threshold).float()
                else:
                    prediction = batch_preds.argmax(dim=1).cpu()
               
                target = batch_targets.argmax(dim=1).cpu()
                # print("model prediction: ", batch_preds)
                # print("Prediction: ", prediction)
                # print("Target: ", target)

                predictions.extend(prediction.detach().numpy())
                targets.extend(target.detach().numpy())

            f1 = f1_score(targets, predictions)
            report = classification_report(targets, predictions)
            confusion = confusion_matrix(targets, predictions)
            if print_report:
                print(report)
                print(confusion)
            # print("F1: ", f1)
            # print("true positives: ", np.sum(np.logical_and(targets, predictions)))
            # print("true negatives: ", np.sum(np.logical_and(np.logical_not(targets), np.logical_not(predictions))))
            # print("false positives: ", np.sum(np.logical_and(np.logical_not(targets), predictions)))
            # print("false negatives: ", np.sum(np.logical_and(targets, np.logical_not(predictions))))
            accuracy = accuracy_score(targets, predictions)

        if metric == "f1":
            return (f1, "F1")
        elif metric == "accuracy":
            return (accuracy, "Accuracy")

    def save_model(self, path, model_name):
        save_path = os.path.join(Path(path, model_name + "_cnn_state.pth"))
        self._model_state_dict = deepcopy(self.get_model_dict())
        torch.save(self._model_state_dict, save_path)
        self.logger.info(f"CNN model state dict saved to {save_path}!")

    def __call__(self, x):
        return self.cnn(x)

    def __str__(self):
        return str(self.cnn)

    def __repr__(self):
        return str(self.cnn)
