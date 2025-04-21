import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from eso.model.model import Model


@pytest.fixture
def sample_input_shape():
    """Sample input shape for model initialization."""
    return (1, 28, 28)  # Single channel, 28x28 image


@pytest.fixture
def sample_model_params():
    """Common model parameters for tests."""
    return {
        "optimizer_name": "adam",
        "loss_function_name": "cross_entropy",
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "metric": "accuracy",
        "shuffle": True,
    }


@pytest.fixture
def sample_model(sample_input_shape, sample_model_params):
    """Create a sample model for testing."""
    return Model(input_shape=sample_input_shape, **sample_model_params)


@pytest.fixture
def sample_data():
    """Create sample data for training and evaluation."""
    # Generate 100 samples of 28x28 images with single channel
    X = np.random.rand(100, 28, 28).astype(np.float32)
    # Create one-hot encoded targets (assuming binary classification)
    Y = np.zeros((100, 2), dtype=np.float32)
    Y[np.arange(100), np.random.randint(0, 2, 100)] = 1.0
    return X, Y


class TestModelInitialization:
    def test_init(self, sample_input_shape, sample_model_params):
        """Test model initialization."""
        model = Model(input_shape=sample_input_shape, **sample_model_params)
        assert model.batch_size == sample_model_params["batch_size"]
        assert model.learning_rate == sample_model_params["learning_rate"]
        assert model.n_epochs == sample_model_params["num_epochs"]
        assert model.metric == sample_model_params["metric"]
        assert model.shuffle == sample_model_params["shuffle"]
        assert model.optimizer_name == sample_model_params["optimizer_name"]
        assert model.loss_name == sample_model_params["loss_function_name"]
        assert isinstance(model.cnn, torch.nn.Module)

    def test_init_with_chromosome(self, sample_input_shape, sample_model_params):
        """Test model initialization with chromosome architecture."""
        model = Model(
            input_shape=sample_input_shape, use_chromosome=True, **sample_model_params
        )
        assert model.batch_size == sample_model_params["batch_size"]
        assert isinstance(model.cnn, torch.nn.Module)

    def test_optimizer_initialization(self, sample_model):
        """Test optimizer is correctly initialized."""
        assert isinstance(sample_model.optimizer, torch.optim.Adam)

    def test_loss_initialization(self, sample_model):
        """Test loss function is correctly initialized."""
        assert isinstance(sample_model.criterion, torch.nn.CrossEntropyLoss)

    def test_unsupported_optimizer(self, sample_input_shape, sample_model_params):
        """Test unsupported optimizer raises exception."""
        with pytest.raises(NotImplementedError):
            Model(
                input_shape=sample_input_shape,  # type: ignore
                **{**sample_model_params, "optimizer_name": "sgd"},  # type: ignore
            )

    def test_unsupported_loss(self, sample_input_shape, sample_model_params):
        """Test unsupported loss raises exception."""
        with pytest.raises(NotImplementedError):
            Model(
                input_shape=sample_input_shape,
                **{**sample_model_params, "loss_function_name": "mse"},  # type: ignore
            )


class TestModelDataLoading:
    def test_create_dataloader(self, sample_model, sample_data):
        """Test dataloader creation."""
        X, Y = sample_data
        loader = sample_model._create_dataloader(X, Y)
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == sample_model.batch_size

        # Test first batch
        inputs, targets = next(iter(loader))
        assert (
            inputs.shape[0] <= sample_model.batch_size
        )  # May be smaller for last batch
        assert inputs.shape[1] == 1  # Channel dimension added
        assert inputs.shape[2:] == (28, 28)
        assert targets.shape[0] == inputs.shape[0]


class TestModelTrainingAndEvaluation:
    def test_train(self, sample_model, sample_data):
        """Test model training."""
        X, Y = sample_data
        losses = sample_model.train(X, Y)
        assert len(losses) > 0
        assert all(isinstance(loss, float) for loss in losses)

    def test_evaluate_accuracy(self, sample_model, sample_data):
        """Test model evaluation with accuracy metric."""
        X, Y = sample_data
        sample_model.train(X, Y)  # Train first
        score, metric_name = sample_model.evaluate(X, Y, metric="accuracy")
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert metric_name == "Accuracy"

    def test_evaluate_f1(self, sample_model, sample_data):
        """Test model evaluation with F1 metric."""
        X, Y = sample_data
        sample_model.train(X, Y)  # Train first
        score, metric_name = sample_model.evaluate(X, Y, metric="f1")
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert metric_name == "F1"

    def test_evaluate_invalid_metric(self, sample_model, sample_data):
        """Test evaluation with invalid metric raises exception."""
        X, Y = sample_data
        with pytest.raises(ValueError):
            sample_model.evaluate(X, Y, metric="invalid_metric")

    def test_evaluate_with_threshold(self, sample_model, sample_data):
        """Test evaluation with threshold parameter."""
        X, Y = sample_data
        sample_model.train(X, Y)
        with pytest.raises(ValueError, match="mix of"):
            sample_model.evaluate(X, Y, threshold=0.5)


class TestModelSaveLoad:
    def test_get_model_dict(self, sample_model):
        """Test getting model dictionary."""
        model_dict = sample_model.get_model_dict()
        assert "state_dict" in model_dict
        assert "architecture" in model_dict

    @pytest.mark.parametrize("use_dict", [True, False])
    def test_load_cnn(self, sample_model, use_dict, tmp_path):
        """Test loading CNN from dictionary or path."""
        # Get model dict
        model_dict = sample_model.get_model_dict()

        if use_dict:
            # Load directly from dict
            cnn = Model.load_cnn(model_dict, sample_model.device)
        else:
            # Save to file and load from path
            save_path = tmp_path / "test_model.pth"
            torch.save(model_dict, save_path)
            cnn = Model.load_cnn(str(save_path), sample_model.device)

        assert isinstance(cnn, torch.nn.Module)

    def test_load_cnn_file_not_found(self, sample_model):
        """Test loading CNN with non-existent file raises exception."""
        with pytest.raises(FileNotFoundError):
            Model.load_cnn("non_existent_path.pth", sample_model.device)

    def test_load(self, sample_model):
        """Test loading model state."""
        model_dict = sample_model.get_model_dict()
        new_model = Model(
            input_shape=sample_model._architecture["input_shape"],
            optimizer_name="adam",
            loss_function_name="cross_entropy",
            batch_size=32,
            learning_rate=0.001,
            num_epochs=5,
            metric="accuracy",
        )
        new_model.load(model_dict)

        # Check state_dicts are equal
        for (k1, v1), (k2, v2) in zip(
            sample_model.cnn.state_dict().items(), new_model.cnn.state_dict().items()
        ):
            assert k1 == k2
            assert torch.equal(v1, v2)

    @patch("torch.save")
    def test_save_model(self, mock_save, sample_model, tmp_path):
        """Test saving model."""
        path = str(tmp_path)
        model_name = "test_model"

        # Mock logger
        sample_model.logger = MagicMock()

        sample_model.save_model(path, model_name)

        # Check torch.save was called
        mock_save.assert_called_once()

        # Check logger was called
        sample_model.logger.info.assert_called_once()


class TestModelUtilities:
    def test_get_number_of_parameters(self, sample_model):
        """Test getting number of parameters."""
        num_params = sample_model.get_number_of_parameters()
        assert isinstance(num_params, int)
        assert num_params > 0

    def test_get_minimum_input_shape(self, sample_model):
        """Test getting minimum input shape."""
        min_shape = sample_model.get_minimum_input_shape()
        assert isinstance(min_shape, tuple)
        assert len(min_shape) == 2  # Height and width

    def test_call(self, sample_model):
        """Test calling model directly."""
        # Create a dummy input tensor
        input_tensor = torch.rand(1, 1, 28, 28)
        output = sample_model(input_tensor)
        assert isinstance(output, torch.Tensor)

    def test_str_repr(self, sample_model):
        """Test string representation of model."""
        assert isinstance(str(sample_model), str)
        assert isinstance(repr(sample_model), str)
