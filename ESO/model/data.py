from ..utils.preprocessing import Preprocessing
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from ..utils.logger import setup_logger
from pathlib import Path


class Data:
    """Class for handling the data. This class will be used by the ESO class.

    It initializes the preprocessing class and handles the creation of the
    dataset based on the preprocessing settings.
    """

    def __init__(
        self,
        apply_preprocessing: bool,
        force_recreate_dataset: bool,
        species_folder: str,
        keep_in_memory: bool,
        preprocessing_args: dict,
        train_size: float,
        test_size: float,
        positive_class: str,
        negative_class: str,
        reshuffle: bool = False,
        logger=None,
        log_path=None,
        log_level=10,
    ) -> None:
        """Initialize the Data class

        Parameters
        ----------
        config : dict
            The config dictionary containing the settings for preprocessing
        logger : logging.Logger
            The logger object to log messages

        Returns
        -------
        None

        """

        # This should only contain the confg for data settings
        self.logger = setup_logger(
            logger=logger, log_path=log_path, log_level=log_level
        )
        self._positive_class = positive_class
        self._negative_class = negative_class
        self._preprocessing_flag = apply_preprocessing
        self._force_recreate_dataset = force_recreate_dataset
        self.species_folder = species_folder
        self._keep_in_memory = keep_in_memory
        self._train_size = train_size
        self._reshuffle = reshuffle
        self._test_size = test_size
        self.preprocessing_args = preprocessing_args

    def create_datasets(self):
        types = ["train", "validation", "test"]
        # self._shuffle_files_names()
        if self._preprocessing_flag:
            preproces_name = "preprocessed"
        else:
            preproces_name = "unpreprocessed"
        self.save_path = Path(self.species_folder, "SavedData", preproces_name)
        preprocessing = Preprocessing(
            **self.preprocessing_args,
            apply_preprocessing=self._preprocessing_flag,
            species_folder=self.species_folder,
            positive_class=self._positive_class,
            negative_class=self._negative_class,
        )

        train_path = Path(self.species_folder, "DataFiles", "train.txt")
        validation_path = Path(self.species_folder, "DataFiles", "validation.txt")
        test_path = Path(self.species_folder, "DataFiles", "test.txt")
        if (
            os.path.exists(train_path)
            and os.path.exists(validation_path)
            and os.path.exists(test_path)
        ):
            # This means the files have already been shuffled,
            # check if they should be reshuffled egein
            if self._reshuffle:
                self.logger.info(
                    "Found Existing Files but reshuffle flag ist set. Reshuffling.."
                )
                preprocessing._shuffle_files_names(
                    train_size=self._train_size, test_size=self._test_size
                )
            else:
                self.logger.info(
                    "Found already existing shuffled file names! Loading from memory.."
                )
        else:
            # Files dont exist, create the split
            self.logger.info("Reshuffling file names for the first time...")
            preprocessing._shuffle_files_names(
                train_size=self._train_size, test_size=self._test_size
            )
        for type in types:
            save_type_path = str(Path(self.save_path) / type)
            # Check if the dataset already exists
            if (
                os.path.exists(Path(save_type_path, "X.pkl"))
                and not self._force_recreate_dataset
            ):
                self.logger.info("The dataset already exists. Skipping...")
                if not hasattr(self, "image_shape"):
                    # Load the dataset to get the image shape
                    self.logger.debug("Loading dataset to set image shape...")
                    X, Y = self._load_dataset(type)
                    self.image_shape = X.shape[1:]
                continue

            # Create the folder
            os.makedirs(save_type_path, exist_ok=True)
            path = Path(self.species_folder, "DataFiles", f"{type}.txt")
            self.logger.debug("File path: " + str(path))
            if type == "train":
                self.logger.info("Creating the training dataset")
                # Create the dataset WITH augmentation
                X, Y = preprocessing.create_dataset(
                    verbose=False,
                    file_names=path,
                    augmentation=True,
                    annotation_folder="Annotations",
                    sufix_file=".svl",
                )
            else:
                self.logger.info("Creating the validation dataset")
                X, Y = preprocessing.create_dataset(
                    verbose=False,
                    file_names=path,
                    augmentation=False,
                    annotation_folder="Annotations",
                    sufix_file=".svl",
                )

            if not hasattr(self, "image_shape"):
                self.image_shape = X.shape[1:]

            # Check if the dataset is empty
            if Y.shape[0] == 0:
                raise Exception("The dataset is empty. Please check the data files.")
            Y = self._one_hot_encode(Y)

            if not os.path.exists(Path(self.save_path, "encoded_mapping.txt")):
                # Save encoded mapping as text file
                encoded_mapping = self.get_encoded_mapping()
                with open(Path(self.save_path, "encoded_mapping.txt"), "w") as f:
                    f.write(str(encoded_mapping))

            # Save the dataset
            with open(Path(save_type_path, "X.pkl"), "wb") as f:
                pickle.dump(X, f)
            with open(Path(save_type_path, "Y.pkl"), "wb") as f:
                pickle.dump(Y, f)

            self.logger.info(
                "Dataset created and saved at " + save_type_path + "/X.pkl"
            )
        self._distribution = preprocessing.check_distribution(Y)

    def get_image_shape(self) -> tuple:
        """Returns the shape of one image"""
        return self.image_shape

    def get_data(self, type="train") -> tuple:
        """Returns the dataset
        Returns
        -------
        X : ndarray
            The Images
        Y : ndarray
            The labels
        """
        path = Path(self.save_path, type)
        # Check if the dataset exists
        if not os.path.exists(os.path.join(path, "X.pkl")):
            raise Exception(
                "The dataset does not exist. Please create the dataset first."
            )
        # Check keep in memory flag
        if self._keep_in_memory:
            # Check if the dataset is already loaded
            if not hasattr(self, "_X"):
                self.logger.debug("Loading dataset into memory...")
                self._X, self._Y = self._load_dataset(type)
            else:
                self.logger.debug("Dataset already loaded into memory.")
            X = self._X
            Y = self._Y
        else:
            self.logger.debug("Loading dataset...")
            X, Y = self._load_dataset(type)
        return X, Y

    def _one_hot_encode(self, Y):
        # Check if the encoder is fitted
        # Reshape the labels
        Y = Y.reshape(-1, 1)
        if not hasattr(self, "_encoder"):
            # Fit the encoder
            self._encoder = OneHotEncoder(
                categories=[[self._negative_class, self._positive_class]]
            )
            self._encoder.fit(Y)
        # Encode the labels
        Y = self._encoder.transform(Y)
        # Convert to numpy array
        Y = Y.toarray()
        return Y

    def _one_hot_decode(self, Y):
        # Check if the encoder is fitted
        if not hasattr(self, "_encoder"):
            # Fit the encoders
            self._encoder = OneHotEncoder(
                categories=[[self._negative_class, self._positive_class]]
            )
            self._encoder.fit(Y)
        # Decode the labels
        Y = self._encoder.inverse_transform(Y)
        # Convert to numpy array
        Y = Y.toarray()
        # Reshape the labels
        Y = Y.flatten()
        return Y

    def get_encoded_mapping(self):
        """Returns the encoded mapping of the labels"""
        # Check if the encoder is fitted
        if not hasattr(self, "_encoder"):
            if os.path.exists(Path(self.save_path, "encoded_mapping.txt")):
                with Path(self.save_path, "encoded_mapping.txt").open("r") as file:
                    encoded_mapping = file.read()
                return encoded_mapping
            else:
                raise Exception(
                    "The encoder is not fitted and no file found. Please fit the encoder first."
                )
        # The categories are stored in a list of lists
        categories = self._encoder.categories_[0]
        # Create a dictionary of the categories
        categories_one_hot = self._encoder.transform(
            categories.reshape(-1, 1)
        ).toarray()
        categories_dict = dict(zip(categories, categories_one_hot))
        return categories_dict

    def _load_dataset(self, type="train") -> tuple:
        """Loads the dataset from the save path"""
        # Load the dataset
        with open(Path(self.save_path, type, "X.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(Path(self.save_path, type, "Y.pkl"), "rb") as f:
            Y = pickle.load(f)
        self.logger.info("Dataset loaded from " + str(self.save_path) + f"/{type}")
        return X, Y
