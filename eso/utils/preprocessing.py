from glob import glob
import os
import numpy as np
import random
import librosa
from scipy import signal
from random import randint
import pickle
import pandas as pd
from pathlib import Path

from .AnnotationReader import *


class Preprocessing:
    def __init__(
        self,
        species_folder : str,
        sample_rate: int,
        lowpass_cutoff : int,
        downsample_rate : int,
        nyquist_rate : int,
        segment_duration : int,
        positive_class : str,
        negative_class : str,
        nb_negative_class : int,
        n_fft : int,
        hop_length : int,
        n_mels : int,
        f_min : int,
        f_max : int,
        file_type : str,
        audio_extension : str,
        apply_preprocessing: bool=True,
        
    ) -> None:
        """
        Initialize the Preprocessing object.

        Parameters
        ----------
        species_folder : str
            Path to the species folder containing audio and annotation data.
        sample_rate : int
            The sample rate for unprocessed audio files.
        lowpass_cutoff : int
            The cutoff frequency for the low-pass filter.
        downsample_rate : int
            The rate at which to downsample the audio.
        nyquist_rate : int
            The Nyquist rate, half of the sampling rate.
        segment_duration : int
            Duration of each audio segment in seconds.
        positive_class : str
            Label representing the positive class in the dataset.
        negative_class : str
            Label representing the negative class in the dataset.
        nb_negative_class : int
            Number of negative class samples.
        n_fft : int
            The length of the FFT window for spectrograms.
        hop_length : int
            The hop length for generating spectrograms.
        n_mels : int
            The number of mel bands to use in the spectrogram.
        f_min : int
            The minimum frequency for the mel filter bank.
        f_max : int
            The maximum frequency for the mel filter bank.
        file_type : str
            The type of annotation files to process (e.g., '.svl').
        audio_extension : str
            The file extension for the audio files (e.g., '.wav').
        apply_preprocessing : bool, optional
            Whether to apply preprocessing steps like filtering and downsampling. Default is True.

        Returns
        -------
        None
        """
        self.sample_rate_unpreprocessed=sample_rate
        self.species_folder = species_folder
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.nb_negative_class = nb_negative_class
        self.audio_path = Path(self.species_folder, "Audio")
        self.annotations_path = Path(self.species_folder, "Annotations")
        self.saved_data_path = Path(self.species_folder, "Saved_Data")
        self.training_files = Path(self.species_folder, "DataFiles", "TrainingFiles.txt")      
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.file_type = file_type
        self.audio_extension = audio_extension
        self.apply_preprocessing = apply_preprocessing
        self.n_fft = n_fft
        self.hop_length = hop_length

    def read_audio_file(self, file_name):
        """
        Load an audio file and return its waveform and sample rate.

        Parameters
        ----------
        file_name : str
            Name of the audio file including the extension (e.g., "audio1.wav").

        Returns
        -------
        tuple
            A tuple containing:
            - np.ndarray: The audio waveform (amplitude values).
            - int: The sampling rate of the audio file.
        """
        # Get the path to the file
        audio_folder = Path(file_name)

        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)

        return audio_amps, audio_sample_rate

    def _butter_lowpass(self, cutoff, nyq_freq, order=4):
        """
        Design a Butterworth low-pass filter.

        This method computes the filter coefficients for a Butterworth low-pass filter
        given the desired cutoff frequency and the Nyquist frequency.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the low-pass filter (in Hz).
        nyq_freq : float
            The Nyquist frequency (typically half the sampling rate).
        order : int, optional
            The order of the filter. Higher orders result in a steeper roll-off. 
            Default is 4.

        Returns
        -------
        tuple
            A tuple (b, a) of filter coefficients to be used with `scipy.signal.lfilter`.
        """
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype="lowpass")
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        """
        Apply a Butterworth low-pass filter to the input signal.

        This method filters the input signal using a zero-phase Butterworth low-pass
        filter designed with the specified cutoff and Nyquist frequencies.

        Parameters
        ----------
        data : np.ndarray
            The input signal (1D array) to be filtered.
        cutoff_freq : float
            The cutoff frequency of the low-pass filter (in Hz).
        nyq_freq : float
            The Nyquist frequency (typically half the sampling rate).
        order : int, optional
            The order of the Butterworth filter. Default is 4.

        Returns
        -------
        np.ndarray
            The filtered signal with the same shape as the input.
        """ 
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        b, a = self._butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        """
        Downsample an audio waveform to a specified sample rate.

        This function resamples the input audio from the original sample rate
        to a new, lower sample rate using the 'kaiser_fast' resampling method.

        Parameters
        ----------
        amplitudes : np.ndarray
            The raw audio waveform (1D NumPy array of amplitude values).
        original_sr : int
            The original sampling rate of the audio signal (in Hz).
        new_sample_rate : int
            The desired sampling rate to downsample the audio to (in Hz).

        Returns
        -------
        tuple
            A tuple containing:
            - np.ndarray: The downsampled audio waveform.
            - int: The new sampling rate (same as `new_sample_rate`).
        """
        return (
            librosa.resample(
                amplitudes,
                orig_sr=original_sr,
                target_sr=new_sample_rate,
                res_type="kaiser_fast",
            ),
            new_sample_rate,
        )

    def convert_single_to_image(self, audio, sample_rate):
        """
        Convert an audio waveform into a normalized mel-spectrogram image.

        This function computes the mel-spectrogram from a raw audio signal and 
        applies normalization to scale the spectrogram values between 0 and 1.
        If preprocessing is enabled, user-defined frequency limits are used;
        otherwise, default frequency bounds are applied.

        Parameters
        ----------
        audio : np.ndarray
            The raw audio waveform (1D NumPy array of amplitude values).
        sample_rate : int
            The sampling rate of the audio signal (in Hz).

        Returns
        -------
        np.ndarray
            A 2D NumPy array representing the normalized mel-spectrogram image.
        """
        if not self.apply_preprocessing:
            f_min = 0
            f_max = 11000
        else:
            f_min = self.f_min
            f_max = self.f_max

        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=f_min,
            fmax=f_max,
        )

        
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = image_np - np.min(image_np)
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps = 1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled

        return S1

    def _convert_all_to_image(self, segments, sample_rate):
        """
        Convert multiple audio segments into their corresponding mel-spectrogram images.

        This method applies `convert_single_to_image` to each segment in the input
        and returns a NumPy array of spectrograms.

        Parameters
        ----------
        segments : list or np.ndarray
            A list or array of 1D audio waveforms (amplitude values), one per segment.
        sample_rate : int
            The sampling rate of the audio segments (in Hz).

        Returns
        -------
        np.ndarray
            A 3D NumPy array of shape (n_segments, n_mels, time_steps) containing the
            normalized spectrogram images for each audio segment.
        """
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment, sample_rate))

        return np.array(spectrograms)


    def _getXY(
        self,
        audio_amplitudes,
        sampling_rate,
        start_sec,
        annotation_duration_seconds,
        label
    ):
        """
        Extract audio segments of an audio file based on the user-annotations.

        This method slices the audio based on annotation start time and duration,
        shifting the window by 1 second for each segment. Each extracted segment 
        is assigned the provided label. 

        Parameters
        ----------
        audio_amplitudes : np.ndarray
            The raw audio waveform (amplitude values).
        sampling_rate : int
            Sampling rate of the audio file (in Hz).
        start_sec : int or float
            The starting time (in seconds) of the annotated event.
        annotation_duration_seconds : int or float
            The total duration (in seconds) of the annotated event.
        label : str
            The class label associated with the annotation. 

        Returns
        -------
        tuple
            A tuple containing:
            - X_segments (list of np.ndarray): List of audio segments extracted.
            - Y_labels (list of str): Corresponding list of labels for each segment.

        Notes
        -----
        - The number of segments extracted depends on the annotation duration and 
        the configured segment duration (`self.segment_duration`).
        - For negative class labels, the number of extracted segments is limited to 
        `self.nb_negative_class` if applicable.
        - Segments that would exceed the length of the audio signal are skipped.
        """
        X_segments = []
        Y_labels = []

        # Calculate how many segments can be extracted based on the duration of
        # the annotated duration. If the annotated duration is too short then
        # simply extract one segment. If the annotated duration is long enough
        # then multiple segments can be extracted.
        if annotation_duration_seconds - self.segment_duration < 0:
            segments_to_extract = 1
        else:
            segments_to_extract = (
                annotation_duration_seconds - self.segment_duration + 1
            )

        if label in self.negative_class:
            if segments_to_extract > self.nb_negative_class:
                segments_to_extract = self.nb_negative_class

        for i in range(0, segments_to_extract):
            # Set the correct location to start with.
            # The correct start is with respect to the location in time
            # in the audio file start+i*sample_rate
            start_data_observation = start_sec * sampling_rate + i * (sampling_rate)
            # The end location is based off the start
            end_data_observation = start_data_observation + (
                sampling_rate * self.segment_duration
            )

            # This case occurs when something is annotated towards the end of a file
            # and can result in a segment which is too short.
            if end_data_observation > len(audio_amplitudes):
                continue

            # Extract the segment of audio
            X_audio = audio_amplitudes[start_data_observation:end_data_observation]

            # Determine the actual time for the event
            start_time_seconds = start_sec + i

            # Extend the augmented segments and labels
            X_segments.append(X_audio)
            Y_labels.append(label)

        return X_segments, Y_labels

    def save_data_to_pickle(self, X, Y):
        """
        Save the input data and labels to pickle files.

        This function saves the spectrogram data (`X`) and their corresponding
        labels (`Y`) into separate pickle files (`X.pkl` and `Y.pkl`) in the directory 
        specified by `self.saved_data_path`.

        Parameters
        ----------
        X : any
            The data to be saved (e.g., spectrograms). Must be pickle-serializable.
        Y : any
            The corresponding labels for `X`. Must also be pickle-serializable.

        Returns
        -------
        None
        """
        outfile = open(Path(self.saved_data_path, "X.pkl"), "wb")
        pickle.dump(X, outfile, protocol=4)
        outfile.close()

        outfile = open(Path(self.saved_data_path, "Y.pkl"), "wb")
        pickle.dump(Y, outfile, protocol=4)
        outfile.close()

    def load_data_from_pickle(self):
        """
        Load the data and labels from pickle files.

        This function loads spectrogram data (`X`) and their corresponding
        labels (`Y`) from pickle files (`X.pkl` and `Y.pkl`) located in the directory 
        specified by `self.saved_data_path`.

        Returns
        -------
        X : any
            The loaded data (e.g., spectrograms), as previously saved using `save_data_to_pickle`.
        Y : any
            The corresponding labels for `X`.
        """
        infile = open(Path(self.saved_data_path, "X.pkl"), "rb")
        X = pickle.load(infile)
        infile.close()

        infile = open(Path(self.saved_data_path, "Y.pkl"), "rb")
        Y = pickle.load(infile)
        infile.close()

        return X, Y

    def _time_shifting(self, X, index):
        """
        Augment a segment of amplitude values by applying a time shift.

        Args:
            X (ndarray): Array of amplitude values.
            X_meta (ndarray): Array of corresponding metadata.
            index (list): List of indices of the files to choose from.

        Returns:
            tuple: Augmented segment and its metadata.
        """
        # Convert index to list
        index = list(index)
        # Randomly select an index from the given index list
        idx_pickup = random.sample(index, 1)

        # Retrieve the segment and metadata corresponding to the selected index
        segment = X[idx_pickup][0]

        # Randomly select time into the segments
        random_time_point_segment = randint(1, len(segment) - 1)

        # Apply time shift to the segment
        segment = self._time_shift(segment, random_time_point_segment)

        return segment

    def _time_shift(self, audio, shift):
        """
        Shift amplitude values of the waveform to the right.

        This function shifts the audio waveform to the right by a specified number of samples. 
        The values that overflow on the right are wrapped around to the left side.

        Parameters
        ----------
        audio : ndarray
            Array of amplitude values representing the audio waveform.
        shift : int
            Number of samples to shift the waveform to the right.

        Returns
        -------
        ndarray
            The time-shifted audio waveform with wrapped-around amplitude values.
        """
        augmented = np.zeros(len(audio))
        augmented[0:shift] = audio[-shift:]
        augmented[shift:] = audio[:-shift]

        return augmented

    def _combining(self, X, index, index_negative_class):
        """
        Combine two segments to create an augmented segment.

        This function randomly selects one segment from the `index` list and another from the
        `index_negative_class` list, then blends them using fixed weights (0.8 and 0.2) to produce
        a new augmented segment.

        Parameters
        ----------
        X : ndarray
            Input data containing segments, where each segment is accessible by index.
        index : list of int
            List of indices corresponding to the primary class from which to select one segment.
        index_negative_class : list of int
            List of indices corresponding to the negative class from which to select another segment.

        Returns
        -------
        ndarray
            Augmented segment created by blending two selected segments.
        """
        # Convert index to list
        index=list(index)
        index_negative_class=list(index_negative_class)
        
        # Randomly select an index from the given index list
        idx_pickup=random.sample(index, 1)
            
        # Randomly select another file to combine with
        idx_combining=random.sample(index_negative_class, 1)
        
        # combine the two files with different weights
        segment=self._blend(X[idx_pickup][0], X[idx_combining][0], 0.8, 0.2)
             
        return segment


    def _blend(self, audio_1, audio_2, w_1, w_2):
        """
        Blend two audio segments using specified weights.

        This function combines two audio waveforms by computing a weighted sum of the 
        two input segments. The resulting segment reflects the relative contributions 
        specified by the weights.

        Parameters
        ----------
        audio_1 : ndarray
            First audio segment to be blended.
        audio_2 : ndarray
            Second audio segment to be blended.
        w_1 : float
            Weight applied to `audio_1`.
        w_2 : float
            Weight applied to `audio_2`.

        Returns
        -------
        ndarray
            The resulting audio segment after blending.
        """
        augmented = w_1 * audio_1 + w_2 * audio_2
        return augmented
        
    def _add_noise_gaussian(self, X, index):
        """
        Add Gaussian noise to a randomly selected audio segment.

        This function selects a random segment from the specified indices and adds Gaussian 
        noise to it. The noise is generated with a mean of 0 and a standard deviation scaled 
        by 0.0009.

        Parameters
        ----------
        X : ndarray
            Input data containing audio segments.
        index : list of int
            List of indices from which to randomly select a segment.

        Returns
        -------
        ndarray
            The audio segment with added Gaussian noise.
        """
        # Convert index to list
        index=list(index)
           
        # Randomly select an index from the given index list
        idx_pickup=random.sample(index, 1)
        
        # Retrieve the segment corresponding to the selected index
        segment=X[idx_pickup][0]
       
        
        # Add Gaussian noise to the segment
        segment=segment+ 0.0009*np.random.normal(0,1,len(segment))
            

        return segment

    

    def _augment_dataset(self, X, Y):
        """
        Perform class balancing through data augmentation.

        This function balances an imbalanced dataset by augmenting the minority class and/or 
        reducing the majority class. Augmentation methods include time shifting, adding Gaussian 
        noise, and blending with segments from a negative class. If the class imbalance is too large,
        it also subsamples the majority class.

        Parameters
        ----------
        X : array-like
            Input audio segments.
        Y : array-like
            Corresponding class labels for each audio segment.

        Returns
        -------
        tuple of list
            Augmented dataset as a tuple (X_augmented, Y_augmented), where:
            - X_augmented is a list of augmented audio segments,
            - Y_augmented is the corresponding list of labels.
        """      
        
        X = np.asarray(X)
        
        label_to_augment = np.argmin(np.unique(np.asarray(Y), return_counts=True)[1])
        label_second_class=np.argmax(np.unique(np.asarray(Y), return_counts=True)[1])
        difference = np.max(np.unique(np.asarray(Y), return_counts=True)[1]) - np.min(
            np.unique(np.asarray(Y), return_counts=True)[1]
        )
 
        index_to_augment = np.where(np.asarray(Y)== np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment])[0]
        index_second_class = np.where(np.asarray(Y)== np.unique(np.asarray(Y), return_counts=True)[0][label_second_class])[0]
        index_negative_class=np.where(np.asarray(Y)== self.negative_class)[0]
          
        X_augmented = []
        Y_augmented = []

        if difference > 2* np.min(np.unique(np.asarray(Y), return_counts=True)[1]):
            # Case 1: Large imbalance — reduce majority class and augment minority class
            number_to_select = 3* np.min(np.unique(np.asarray(Y), return_counts=True)[1])
            index_to_keep=np.array(random.sample(list(index_second_class), number_to_select))

            X_augmented.extend(X[index_to_keep])
            Y_augmented=[np.unique(np.asarray(Y), return_counts=True)[0][label_second_class]] * number_to_select
 
            nb_to_add= np.min(np.unique(np.asarray(Y), return_counts=True)[1]) 
            nb_to_augm_per_method=(nb_to_add//3)+1

            X_augmented.extend(X[index_to_augment])
            Y_augmented.extend([np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]] * len(index_to_augment))

            for j in range(0,nb_to_augm_per_method ):
                X_augmented.append(self._time_shifting(X, index_to_augment))
                X_augmented.append(self._add_noise_gaussian(X, index_to_augment))
                X_augmented.append(self._combining(X, index_to_augment,index_negative_class))
                Y_augmented.extend([np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]] * 3)

        else : 
            # Case 2: Moderate imbalance — only augment the minority class
            X_augmented.extend(X)
            Y_augmented.extend(Y)
            nb_to_augm_per_method=(difference//3)+1
            for i in range(0,nb_to_augm_per_method):
                X_augmented.append(self._time_shifting(X, index_to_augment))
                X_augmented.append(self._add_noise_gaussian(X, index_to_augment))
                X_augmented.append(self._combining(X, index_to_augment,index_negative_class))
                Y_augmented.extend([np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]] * 3)


        return X_augmented, Y_augmented

    def create_dataset(self, annotation_folder, sufix_file, file_names=None, augmentation=False):
        """
        Create the dataset of audio segments and labels for machine learning.

        This function reads audio files and their corresponding annotation files,
        applies preprocessing (optional low-pass filtering and downsampling),
        extracts labeled audio segments, and optionally augments the data to
        balance class distributions.

        Parameters
        ----------
        annotation_folder : str or Path
            Path to the folder containing the `.svl` annotation files.
        sufix_file : str
            Suffix to append to the annotation filenames for retrieval.
        file_names : str or Path, optional
            Path to a CSV file containing a list of filenames to process (without extensions).
            If None, uses `self.training_files`.
        augmentation : bool, optional
            Whether to perform data augmentation to balance the dataset.

        Returns
        -------
        tuple of np.ndarray
            - `X_calls` : ndarray of shape (n_samples, ...)
                Array of preprocessed and optionally augmented audio segments,
                typically converted into spectrogram images.
            - `Y_calls` : ndarray of shape (n_samples,)
                Corresponding class labels for each segment (binary or multi-class).
        
        Raises
        ------
        ValueError
            If the `file_names` CSV is missing or empty.
        
        Notes
        -----
        - Annotations are expected in `.svl` format, created with Sonic Visualiser,
        using the "boxes area" annotation layer.
        - Each annotation provides a labeled time segment which is then transformed
        into a training example.
        - Augmentation methods include time shifting, noise addition, and mixing
        with negative samples to improve dataset balance.
        """

        if file_names is None:
            file_names = self.training_files
        # Keep track of how many calls were found in the annotation files
        total_calls = 0

        # Initialise lists to store the X and Y values
        X_calls = []
        Y_calls = []

        # Read all names of the files
        try:
            files = pd.read_csv(file_names, header=None)
        except Exception:
            raise ValueError(
                f"Error loading filenames from {file_names}. Check if File is not empty."
            )
        # Iterate over each annotation file
        for file in files.values:
            file = file[0]

            file_name_no_extension = file

            reader = AnnotationReader(self.species_folder,file, self.file_type, self.audio_extension, self.positive_class
            )
            # Check if the audio file exists before processing
            if str(
                Path(self.audio_path, file_name_no_extension + self.audio_extension)
            ) in glob(str(self.audio_path / f"*{self.audio_extension}")):

                # Read audio file
                audio_amps, original_sample_rate = self.read_audio_file(
                    str(
                        Path(
                            self.audio_path,
                            file_name_no_extension + self.audio_extension,
                        )
                    )
                )
               
                if self.apply_preprocessing:
                    # Low pass filter
                    filtered = self.butter_lowpass_filter(
                        audio_amps, self.lowpass_cutoff, self.nyquist_rate
                    )
                    # Downsample
                    amplitudes, sample_rate = self._downsample_file(
                        filtered, original_sample_rate, self.downsample_rate
                    )
                    del filtered

                else:
                    
                    if original_sample_rate!=self.sample_rate_unpreprocessed: 
                        amplitudes, sample_rate = self._downsample_file(
                        audio_amps, original_sample_rate, self.sample_rate_unpreprocessed
                    )
                    else :
                        amplitudes, sample_rate = audio_amps, original_sample_rate
                    
                del audio_amps
                df, audio_file_name = reader.get_annotation_information(annotation_folder, sufix_file)

             
                for index, row in df.iterrows():
                    start_seconds = int(round(row["Start"]))
                    end_seconds = int(round(row["End"]))
                    label = row["Label"]
                    annotation_duration_seconds = end_seconds - start_seconds

                    # Extract augmented audio segments and corresponding binary labels
                    X_data, y_data = self._getXY(
                        amplitudes,
                        sample_rate,
                        start_seconds,
                        annotation_duration_seconds,
                        label
                    )

                    # Append the segments and labels
                    X_calls.extend(X_data)
                    Y_calls.extend(y_data)



        if augmentation:
            # Augment dataset to get a balance dataset
            X_calls, Y_calls = self._augment_dataset(X_calls, Y_calls)


        X_calls = self._convert_all_to_image(X_calls, sample_rate)

        # Convert to numpy arrays
        X_calls, Y_calls = np.asarray(X_calls), np.asarray(Y_calls)

        return X_calls, Y_calls

    def shuffle_files_names(self, train_size=0.8, test_size=0.1, validation_size=0.1):
        """
        Shuffle audio file names and split them into training, testing, and validation sets.

        This method scans the `Audio` folder inside the species directory for all
        files with the specified audio extension. It then randomly shuffles and splits
        the file names into training, testing, and validation sets according to the 
        specified proportions. The resulting file names (without extensions) are saved
        as text files (`train.txt`, `test.txt`, `validation.txt`) inside the `DataFiles`
        subdirectory of the species folder.

        Parameters
        ----------
        train_size : float, optional
            Proportion of files to use for training. Default is 0.8.
        test_size : float, optional
            Proportion of files to use for testing. Default is 0.1.
        validation_size : float, optional
            Proportion of files to use for validation. Default is 0.1.

        Raises
        ------
        Exception
            If no audio files are found in the specified audio directory.

        Notes
        -----
        - The sum of `train_size`, `test_size`, and `validation_size` should be 1.0.
        - Output files are saved as plain text, with one file name (without extension) per line.
        - The audio extension is read from `self.audio_extension`, and the species folder
        from `self.species_folder`.
        """        
        # Get all file names in Audio folder
        path = Path(self.species_folder, "Audio", f"*{self.audio_extension}")
        files = glob(str(path))

        if len(files) == 0:
            raise Exception(
                f"No audio files found in {self.species_folder}/Audio.\
                Please check the audio_extension setting in the settings file."
            )
        # Shuffle the files
        np.random.shuffle(files)

        train_samples = int(np.floor(len(files) * train_size))
        test_samples = int(np.floor(len(files) * test_size))

        # Split the files into train, test, validation
        train_split = train_samples
        test_split = test_samples

        train_files = files[:train_split]
        test_files = files[train_split : train_split + test_split]
        # Use the rest for validation
        validation_files = files[train_split + test_split :]

        # Only get the file names
        train_files = [os.path.basename(file) for file in train_files]
        test_files = [os.path.basename(file) for file in test_files]
        validation_files = [os.path.basename(file) for file in validation_files]

        # Remove the file extension
        train_files = [os.path.splitext(file)[0] for file in train_files]
        test_files = [os.path.splitext(file)[0] for file in test_files]
        validation_files = [os.path.splitext(file)[0] for file in validation_files]

        # Create the folders
        os.makedirs(Path(self.species_folder, "DataFiles"), exist_ok=True)

        # Save the files as .txt
        with open(Path(self.species_folder, "DataFiles", "train.txt"), "w") as f:
            f.write("\n".join(train_files))
        with open(os.path.join(self.species_folder, "DataFiles", "test.txt"), "w") as f:
            f.write("\n".join(test_files))

        with open(Path(self.species_folder, "DataFiles", "validation.txt"), "w") as f:
            f.write("\n".join(validation_files))

    def check_distribution(self, Y):
        unique, counts = np.unique(Y, return_counts=True)
        original_distribution = dict(zip(unique, counts))
        return original_distribution



    
    
