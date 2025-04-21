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

from .AnnotationReader import *  # type: ignore


class Preprocessing:
    def __init__(
        self,
        species_folder,
        lowpass_cutoff,
        downsample_rate,
        nyquist_rate,
        segment_duration,
        positive_class,
        negative_class,
        nb_negative_class,
        n_fft,
        hop_length,
        n_mels,
        f_min,
        f_max,
        file_type,
        audio_extension,
        apply_preprocessing=True,
    ) -> None:
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
        self.training_files = Path(
            self.species_folder, "DataFiles", "TrainingFiles.txt"
        )
        self.n_ftt = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.file_type = file_type
        self.audio_extension = audio_extension

        self.apply_preprocessing = apply_preprocessing

    def update_audio_path(self, audio_path):
        self.audio_path = Path(self.species_folder, audio_path)

    def read_audio_file(self, file_name):
        """
        file_name: string, name of file including extension, e.g. "audio1.wav"

        """
        # Get the path to the file
        audio_folder = Path(file_name)

        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)

        return audio_amps, audio_sample_rate

    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype="lowpass")  # type: ignore
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        """
        Downsample an audio file to a given new sample rate.
        amplitudes:
        original_sr:
        new_sample_rate:

        """
        """
        return librosa.resample(amplitudes, 
                                original_sr, 
                                new_sample_rate, 
                                res_type='kaiser_fast'), new_sample_rate
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

    def convert_single_to_image(self, audio):
        """
        Convert amplitude values into a mel-spectrogram.
        """
        """
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_ftt,hop_length=self.hop_length, 
                                            n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        """
        if not self.apply_preprocessing:
            f_min = 0
            f_max = 11000
        else:
            f_min = self.f_min
            f_max = self.f_max

        S = librosa.feature.melspectrogram(
            y=audio,
            n_fft=self.n_ftt,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = image_np - np.min(image_np)
        _ = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps = 1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled

        return S1

    def convert_all_to_image(self, segments):
        """
        Convert a number of segments into their corresponding spectrograms.
        """
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment))

        return np.array(spectrograms)

    def add_extra_dim(self, spectrograms):
        """
        Add an extra dimension to the data so that it matches
        the input requirement of Tensorflow.
        """
        spectrograms = np.reshape(
            spectrograms,
            (spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1),
        )
        return spectrograms

    def getXY(
        self,
        audio_amplitudes,
        sampling_rate,
        start_sec,
        annotation_duration_seconds,
        label,
        verbose,
    ):
        """
        Extract a number of segments based on the user-annotations.
        If possible, a number of segments are extracted provided
        that the duration of the annotation is long enough. The segments
        are extracted by shifting by 1 second in time to the right.
        Each segment is then augmented a number of times based on a pre-defined
        user value.
        """

        if verbose:
            print("start_sec", start_sec)
            print("annotation_duration_seconds", annotation_duration_seconds)
            print("self.segment_duration ", self.segment_duration)

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

        if verbose:
            print("segments_to_extract", segments_to_extract)

        if label in self.negative_class:
            if segments_to_extract > self.nb_negative_class:
                segments_to_extract = self.nb_negative_class

        for i in range(0, segments_to_extract):
            if verbose:
                print("Semgnet {} of {}".format(i, segments_to_extract - 1))
                print("*******************")

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
            start_time_seconds = start_sec + i  # type: ignore

            if verbose:
                print("start frame", start_data_observation)
                print("end frame", end_data_observation)

            # Extend the augmented segments and labels
            X_segments.append(X_audio)
            Y_labels.append(label)

        return X_segments, Y_labels

    def save_data_to_pickle(self, X, Y):
        """
        Save all of the spectrograms to a pickle file.

        """
        outfile = open(Path(self.saved_data_path, "X.pkl"), "wb")
        pickle.dump(X, outfile, protocol=4)
        outfile.close()

        outfile = open(Path(self.saved_data_path, "Y.pkl"), "wb")
        pickle.dump(Y, outfile, protocol=4)
        outfile.close()

    def load_data_from_pickle(self):
        """
        Load all of the spectrograms from a pickle file

        """
        infile = open(Path(self.saved_data_path, "X.pkl"), "rb")
        X = pickle.load(infile)
        infile.close()

        infile = open(Path(self.saved_data_path, "Y.pkl"), "rb")
        Y = pickle.load(infile)
        infile.close()

        return X, Y

    def time_shifting(self, X, index):
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
        segment = self.time_shift(segment, random_time_point_segment)

        return segment

    def time_shift(self, audio, shift):
        """
        Shift amplitude values to the right by a random value.

        The amplitude values are wrapped back to the left side of the waveform.

        Args:
            audio (ndarray): Array of amplitude values representing the audio waveform.
            shift (int): Amount of shift to apply to the waveform.

        Returns:
            ndarray: Augmented waveform with the shifted amplitude values.
        """

        augmented = np.zeros(len(audio))
        augmented[0:shift] = audio[-shift:]
        augmented[shift:] = audio[:-shift]

        return augmented

    def augment_dataset(self, X, Y):
        label_to_augment = np.argmin(np.unique(np.asarray(Y), return_counts=True)[1])
        difference = np.max(np.unique(np.asarray(Y), return_counts=True)[1]) - np.min(
            np.unique(np.asarray(Y), return_counts=True)[1]
        )
        index = np.where(
            np.asarray(Y)
            == np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]
        )[0]

        X_augmented = []
        Y_augmented = []
        X_augmented.extend(X)
        Y_augmented.extend(Y)
        X = np.asarray(X)
        for i in range(difference):
            X_augmented.append(self.time_shifting(X, index))
            Y_augmented.append(
                np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]
            )

        return X_augmented, Y_augmented

    def create_dataset(
        self,
        verbose,
        annotation_folder,
        sufix_file,
        file_names=None,
        augmentation=False,
    ):
        """
        Create X and Y values which are inputs to a ML algorithm.
        Annotated files (.svl) are read and the corresponding audio file (.wav)
        is read. A low pass filter is applied, followed by downsampling. A
        number of segments are extracted and augmented to create the final dataset.
        Annotated files (.svl) are created using SonicVisualiser and it is assumed
        that the "boxes area" layer was used to annotate the audio files.
        """

        if file_names is None:
            file_names = self.training_files
        # Keep track of how many calls were found in the annotation files
        total_calls = 0

        # Initialise lists to store the X and Y values
        X_calls = []
        Y_calls = []

        if verbose == True:
            print("Annotations path:", Path(self.annotations_path, "*.svl"))
            print("Audio path", Path(self.audio_path, "*.wav"))

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

            if self.file_type == "svl":
                # Get the file name without paths and extensions
                file_name_no_extension = file
                # print ('file_name_no_extension', file_name_no_extension)
            if self.file_type == "raven_caovitgibbons":
                file_name_no_extension = file[file.rfind("-") + 1 : file.find(".")]

            print("Processing:", file_name_no_extension) if verbose else None

            reader = AnnotationReader(
                self.species_folder, file, self.file_type, self.audio_extension
            )
            # Check if the .wav file exists before processing
            # if self.audio_path+file_name_no_extension+self.audio_extension  in glob.glob(self.audio_path+"*"+self.audio_extension):
            if str(
                Path(self.audio_path, file_name_no_extension + self.audio_extension)
            ) in glob(str(self.audio_path / f"*{self.audio_extension}")):
                print(f"Found file {file_name_no_extension}")

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
                    print("Filtering...") if verbose else None
                    # Low pass filter
                    filtered = self.butter_lowpass_filter(
                        audio_amps, self.lowpass_cutoff, self.nyquist_rate
                    )
                    print("Downsampling...") if verbose else None
                    # Downsample
                    amplitudes, sample_rate = self.downsample_file(
                        filtered, original_sample_rate, self.downsample_rate
                    )

                else:
                    print("No preprocessing applied") if verbose else None
                    amplitudes, sample_rate = audio_amps, original_sample_rate

                df, audio_file_name = reader.get_annotation_information(
                    annotation_folder, sufix_file
                )

                print("Reading annotations...") if verbose else None
                for index, row in df.iterrows():
                    start_seconds = int(round(row["Start"]))
                    end_seconds = int(round(row["End"]))
                    label = row["Label"]
                    annotation_duration_seconds = end_seconds - start_seconds

                    # Extract augmented audio segments and corresponding binary labels
                    X_data, y_data = self.getXY(
                        amplitudes,
                        sample_rate,
                        start_seconds,
                        annotation_duration_seconds,
                        label,
                        file_name_no_extension,
                        verbose,
                    )

                    # Append the segments and labels
                    X_calls.extend(X_data)
                    Y_calls.extend(y_data)

        print(len(X_calls))
        print(np.unique(Y_calls, return_counts=True))

        if augmentation:
            # Augment dataset to get a balance dataset
            print("Augmentation ")
            print(len(X_calls), len(Y_calls))
            X_calls, Y_calls = self.augment_dataset(X_calls, Y_calls)

        X_calls = self.convert_all_to_image(X_calls)

        # Convert to numpy arrays
        X_calls, Y_calls = np.asarray(X_calls), np.asarray(Y_calls)

        return X_calls, Y_calls

    def _shuffle_files_names(self, train_size=0.8, test_size=0.1, validation_size=0.1):
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

    def repair_svl(self, file_names):
        saved_folder = Path(self.species_folder, "Annotations_corrected")
        os.makedirs(saved_folder, exist_ok=True)

        files = pd.read_csv(file_names, header=None)

        for file in files.values:
            file = file[0]
            reader = AnnotationReader(
                file, self.species_folder, self.file_type, self.audio_extension
            )

            df, sampleRate, start_m, end_m, audio_len, audio_seconds = (
                reader.get_annotation_information_testing()
            )

            new_frames = []
            new_values = []
            new_extents = []
            new_durations = []
            new_labels = []

            for i in range(0, int(int(audio_len) / int(sampleRate)), 4):
                index_start = i
                index_end = i + 4

                overlap_label = False

                for index, row in df.iterrows():
                    labeled_start = row["frame"] / int(sampleRate)
                    labeled_end = int(
                        (row["frame"] + row["duration"]) / int(sampleRate)
                    )

                    if index_start >= labeled_start and index_start <= labeled_end:
                        if row["label"] == self.positive_class:
                            overlap_label = True

                    if index_end >= labeled_start and index_end <= labeled_end:
                        if row["label"] == self.positive_class:
                            overlap_label = True

                if overlap_label != True:
                    new_frames.append(index_start * int(sampleRate))
                    new_values.append(700)
                    new_durations.append(4 * int(sampleRate))
                    new_extents.append(1500)
                    new_labels.append(self.negative_class)

            print("---")

            df_repaired = pd.DataFrame(
                {
                    "frame": new_frames,
                    "value": new_values,
                    "duration": new_durations,
                    "extent": new_extents,
                    "label": new_labels,
                }
            )

            df_repaired = pd.concat(
                [df_repaired, df[df["label"] == self.positive_class]]
            )

            xml = reader.dataframe_to_svl(df_repaired, sampleRate, start_m, end_m)

            svl_file_path = Path(f"{save_folder}", f"{file}_repaired.svl")
            text_file = open(str(svl_file_path), "a")
            # text_file = open('{}{}_repaired.svl'.format(str(saved_folder)+'\\', file), "a")
            n = text_file.write(xml)
            text_file.close()
