import datetime
import glob, os
from yattag import Doc, indent
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import shutil
import time
import io
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from .logger import *
from .preprocessing import *
from .settings import Config
from ..model.data import Data
from ..model.model import Model


# Class to open the chromosome saved on GPU machine on CPU machine
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)


class Evaluation:
    def __init__(
        self,
        species_folder: str,
        settings,
        chromosome=None,
        apply_preprocessing: bool = True,
        logger=None,
        log_path=None,
        log_level=10,
        save_folder: str = "Predictions",
    ) -> None:
        self.logger = setup_logger(
            logger=logger, log_path=log_path, log_level=log_level
        )

        self.species_folder = species_folder
        __preprocessing_name = "preprocessed" if apply_preprocessing else "unpreprocessed"
        self.saved_data_folder = Path(species_folder, "SavedData", __preprocessing_name)

        self.apply_preprocessing_flag = apply_preprocessing
        self.config = settings
        self.positive_class = self.config.data.dict()["positive_class"]
        self.negative_class = self.config.data.dict()["negative_class"]

        self.chromosome = chromosome
        self.save_amplitudes_path = Path(save_folder, "amplitudes_to_predict")
        os.makedirs(self.save_amplitudes_path, exist_ok=True)
        if self.chromosome == None:
            self.save_folder = save_folder + "_baseline"
        else:
            self.save_folder = save_folder + "_chromosome"

        self.save_results = Path(self.species_folder, self.save_folder)

    def _group_consecutives(self, vals, step=1):
        """Return list of consecutive lists of numbers from vals (number list)."""
        run = []
        result = [run]
        expect = None
        for v in vals:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result

    def _group(self, L):
        L.sort()
        first = last = L[0]
        for n in L[1:]:
            if n - 1 == last:  # Part of the group, bump the end
                last = n
            else:  # Not part of the group, yield current group and start a new
                yield first, last
                first = last = n
        yield first, last  # Yield the last group

    def _dataframe_to_svl(self, dataframe, sample_rate, length_audio_file_frames):
        doc, tag, text = Doc().tagtext()
        doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
        doc.asis("<!DOCTYPE sonic-visualiser>")

        with tag("sv"):
            with tag("data"):
                model_string = '<model id="1" name="" sampleRate="{}" start="0" end="{}" type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="0" subtype="box" minimum="0" maximum="{}" units="Hz" />'.format(
                    sample_rate, length_audio_file_frames, sample_rate / 2
                )
                doc.asis(model_string)

                with tag("dataset", id="0", dimensions="2"):
                    # Read dataframe or other data structure and add the values here
                    # These are added as "point" elements, for example:
                    # '<point frame="15360" value="3136.87" duration="1724416" extent="2139.22" label="Cape Robin" />'
                    for index, row in dataframe.iterrows():
                        point = '<point frame="{}" value="{}" duration="{}" extent="{}" label="{}" />'.format(
                            int(int(row["start(sec)"]) * sample_rate),
                            int(row["low(freq)"]),
                            int(
                                (int(row["end(sec)"]) - int(row["start(sec)"]))
                                * sample_rate
                            ),
                            int(row["high(freq)"]),
                            row["label"],
                        )

                        # add the point
                        doc.asis(point)
            with tag("display"):
                display_string = '<layer id="2" type="boxes" name="Boxes" model="1"  verticalScale="0"  colourName="White" colour="#ffffff" darkBackground="true" />'
                doc.asis(display_string)

        result = indent(doc.getvalue(), indentation=" " * 2, newline="\r\n")

        return result

    def _calculate_vocalisation_time(self, file_name, detected_time):
        # print ('calculate_vocalisation_time')
        # print (type(file_name))
        # print (file_name)
        # print (type(detected_time))
        # print (detected_time)
        # print (type(int(detected_time)))

        date_time = file_name[file_name.find("_") + 1 :]
        date = date_time[: date_time.find("_")]
        time = date_time[date_time.find("_") + 1 :]
        now = datetime.datetime(
            int(date[0:4]),
            int(date[4:6]),
            int(date[6:8]),
            int(time[0:2]),
            int(time[2:4]),
            int(time[4:6]),
        )
        difference1 = datetime.timedelta(seconds=detected_time)
        new_time = now + difference1

        return new_time.strftime("%Y%m%d_%H%M%S")

    def _save_detected_calls(
        self, groupped_detection, amplitudes, file_name, sample_rate, sub_folder_name
    ):
        if len(groupped_detection) == 0:
            print("No predictions to save.")
            return

        for counter, group in enumerate(groupped_detection):
            # Added 23 Nov 2021
            if len(group) > 4:
                start_index = group[0] - 2  # Add 2 seconds before
                end_index = group[-1] + 2  # Add 2 seconds after

                print("Saving...")

                time_stamp = self._calculate_vocalisation_time(
                    file_name, int(start_index)
                )
                sf.write(
                    str(
                        Path(
                            self.save_results,
                            file_name
                            + "_"
                            + str(counter)
                            + "_detected_"
                            + str(time_stamp)
                            + ".wav",
                        )
                    ),
                    amplitudes[sample_rate * start_index : sample_rate * end_index],
                    sample_rate,
                )

            print("")

    def _predictions(self, model, inputs, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        prediction_list = []
        model = model.to(device)
        model.eval()
        np.save("inputs.npy", inputs)
        X_tensor = torch.from_numpy(inputs).float()

        # Reshape X_tensor
        # print("X_TENSOR SHAPE", X_tensor.shape)
        if len(X_tensor.shape) == 3:
            X_tensor = X_tensor.unsqueeze(1)

        loader = torch.utils.data.DataLoader(
            X_tensor, batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                prediction_list.append(pred.cpu())

        softmax_prediction = [i.detach().numpy() for i in prediction_list]
        return np.vstack(softmax_prediction)

    def _calc_amplitudes_to_predict(self, file_name, preprocessing, verbose = True):
        if str(
            Path(preprocessing.audio_path, file_name + preprocessing.audio_extension)
        ) in glob(str(preprocessing.audio_path / f"*{preprocessing.audio_extension}")):
            print("Found file")

        audio_amps, original_sample_rate = preprocessing.read_audio_file(
            str(
                Path(
                    preprocessing.audio_path, file_name + preprocessing.audio_extension
                )
            )
        )

        if self.chromosome == None:
            print("Filtering...") if verbose else None
            # Low pass filter
            filtered = preprocessing.butter_lowpass_filter(
                audio_amps, preprocessing.lowpass_cutoff, preprocessing.nyquist_rate
            )
            print("Downsampling...") if verbose else None
            # Downsample
            amplitudes, sample_rate = preprocessing.downsample_file(
                filtered, original_sample_rate, preprocessing.downsample_rate
            )
            del filtered

        else:
            print("No preprocessing applied") if verbose else None
            amplitudes, sample_rate = audio_amps, original_sample_rate

        start_values = np.arange(
            0, len(audio_amps) / original_sample_rate - preprocessing.segment_duration
        ).astype(int)
        end_values = np.arange(
            preprocessing.segment_duration, len(audio_amps) / original_sample_rate
        ).astype(int)

        """print("START VALUES", start_values)
        print("END VALUES", end_values)
        print("SAMPLE RATE", sample_rate)
        print("SEGMENT DURATION", preprocessing.segment_duration)
        print("LEN AUDIO AMPS", len(audio_amps))"""

        amplitudes_to_predict = []

        for i in range(len(start_values)):
            s = start_values[i]
            e = end_values[i]

            S = preprocessing.convert_single_to_image(
                amplitudes[s * sample_rate : e * sample_rate]
            )
            amplitudes_to_predict.append(S)

        len_audio_amps = len(audio_amps)
        del audio_amps
        amplitudes_to_predict = np.asarray(amplitudes_to_predict)
        
        # save to disk
        save_dict = {
            "amplitudes_to_predict": amplitudes_to_predict,
            "sample_rate": sample_rate,
            "original_sample_rate": original_sample_rate,
            "len_audio_amps": len_audio_amps,
        }
        # Save to results folder
        save_name = Path(self.save_amplitudes_path, file_name + "_amplitudes_to_predict_preprecoessing_" + str(self.apply_preprocessing_flag) + ".npy")
        np.save(save_name, save_dict)
        print("Saved amplitudes to predict to disk: ", save_name)
        return  amplitudes_to_predict, sample_rate, original_sample_rate,len_audio_amps
    
    
    def _get_amplitudes_to_predict(self, file_name, preprocessing, verbose = True):
        # Change this to True to force recalculation of amplitudes to predict, make this a parameter later
        if False:
            print("Forcing recalculation of amplitudes to predict")
            return self._calc_amplitudes_to_predict(file_name, preprocessing, verbose=verbose)
        # Check if the amplitudes to predict have already been calculated
        save_name = Path(self.save_amplitudes_path, file_name + "_amplitudes_to_predict_preprecoessing_" + str(self.apply_preprocessing_flag) + ".npy")
        if os.path.exists(save_name):
            print("Found amplitudes to predict on disk: ", save_name)
            data = np.load(save_name, allow_pickle=True)
            amplitudes_to_predict = data.item().get("amplitudes_to_predict")
            sample_rate = data.item().get("sample_rate")
            original_sample_rate = data.item().get("original_sample_rate")
            len_audio_amps = data.item().get("len_audio_amps")
            return amplitudes_to_predict, sample_rate, original_sample_rate, len_audio_amps
        else:
            print("No amplitudes to predict found on disk")
            return self._calc_amplitudes_to_predict(file_name, preprocessing, verbose=verbose)
    
    def _process_one_file(self, file_name, model, preprocessing, verbose=True):
        amplitudes_to_predict, sample_rate, original_sample_rate, len_audio_amps = self._get_amplitudes_to_predict(file_name, preprocessing, verbose=verbose)
        if self.chromosome != None:
            print("Extracting bands of spectrogram from chromosome genes..")
            amplitudes_to_predict = self.chromosome._create_dataset(
                amplitudes_to_predict
            )
     
        print("Predicting...")

        # predictions
        softmax_predictions = self._predictions(
            model, amplitudes_to_predict, batch_size=128
        )
        del amplitudes_to_predict

        binary_predictions = []
        prediction_seconds = []
        for index, softmax_values in enumerate(softmax_predictions):
            # to check : gibbon call has to be associated with (0,1)
            if softmax_values[1] < 0.8:
                binary_predictions.append(preprocessing.negative_class)
                prediction_seconds.append(0)
            else:
                binary_predictions.append(preprocessing.positive_class)
                prediction_seconds.append(1)

        # Group the detections together
        groupped_detection = self._group_consecutives(
            np.where(np.asarray(prediction_seconds) == 1)[0]
        )

        if len(groupped_detection[0]) > 0:
            # Save the groupped detections to .wav output clips
            # save_detected_calls(groupped_detection, amplitudes, file_name, sample_rate, sub_folder_name)
            # Save the groupped predictions to .svl file to visualize them with Sonic Visualizer
            predictions = []
            for pred in groupped_detection:
                if len(pred) >= 2:
                    for predicted_second in pred:
                        # Update the set of all the predicted calls
                        predictions.append(predicted_second)

            predictions.sort()

            # Only process if there are consecutive groups
            if len(predictions) > 0:
                predicted_groups = list(self._group(predictions))

                # Create a dataframe to store each prediction
                df_values = []
                prediction_name = "predicted_chromosome" if self.chromosome != None else "predicted_baseline"
                for pred_values in predicted_groups:
                    df_values.append(
                        [
                            pred_values[0],
                            pred_values[1] + preprocessing.segment_duration,
                            600,
                            2000,
                            prediction_name,
                        ]
                    )
                df_preds = pd.DataFrame(
                    df_values,
                    columns=[
                        ["start(sec)", "end(sec)", "low(freq)", "high(freq)", "label"]
                    ],
                )

                # Create a .svl outpupt file
                xml = self._dataframe_to_svl(
                    df_preds, original_sample_rate, len_audio_amps
                )

                # Write the .svl file
                # text_file = open(species_folder+'/Model_Output/'+file_name_no_extension+"_"+self.model_type+".svl", "w")
                text_file = open(
                    "{}_predictions.svl".format(
                        str(Path(self.save_results, file_name))
                    ),
                    "w",
                )
                n = text_file.write(xml)
                text_file.close()

        else:
            print("No detected calls to save.")
        #del amplitudes

        print("Done")

    def prediction_files(self, model, type = "test"):
        test_path = Path(self.species_folder, "DataFiles", type + ".txt")
        #test_path = Path(self.species_folder, "DataFiles", "test.txt")
        file_names = pd.read_csv(test_path, header=None)

        preprocessing = Preprocessing(
            **self.config.preprocessing.dict(),
            positive_class=self.config.data.dict()["positive_class"],
            negative_class=self.config.data.dict()["negative_class"],
            apply_preprocessing=self.apply_preprocessing_flag,
            species_folder=self.species_folder,
        )

        if os.path.isdir(self.save_results) == False:
            print("creating the folder")
        else:
            shutil.rmtree(self.save_results)
            print("clean the folder")

        os.mkdir(self.save_results)

        for file in file_names.values:
            file = file[0]

            print("Processing file:", file)

            self._process_one_file(file, model, preprocessing, verbose=True)

    def _overlap(self, start1, end1, start2, end2):
        """how much does the range (start1, end1) overlap with (start2, end2)"""
        return max(
            max((end2 - start1), 0) - max((end2 - end1), 0) - max((start2 - start1), 0),
            0,
        )

    def _repair_svl(
        self, file_names, file_type, audio_extension, annotation_folder, sufix_file
    ):
        saved_folder = Path(self.species_folder, "Annotations_corrected")
        os.makedirs(saved_folder, exist_ok=True)

        for file in file_names.values:
            file = file[0]
            reader = AnnotationReader(
                self.species_folder, file, file_type, audio_extension
            )

            (
                df,
                sampleRate,
                start_m,
                end_m,
                audio_len,
                audio_seconds,
            ) = reader.get_annotation_information_testing()

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
            saved_folder_path = Path(saved_folder)
            text_file_path = saved_folder_path / f"{file}_repaired.svl"
            text_file = open(str(text_file_path), "a")
            n = text_file.write(xml)
            text_file.close()

        return audio_seconds

    def comparison_predictions_annotations(self, folder, type="test"):
        print("comparing prediction and annotation")
        test_path = Path(self.species_folder, "DataFiles", type + ".txt")
        #test_path = Path(self.species_folder, "DataFiles", "test.txt")
        file_names = pd.read_csv(test_path, header=None)

        preprocessing = Preprocessing(
            **self.config.preprocessing.dict(),
            positive_class=self.config.data.dict()["positive_class"],
            negative_class=self.config.data.dict()["negative_class"],
            apply_preprocessing=self.apply_preprocessing_flag,
            species_folder=self.species_folder,
        )

        predictions = []
        annotations = []

        # check if corrected annotations for the testing files have been done
        if os.path.exists(Path(self.species_folder, "Annotations_corrected")):
            self.logger.info(
                "the corrected annotations of the testing dataset have already been created "
            )

        else:
            self.logger.info(
                "Need to modofity the annotations of the testing dataset to allow a correct evaluation of the model "
            )
            self._repair_svl(
                file_names,
                preprocessing.file_type,
                preprocessing.audio_extension,
                annotation_folder="Annotations",
                sufix_file=".svl",
            )
        for file in file_names.values:
            file = file[0]

            reader = AnnotationReader(
                self.species_folder,
                file,
                preprocessing.file_type,
                preprocessing.audio_extension,
            )

            svl = reader.get_annotation_information(
                annotation_folder="Annotations_corrected", sufix_file="_repaired.svl"
            )[0]
            svl["Overlap"] = 0
            svl["Cat"] = "TN"
            svl.loc[svl.Label == preprocessing.positive_class, "Cat"] = "FN"
            svl["Index"] = np.nan
            svl["Nb overlap"] = 0
            svl["Name"] = file

            if os.path.exists(
                Path(self.species_folder, folder, file + "_predictions.svl")
            ):
                print("Found Prediction: ", file)
                predict = reader.get_annotation_information(
                    annotation_folder=folder, sufix_file="_predictions.svl"
                )[0]

                predict["Overlap"] = 0
                predict["Cat"] = "FP"
                predict["Index"] = np.nan
                predict["Nb overlap"] = 0
                predict["Name"] = file

                # compare predictions vs annotations
                if svl[svl.Label == preprocessing.positive_class].shape[0] != 0:
                    for index, row in predict.iterrows():
                        idx = np.abs(
                            np.asarray(
                                svl[svl.Label == preprocessing.positive_class]["Start"]
                            )
                            - row.iloc[0]
                        ).argmin()  # get the closest window
                        lap = self._overlap(
                            row.iloc[0],
                            row.iloc[1],
                            svl[svl.Label == preprocessing.positive_class].iloc[idx, 0],
                            svl[svl.Label == preprocessing.positive_class].iloc[idx, 1],
                        )  # check overlap

                        if lap > 2:
                            predict.loc[index, "Overlap"] = int(lap.copy())
                            predict.loc[index, "Cat"] = "TP"
                            predict.loc[index, "Index"] = idx
                        else:
                            predict.loc[index, "Overlap"] = int(lap)

                    for index, row in predict.iterrows():
                        w = 0
                        for idx_svl, row_svl in svl[
                            svl.Label == preprocessing.positive_class
                        ].iterrows():
                            lap = self._overlap(
                                row.iloc[0],
                                row.iloc[1],
                                row_svl.iloc[0],
                                row_svl.iloc[1],
                            )
                            if lap > 2:
                                w += 1
                        predict.loc[index, "Nb overlap"] = w
                else:
                    print("No positive class in the annotation file")
                predictions.append(predict)

                # compare annotations vs predictions
                for index, row in svl.iterrows():
                    idx = np.abs(
                        np.asarray(predict["Start"]) - row.iloc[0]
                    ).argmin()  # get the closest window
                    lap = self._overlap(
                        row.iloc[0],
                        row.iloc[1],
                        predict.iloc[idx, 0],
                        predict.iloc[idx, 1],
                    )  # check overlap

                    if (lap > 2) & (
                        svl.loc[index, "Label"] == preprocessing.positive_class
                    ):
                        svl.loc[index, "Overlap"] = int(lap.copy())
                        svl.loc[index, "Index"] = idx
                        svl.loc[index, "Cat"] = "TP"
                    elif (lap > 2) & (
                        svl.loc[index, "Label"] == preprocessing.negative_class
                    ):
                        svl.loc[index, "Overlap"] = lap.copy()
                        svl.loc[index, "Index"] = idx
                        svl.loc[index, "Cat"] = "FP"
                    else:
                        svl.loc[index, "Overlap"] = int(lap)

                # Print File and FP TP FN
                print("-------------")
                print(file)
                print("FP : ", predict[predict.Cat == "FP"].shape[0])
                print("TP : ", predict[predict.Cat == "TP"].shape[0])
                print("FN : ", svl[svl.Cat == "FN"].shape[0])
                print("-------------")

                for index, row in svl.iterrows():
                    w = 0
                    for idx_pred, row_pred in predict.iterrows():
                        lap = self._overlap(
                            row.iloc[0], row.iloc[1], row_pred.iloc[0], row_pred.iloc[1]
                        )
                        if lap > 2:
                            w += 1
                    svl.loc[index, "Nb overlap"] = w

            annotations.append(svl)

        Predictions = pd.DataFrame(np.concatenate(predictions, axis=0))
        Predictions.columns = predict.columns
        Predictions.Index = Predictions.Index.astype(float)

        Annotations = pd.DataFrame(np.concatenate(annotations, axis=0))
        Annotations.columns = svl.columns
        Annotations.Index = Annotations.Index.astype(float)

        return Predictions, Annotations

    def testing_score(self, Annotations, Predictions):
        preprocessing = Preprocessing(
            **self.config.preprocessing.dict(),
            positive_class=self.config.data.dict()["positive_class"],
            negative_class=self.config.data.dict()["negative_class"],
            apply_preprocessing=self.apply_preprocessing_flag,
            species_folder=self.species_folder,
        )

        cat, count = np.unique(Predictions["Cat"], return_counts=True)
        print(cat, count)

        cat_a, count_a = np.unique(Annotations["Cat"], return_counts=True)
        print(cat_a, count_a)

        FP = count[cat == "FP"][0] if len(count[cat == "FP"]) > 0 else 0
        TP = count_a[cat_a == "TP"][0] if len(count_a[cat_a == "TP"]) > 0 else 0
        FN = count_a[cat_a == "FN"][0] if len(count_a[cat_a == "FN"]) > 0 else 0
        TN = count_a[cat_a == "TN"][0] if len(count_a[cat_a == "TN"]) > 0 else 0

        F_score = TP / (TP + ((FN + FP) / 2))
        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        print(
            "Number of calls to detect : ",
            Annotations[Annotations.Label == preprocessing.positive_class].shape[0],
        )
        print()
        print("False Positif : ", FP)
        print("True Positif : ", TP)
        print("False Negatif : ", FN)
        print()
        print("F1-score : ", F_score)
        print("Accuracy : ", Accuracy)

        return F_score, Accuracy


    def _normal_run(self, model, type="test", threshold=None, print_report=True, metric="f1",return_image_shape_and_parameters = True):
        data_path = Path(self.saved_data_folder, type)
        X = np.load(data_path / "X.pkl", allow_pickle=True)
        if self.chromosome is not None:
            X = self.chromosome._create_dataset(X)
        
        Y = np.load(data_path / "Y.pkl", allow_pickle=True)
        print("Data Loaded from: ", data_path)
        print("Evaluating...")
        
        
        predictions = self._predictions(model, X, batch_size=128)
        
        if threshold is None:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (predictions[:,1] > threshold).astype(int)
        
        targets = np.argmax(Y, axis=1)
        
      
        f1 = f1_score(targets, predictions)
        report = classification_report(targets, predictions)
        confusion = confusion_matrix(targets, predictions)
        if print_report:
            print(report)
            print(confusion)
         
        if return_image_shape_and_parameters:
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            image_shape = X.shape[1:]
            number_of_pixels = np.prod(image_shape)
            return f1, confusion, trainable_parameters, image_shape, number_of_pixels
            
        else:
            return f1, confusion

    def run(self, model, type="test", test_type = "normal"):
        if test_type == "normal":
            return self._normal_run(model, type=type)
        starting = time.time()
        self.prediction_files(model, type=type)
        Predictions, Annotations = self.comparison_predictions_annotations(
            self.save_folder, type=type
        )
        F_score, Accuracy = self.testing_score(Annotations, Predictions)
        execution_time = time.time() - starting
        return F_score, Accuracy, execution_time
