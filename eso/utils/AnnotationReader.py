import pandas as pd
import os
import librosa
from xml.dom import minidom
import soundfile as sf
from yattag import Doc, indent
from pathlib import Path


class AnnotationReader:
    def __init__(
        self, 
        path : str, 
        annotation_file_name : str, 
        file_type : str, 
        audio_extension : str, 
        positive_class: str):
        

        self.path = path
        self.annotation_file_name = annotation_file_name
        self.file_type = file_type
        self.audio_extension = audio_extension
        self.positive_class=positive_class
        """
        Initializes the AnnotationReader class.

        Parameters
        ----------
        path : str
            The path to the directory containing the annotation and audio files.
        annotation_file_name : str
            The name of the annotation file (without extension) to be read.
        file_type : str
            The type of annotation file (e.g., "svl", "xml").
        audio_extension : str
            The file extension for the associated audio files (e.g., ".wav", ".mp3").
        positive_class : str
            The label representing the positive class in classification tasks.
        
        Returns
        -------
        None
        """

    def get_annotation_information(self, annotation_folder, sufix_file ):
        """
        Extract annotation information from an `.svl` XML file and return a DataFrame
        with start times, end times, and labels for the annotations.

        This method parses an XML annotation file (`.svl` format) to extract annotation
        details including the start time, end time, and label for each annotation.
        It processes the XML file, handles any confidence values, and adjusts labels
        accordingly (e.g., using the positive class label for predicted annotations).
        
        Parameters
        ----------
        annotation_folder : str
            The folder where the annotation file is located.
        sufix_file : str
            The suffix to append to the base annotation file name to get the full file name.

        Returns
        -------
        tuple
            A tuple containing:
            - pd.DataFrame: A DataFrame with three columns:
                - 'Start': The start time of the annotation in seconds.
                - 'End': The end time of the annotation in seconds.
                - 'Label': The label associated with the annotation.
            - str: The name of the corresponding audio file (with ".wav" extension).

        Raises
        ------
        Exception
            If the annotation file does not contain valid annotation information.
        """
            
        path = str(Path(
                self.path, annotation_folder, self.annotation_file_name + sufix_file
            ))

                            
        xmldoc = minidom.parse(path)
        itemlist = xmldoc.getElementsByTagName("point")
        idlist = xmldoc.getElementsByTagName("model")

        start_time = []
        end_time = []
        labels = []
        audio_file_name = ""

        if len(idlist) > 0:
            for s in idlist: 
                original_sample_rate = int(s.attributes["sampleRate"].value)


        if len(itemlist) > 0:

            # Iterate over each annotation in the .svl file (annotatation file)
            for s in itemlist:
                # Get the starting seconds from the annotation file. Must be an integer
                # so that the correct frame from the waveform can be extracted
                start_seconds = (
                        float(s.attributes["frame"].value) / original_sample_rate
                    )

                # Get the label from the annotation file
                label = str(s.attributes["label"].value)

                # Set the default confidence to 10 (i.e. high confidence that
                # the label is correct). Annotations that do not have the idea
                # of 'confidence' are teated like normal annotations and it is
                # assumed that the annotation is correct (by the annotator).
                label_confidence = 10

                # Check if a confidence has been assigned
                if "," in label:
                    # Extract the raw label
                    lalel_string = label[: label.find(",") :]

                    # Extract confidence value
                    label_confidence = int(label[label.find(",") + 1 :])

                    # Set the label to the raw label
                    label = lalel_string

                    # If a file has a blank label then skip this annotation
                    # to avoid mislabelling data
                if label == "":
                    break


                #to include predictions obtained from a model
                if label == "predicted" :
                    label=self.positive_class

                # Only considered cases where the labels are very confident
                # 10 = very confident, 5 = medium, 1 = unsure this is represented
                # as "SPECIES:10", "SPECIES:5" when annotating.
                if label_confidence == 10:
                    # Get the duration from the annotation file
                    annotation_duration_seconds = (
                            float(s.attributes["duration"].value) / original_sample_rate
                        )
                    start_time.append(start_seconds)
                    end_time.append(start_seconds + annotation_duration_seconds)
                    labels.append(label)

        df_svl_gibbons = pd.DataFrame(
                {"Start": start_time, "End": end_time, "Label": labels}
            )
        return df_svl_gibbons, self.annotation_file_name + ".wav"

    
    def get_annotation_information_testing(self):
        """
        Extract annotation information from a `.svl` XML file and return a DataFrame
        with frame, value, duration, extent, and label for each annotation.

        This method parses an XML annotation file (`.svl` format) to extract detailed
        annotation information such as frame number, value, duration, extent, and label.
        It also extracts the sample rate, start time, and end time from the file's metadata.
        
        Parameters
        ----------
        None

        Returns
        -------
        tuple
            A tuple containing:
            - pd.DataFrame: A DataFrame with columns:
                - 'frame': The frame number from the annotation.
                - 'value': The value associated with the annotation.
                - 'duration': The duration of the annotation.
                - 'extent': The extent of the annotation.
                - 'label': The label associated with the annotation.
            - int: The sample rate extracted from the `.svl` file.
            - str: The start time of the annotation in the `.svl` file.
            - str: The end time of the annotation in the `.svl` file.
        
        Raises
        ------
        Exception
            If the annotation file is not found or if it does not contain valid annotation information.
        """

        path = os.path.join(
                self.path, "Annotations", self.annotation_file_name + ".svl"
            )

        # Process the .svl xml file
        xmldoc = minidom.parse(path)
        itemlist = xmldoc.getElementsByTagName('point')
        idlist = xmldoc.getElementsByTagName('model')

        sampleRate = idlist.item(0).attributes['sampleRate'].value 
        start_m = idlist.item(0).attributes['start'].value
        end_m = idlist.item(0).attributes['end'].value
    

        values = []
        frames = []
        durations=[]
        extents=[]
        labels = []
        audio_file_name = ''

        if len(idlist) > 0:
            for s in idlist: 
                original_sample_rate = int(s.attributes["sampleRate"].value)

        if (len(itemlist) > 0):

        # Iterate over each annotation in the .svl file (annotatation file)
            for s in itemlist:

                # Get the starting seconds from the annotation file. Must be an integer
                # so that the correct frame from the waveform can be extracted
                frame = float(s.attributes['frame'].value)
                value = float(s.attributes['value'].value)
                duration = float(s.attributes['duration'].value)
                extent = float(s.attributes['extent'].value)
                label = str(s.attributes['label'].value)

                # Set the default confidence to 10 (i.e. high confidence that
                # the label is correct). Annotations that do not have the idea
                # of 'confidence' are teated like normal annotations and it is
                # assumed that the annotation is correct (by the annotator). 
                label_confidence = 10

                # Check if a confidence has been assigned
                if ',' in label:

                    # Extract the raw label
                    lalel_string = label[:label.find(','):]

                    # Extract confidence value
                    label_confidence = int(label[label.find(',')+1:])

                    # Set the label to the raw label
                    label = lalel_string


                # If a file has a blank label then skip this annotation
                # to avoid mislabelling data
                if label == '':
                    break

                # Only considered cases where the labels are very confident
                # 10 = very confident, 5 = medium, 1 = unsure this is represented
                # as "SPECIES:10", "SPECIES:5" when annotating.
                if label_confidence == 10:

                    frames.append(frame)
                    values.append(value)
                    durations.append(duration)
                    extents.append(extent)
                    labels.append(label)

        df_svl_gibbons = pd.DataFrame({'frame': frames, 'value':values ,'duration': durations,
                                  'extent':extents,'label':labels})
        return df_svl_gibbons, sampleRate, start_m, end_m


    def dataframe_to_svl(self, dataframe, sample_rate, start_m, end_m):
        """
        Convert a DataFrame of annotations to a `.svl` format XML string.

        This method generates a `.svl` format XML string containing the annotations
        from a DataFrame. The generated XML includes metadata such as the sample rate,
        start time, end time, and annotation points (frame, value, duration, extent, and label).

        Parameters
        ----------
        dataframe : pd.DataFrame
            A DataFrame containing the annotation information. The DataFrame should have 
            the following columns: 'frame', 'value', 'duration', 'extent', and 'label'.
        sample_rate : int
            The sample rate of the audio associated with the annotations.
        start_m : str
            The start time (in seconds) of the annotation period.
        end_m : str
            The end time (in seconds) of the annotation period.

        Returns
        -------
        str
            A string containing the XML in `.svl` format, representing the annotations
            along with metadata.

        Notes
        -----
        The function generates an XML document that includes:
        - `<model>`: metadata about the annotation model, including sample rate, start time, and end time.
        - `<dataset>`: contains `<point>` elements that represent individual annotations.
        - `<display>`: defines the display settings for the annotation in the software.
        """
        doc, tag, text = Doc().tagtext()
        doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
        doc.asis('<!DOCTYPE sonic-visualiser>')

        with tag('sv'):
            with tag('data'):
            
                model_string = '<model id="10" name="" sampleRate="{}" start="{}" end="{}" type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="9" subtype="box" minimum="600" maximum="{}" units="Hz" />'.format(sample_rate, 
                                                                        start_m,
                                                                        end_m,
                                                                        1000)
                doc.asis(model_string)
            
            with tag('dataset', id='9', dimensions='2'):

                # Read dataframe or other data structure and add the values here
                # These are added as "point" elements, for example:
                # '<point frame="15360" value="3136.87" duration="1724416" extent="2139.22" label="Cape Robin" />'
                for index, row in dataframe.iterrows():

                    point  = '<point frame="{}" value="{}" duration="{}" extent="{}" label="{}" />'.format(
                        int(row['frame']), 
                        row['value'],
                        int(row['duration']),
                        1500,
                        row['label'])
                    
                    # add the point
                    doc.asis(point)
            with tag('display'):
            
                display_string = '<layer id="2" type="boxes" name="Boxes" model="10"  verticalScale="0"  colourName="White" colour="#ffffff" darkBackground="true" />'
                doc.asis(display_string)

        result = indent(
            doc.getvalue(),
            indentation = ' '*2,
            newline = '\r\n'
        )

        return result
    
