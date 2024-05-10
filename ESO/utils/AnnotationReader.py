import pandas as pd
import os
import librosa
from xml.dom import minidom
import soundfile as sf
from yattag import Doc, indent
from pathlib import Path


class AnnotationReader:
    def __init__(self, path, annotation_file_name, file_type, audio_extension):
        self.path = path
        self.annotation_file_name = annotation_file_name
        self.file_type = file_type
        self.audio_extension = audio_extension
        

    def read_audio_file(self, file_name, species_folder):
        """
        file_name: string, name of file including extension, e.g. "audio1.wav"

        """
        # Get the path to the file
        audio_folder = os.path.join(species_folder, "Audio", file_name)

        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)

        f = sf.SoundFile(audio_folder)
        seconds = f.frames / f.samplerate

        return audio_amps, audio_sample_rate, seconds

    def get_annotation_information(self, annotation_folder, sufix_file ):
        if self.file_type == "svl":
            # Process the .svl xml file
            
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

            if len(itemlist) > 0:
                file_name_no_extension = self.annotation_file_name
                audio_amps, original_sample_rate, audio_seconds = self.read_audio_file(
                    file_name_no_extension + self.audio_extension, self.path
                )

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
            return df_svl_gibbons, file_name_no_extension + ".wav"

        if self.file_type == "raven_caovitgibbons":
            df = pd.read_csv(
                os.path.join(self.path, "Annotations", self.annotation_file_name),
                sep="\t",
            )

            start_time = []
            end_time = []
            labels = []
            audio_file_name = ""

            for index, row in df.iterrows():
                audio_file_name = row["Begin File"]
                start_time.append(row["Begin Time (s)"])
                end_time.append(row["End Time (s)"])
                labels.append(row["Label"])

            df_raven_gibbons = pd.DataFrame(
                {"Start": start_time, "End": end_time, "Label": labels}
            )

            return df_raven_gibbons, audio_file_name
    
    def get_annotation_information_testing(self):

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

        if (len(itemlist) > 0):

            file_name_no_extension = self.annotation_file_name
            audio_amps, original_sample_rate, audio_seconds = self.read_audio_file(
                    file_name_no_extension + self.audio_extension, self.path
                )

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
        return df_svl_gibbons, sampleRate, start_m, end_m, len(audio_amps), audio_seconds


    def dataframe_to_svl(self, dataframe, sample_rate, start_m, end_m):

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
    


    

    def get_audio_location(self):
        if "-" in self.annotation_file_name:
            return (
                "/".join(
                    self.annotation_file_name[
                        : self.annotation_file_name.rfind("-")
                    ].split("-")
                )
                + "/"
            )
        else:
            return ""
