from typing import List
from glob2 import glob
import os
import logging
import cv2

class USRFDataHandler:
    def __init__(self, data_directory: str, limit_to_seq = [], img_file_type = ".png", extract_data = True):
        
        self._dataset = self._extract_datafiles(data_directory, limit_to_seq, img_file_type)
        if extract_data:
            self._create_processible_data()

        feedback_msg = "Finished file parsing"
        print(feedback_msg)
        logging.info(feedback_msg)

        self._sequences = None
        self._last_frame = None

    def next_data(self):
        # TODO: turn to generator

        def maintenance():
            finished = False
            if self._sequences is None:
                self._sequences = list(self._dataset.keys())
                self._last_frame = self._dataset[self._sequences[0]]["X"][0]["data"]
            elif self._last_frame is None or len(self._dataset[self._sequences[0]]["X"]) == 0:
                if len(self._sequences) > 1:
                    del self._sequences[0]
                    user_msg = "Processing now {}".format(self._sequences[0])
                    print(user_msg)
                    logging.info(user_msg)
                    self._last_frame = self._dataset[self._sequences[0]]["X"][0]["data"]

                else: # no more data
                    finished = True

            return finished
        
        def compose_data():
            cur_frame = self._dataset[self._sequences[0]]["X"][0]["data"]
            cur_ID = self._dataset[self._sequences[0]]["X"][0]["ID"]
            annotation = self._dataset[self._sequences[0]]["y"].get(cur_ID, None)
            del self._dataset[self._sequences[0]]["X"][0]

            return cur_frame, annotation

        
        processedAllData =  maintenance()
        if processedAllData:
            return None

        return compose_data()


    def _create_processible_data(self):
        """
        Turns the dataset of file paths to machine processible data types

        While the dict structure is taken as it was before, the images are turned into cv2 images and the
        annotations to a dictionary with coordinates per seq ID

        Raises:
            RuntimeError: raised whenever self._dataset has an unexpected state
        """
        logging.info("Turning dataset paths to machine processible data.")
        if self._dataset is None or len(self._dataset.keys()) == 0:
            raise RuntimeError("Internal error: No dataset available, previous file scan is flaky.")

        processible_dataset = dict()
        for sequence in self._dataset.keys():
            progress_msg = "Working on {}".format(sequence)
            print(progress_msg)
            logging.info(progress_msg)
            processible_sequence_frames = []
            for frame_path in self._dataset[sequence]["X"]:
                processible_sequence_frames.append({"ID": int(os.path.basename(frame_path).split(".")[0]), "data":cv2.imread(frame_path)})
            
            processible_annotations = dict()
            with open(self._dataset[sequence]["y"], "r") as annotation_file:
                for line in annotation_file:
                    features = line.split()
                    processible_annotations.update({int(float(features[0])): {"x": float(features[1]), "y": float(features[2])}})
            processible_dataset.update({sequence:{"X": processible_sequence_frames, "y": processible_annotations}})
        
        self._dataset = processible_dataset

    def _extract_datafiles(self, data_directory: str, limit_to_seq: List[str], img_file_type: str):
        """
        Gathers the individual files for each frame and annotation per sequence

        Args:
            data_directory (str): base mining directory
            limit_to_seq (List[str]): a list of relevant sequences that should not be ignored
            img_file_type (str): data type of the expected frames

        Raises:
            IOError: if the data directory is not reachable at all
            IOError: if the data directory is empty

        Returns:
            dict: filepaths per sequence
        """
        datasets = dict()
        # verify base directory
        if not os.path.isdir(data_directory):
            raise IOError("Cannot find data directory: '{}'".format(data_directory))

        # get sequences 
        seq_paths = glob(os.path.join(data_directory, "*"))
        if len(seq_paths) == 0:
            raise IOError("Data directory appears to be empty: '{}'".format(data_directory))
        
        # extract data per sequence
        for cur_path in seq_paths:
            seq_name = os.path.basename(cur_path)
            if seq_name in limit_to_seq:
                if not os.path.isdir(os.path.join(cur_path, "Data")):
                    logging.warning("Missing Data in sequence directory '{}'".format(cur_path))
                    continue
                elif not os.path.isdir(os.path.join(cur_path, "Annotation")):
                    logging.warning("Missing Annotation in sequence directory '{}'".format(cur_path))
                    continue

                seq_data_path = os.path.join(cur_path, "Data")
                seq_anno_path = os.path.join(cur_path, "Annotation")
                seq_frames = glob(os.path.join(seq_data_path, "*{}".format(img_file_type)))
                seq_frames.sort()
                
                if len(seq_frames) == 0:
                    logging.warning("No frames found under: '{}'".format(seq_data_path))
                    continue
                seq_annotation = glob(os.path.join(seq_anno_path, "*.txt"))[0] # there is only one annotation file
                if len(seq_annotation) == 0:
                    logging.warning("No annotation found under: '{}'".format(seq_anno_path))
                    continue

                seq_dataset ={"X": seq_frames,
                              "y": seq_annotation}
                datasets.update({seq_name: seq_dataset})

        return datasets


        