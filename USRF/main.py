import cv2
from USRF.classifier import USRFClf, CustomDimensions, CropData
from USRF.data_handler import USRFDataHandler
import logging

def main():
    logging.basicConfig(filename='USRF.log', level=logging.INFO)
    

    data_directory = "/home/frank-ai/Nextcloud/Projekte/Coding/USRF/Data/Training/"
    limit_to_seq = ["CIL-02", "CIL-01"]
    data_handler = USRFDataHandler(data_directory, limit_to_seq)

    frame, annotation = data_handler.next_data()

    regions_of_interest = [CropData(60, 120, 30, 30), CropData(120, 120, 30, 30)]
    algo = USRFClf(initial_position=[annotation["x"], annotation["y"]],
                   regions_of_interest=regions_of_interest,
                   verbose = True)
    algo.init(frame)

    finished = False
    while not finished:
        package = data_handler.next_data()
        if not package:
            break
        frame, annotation = package
        algo.process_data(frame, annotation)


if __name__ == "__main__":
    main()