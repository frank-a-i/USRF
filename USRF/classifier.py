import cv2
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier


class CropData:
    class AxisData:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def __init__ (self, x, y, width, height, is_center_point = False):
        if is_center_point:
            x = self.AxisData(int(x - width/2.0), int(x + width/2.0))
            y = self.AxisData(int(y - height/2.0), int(y + height/2.0))
        else:
            x = self.AxisData(x, x + width)
            y = self.AxisData(y, y + height)

        self.x = x
        self.y = y

class CustomDimensions:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class USRFClf:
    def __init__ (self, initial_position, 
                        regions_of_interest: List[CropData], 
                        n_training_frames = 20, 
                        neighborhood_search_size = 5,
                        object_size = CustomDimensions(60,60),
                        verbose = False):
        self._training_data = []
        self._regions_of_interest = regions_of_interest
        self._past_position = int(initial_position[0]), int(initial_position[1])
        self._n_training_frames = n_training_frames
        self._neighborhood_search_size = neighborhood_search_size
        self._object_size = object_size
        self._object = None
        self._verbose = verbose
        self._rough_tracker = None
        
    def _crop_img(self, frame, crop_data):
        return frame[crop_data.y.start:crop_data.y.end, crop_data.x.start:crop_data.x.end]

    def _extract_patches(self, frame: cv2.Mat) -> List[cv2.Mat]:
        patches = []
        for region in self._regions_of_interest:
            patches.append(self._crop_img(frame, region))
        return patches

    def _neighborhood_search(self, frame):
        pass

    def init(self, frame, is_center_point = True):
        self._object = self._crop_img(frame, CropData(self._past_position[0], self._past_position[1], 
                                                      self._object_size.width, self._object_size.height,
                                                      is_center_point))
        if self._verbose:
            self._show_cur_roi(self._object, non_destroying=False)

    def _fine_tracking(self, frame):
        highscore = -1
        estimate_position = self._past_position
        for cur_y in range(self._past_position[1]-self._neighborhood_search_size,
                           self._past_position[1]+self._neighborhood_search_size):
            for cur_x in range(self._past_position[0]-self._neighborhood_search_size,
                               self._past_position[0]+self._neighborhood_search_size):
                cur_sample = self._crop_img(frame, CropData(cur_x, 
                                                            cur_y, 
                                                            self._object_size.width, 
                                                            self._object_size.height, True)).flatten()
                #print(self._pearson_correlation_coefficient(cur_sample, self._object.flatten()))
                curscore = np.corrcoef(cur_sample, self._object.flatten())[0][1]
                if curscore > highscore:
                    highscore = curscore
                    estimate_position = cur_x, cur_y

        if self._verbose:
            print("Estimated coordinates from fine tracking ", estimate_position) 
        
        self._past_position = estimate_position

    def _show_cur_roi(self, roi, window_timeout = 1000, non_destroying = True):
        window_title = "ROI to track"
        cv2.imshow(window_title, roi)

        # continue with timeout or window gets closed
        view_debug_window = True
        while view_debug_window:
            timeout = cv2.waitKey(window_timeout) == -1
            if timeout or cv2.getWindowProperty(window_title,cv2.WND_PROP_VISIBLE) < 1:
                view_debug_window = False
        if not non_destroying:
            cv2.destroyAllWindows()

    def _show_frame_and_estimate(self, frame, ground_truth = None, window_timeout  = 50, non_destroying = True, roi_border_thickness = 1, roi_border_color = (200, 200, 200)):

        def _add_cross(frame, pos, color, thickness = 2, scaling = 1.0, alpha = 0.5):
            marker_img = frame.copy()
            cv2.line(marker_img, (pos[0] - int(self._object.shape[0]/2.0 * scaling), pos[1]),
                                 (pos[0] + int(self._object.shape[0]/2.0 * scaling), pos[1]),
                                 color,
                                 thickness)
            cv2.line(marker_img, (pos[0] , pos[1] - int(self._object.shape[1]/2.0 * scaling)),
                                 (pos[0] , pos[1] + int(self._object.shape[1]/2.0 * scaling)),
                                 color,
                                 thickness)
            return cv2.addWeighted(marker_img, alpha, frame, 1 - alpha, 0)
        
        roi_img = self._crop_img(frame, CropData(self._past_position[0],
                                                 self._past_position[1], 
                                                 self._object_size.width,
                                                 self._object_size.height,
                                                 True))
        # cur state
        frame[0:self._object.shape[0], 0:self._object.shape[1]] = roi_img
        # init state
        frame[self._object.shape[0]:self._object.shape[0] * 2,
                                 0 :self._object.shape[1] ] = self._object
        # extraction windows
        for roi in self._regions_of_interest:
            frame = cv2.rectangle(frame, (roi.x.start, roi.y.start), (roi.x.end, roi.y.end), roi_border_color, roi_border_thickness)

        window_title = "Frame and algo estimate"
        # set labels
        frame = cv2.putText(frame, 'Current state', (int(self._object.shape[0] * 1.1), int(self._object.shape[1]/2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        frame = cv2.putText(frame, 'Init state', (int(self._object.shape[0] * 1.1), int(self._object.shape[1]*1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        # set tracking cross
        frame = _add_cross(frame, self._past_position, (0, 124, 255))
        if ground_truth is not None:
            ground_truth = int(ground_truth["x"]), int(ground_truth["y"])
            frame = _add_cross(frame, ground_truth, (10, 200, 10))
        
        cv2.imshow(window_title, frame)
        # continue with timeout or window gets closed
        view_debug_window = True
        while view_debug_window:
            timeout = cv2.waitKey(window_timeout) == -1
            if timeout or cv2.getWindowProperty(window_title,cv2.WND_PROP_VISIBLE) < 1:
                view_debug_window = False
        if not non_destroying:
            cv2.destroyAllWindows()


    def process_data(self, frame, annotation = None):
        if self._object is None:
            raise RuntimeError("Algorithm has not been initialized yet, cannot proceed.")

        if len(self._training_data) < self._n_training_frames + 100:    
            patches = self._extract_patches(frame)
            self._fine_tracking(frame)
            self._training_data.append([patches, self._past_position])
            self._show_frame_and_estimate(frame, ground_truth=annotation)
        elif self._rough_tracker is None:
            self._rough_tracker = RandomForestClassifier()
            training_data = np.array(self._training_data)
            X = training_data[:, 1]
            y = training_data[:, 0]
            knn = NearestNeighbors(n_neighbors=5).fit(X)
            print(knn.kneighbors(X))
            self._rough_tracker.fit(self._training_data[:,0], self._training_data[:,1])
        else:
            pass