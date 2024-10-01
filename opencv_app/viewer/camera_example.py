import sys
import cv2
import numpy as np
from logging import getLogger, basicConfig, INFO
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QComboBox, QPushButton


class ObjectDetector:

    def __init__(self):
        self.logger = getLogger(__name__)
        self.logger.info("ObjectDetector initialized")

    def color_tracking(self, frame):
        """Perform color tracking (for blue objects)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def edge_detection(self, frame):
        """Perform edge detection using Canny edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        frame[edges != 0] = [0, 255, 0]  # Mark the edges in green
        return frame

    def shape_detection(self, frame):
        """Detect shapes like circles and rectangles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            if len(approx) == 3:
                # Triangle
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)
            elif len(approx) == 4:
                # Rectangle
                cv2.drawContours(frame, [approx], 0, (255, 0, 0), 5)
            elif len(approx) > 4:
                # Circle
                cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)
        return frame


class VideoWindow(QWidget):
    def __init__(self, camera_id=0):
        super().__init__(None)
        # Set logger
        basicConfig(level=INFO)
        self.logger = getLogger(__name__)
        # Set cap to None initially
        self.cap = None
        # Initialize the ObjectDetector
        self.detector = ObjectDetector()

        # Set up the UI
        self.setWindowTitle("OpenCV Object Detection with PyQt")
        self.setGeometry(100, 100, 800, 600)

        # Create a label to display the video feed
        self.video_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        # Create a camera selector (ComboBox)
        self.camera_selector = QComboBox(self)
        self.detect_cameras()
        layout.addWidget(self.camera_selector)
        # Function selector (ComboBox)
        self.function_selector = QComboBox(self)
        self.function_selector.addItem("Color Tracking")
        self.function_selector.addItem("Edge Detection")
        self.function_selector.addItem("Shape Detection")
        layout.addWidget(self.function_selector)
        # Create a button to start the selected camera
        self.start_button = QPushButton("Start Camera", self)
        layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_camera)
        # Connect the function selector to update detection method on selection change
        self.function_selector.currentIndexChanged.connect(self.change_detection_function)


        # Set layout
        self.setLayout(layout)

        # Set up a timer to refresh the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def detect_cameras(self):
        """Detect available cameras and add them to the combo box"""
        available_cameras = self.list_cameras()
        for index, camera_name in available_cameras:
            self.camera_selector.addItem(f"Camera {index}: {camera_name}", index)

    def list_cameras(self):
        """Lists available cameras by attempting to connect to them"""
        available_cameras = []
        for i in range(5):  # Check the first 5 possible camera devices
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                available_cameras.append((i, f"Device {i}"))
            cap.release()
        return available_cameras

    def change_detection_function(self):
        """Update the selected detection function when the ComboBox selection changes"""
        self.selected_function = self.function_selector.currentText()


    def start_camera(self):
        """Start the camera feed from the selected camera"""
        # Release previous camera if there was one
        self.logger.info("Starting camera...")
        if self.cap is not None:
            self.cap.release()


        # Get the selected camera index
        selected_camera = self.camera_selector.currentData()
        self.selected_function = self.function_selector.currentText()
        self.logger.info(f"Starting camera: {selected_camera}")

        # Start the selected camera
        self.cap = cv2.VideoCapture(selected_camera)
        self.timer.start(30)
        self.logger.info(f"Camera started: {selected_camera}")

    def update_frame(self):
        # Read a frame from the camera
        ret, frame = self.cap.read()
        if ret:
            match self.selected_function:
                case "Color Tracking":
                    frame = self.detector.color_tracking(frame)
                case "Edge Detection":
                    frame = self.detector.edge_detection(frame)
                case "Shape Detection":
                    frame = self.detector.shape_detection(frame)
                case _:
                    self.logger.error("Invalid function selected")

            # Convert the OpenCV frame (BGR) to a format Qt can display (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))



    def closeEvent(self, event):
        # Release the camera when closing the window
        self.cap.release()


def main():
    app = QApplication(sys.argv)
    window = VideoWindow(1)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
