import sys

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel


class MainWidget(QWidget):
    def __init__(
            self,
            initial_callback: callable,
            auto_pilot_callback: callable,
            disable_auto_pilot_callback: callable,
            capture_callback: callable,
            shutdown_callback: callable,
    ):
        super().__init__()

        self.initial_callback = initial_callback
        self.auto_pilot_callback = auto_pilot_callback
        self.disable_auto_pilot_callback = disable_auto_pilot_callback
        self.capture_callback = capture_callback
        self.exit_callback = shutdown_callback

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setText("Loading ...")

        self._initial_button = QPushButton("Connect")
        self._auto_pilot_button = QPushButton("Start Auto Pilot")
        self._disable_auto_pilot_button = QPushButton("Stop Auto Pilot")
        self._disable_auto_pilot_button.setDisabled(True)

        self._capture_button = QPushButton("Capture")
        self._exit_button = QPushButton("Exit")

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout()

        video_layout = QVBoxLayout()
        video_layout.addWidget(self._video_label)
        video_layout.addWidget(self._capture_button)
        video_layout.addWidget(self._exit_button)

        option_layout = QVBoxLayout()
        option_layout.addWidget(self._initial_button)
        option_layout.addWidget(self._auto_pilot_button)
        option_layout.addWidget(self._disable_auto_pilot_button)
        option_layout.addStretch()
        option_layout.setSpacing(10)

        main_layout.addLayout(video_layout, 3)
        main_layout.addLayout(option_layout, 1)

        self.setLayout(main_layout)

        self._exit_button.clicked.connect(self._exit_callback)
        self._initial_button.clicked.connect(self._initial_callback)
        self._auto_pilot_button.clicked.connect(self._auto_pilot_callback)
        self._disable_auto_pilot_button.clicked.connect(self._disable_auto_pilot_callback)
        self._capture_button.clicked.connect(self._capture_callback)

        self.setWindowTitle("Tello Control")
        self.setGeometry(0, 0, 1080, 720)

    def _exit_callback(self):
        self.exit_callback()
        self.close()
        self.destroy()
        self.deleteLater()

    def _auto_pilot_callback(self):
        self.auto_pilot_callback()
        self._auto_pilot_button.setDisabled(True)
        self._disable_auto_pilot_button.setDisabled(False)

    def _capture_callback(self):
        self._video_label.setText("Capturing ...")
        self.capture_callback()

    def _disable_auto_pilot_callback(self):
        self.disable_auto_pilot_callback()
        self._auto_pilot_button.setDisabled(False)
        self._disable_auto_pilot_button.setDisabled(True)

    def _initial_callback(self):
        self._initial_button.setDisabled(True)
        self.initial_callback()

    def update_video_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self._video_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = MainWidget(
        initial_callback=lambda: print("Initial"),
        auto_pilot_callback=lambda: print("Auto Pilot"),
        disable_auto_pilot_callback=lambda: print("Disable Auto Pilot"),
        capture_callback=lambda: print("Capture"),
        shutdown_callback=lambda: print("Shutdown")
    )
    view.show()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            view.update_video_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    sys.exit(app.exec_())
