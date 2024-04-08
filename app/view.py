import sys

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel


class MainWidget(QWidget):
    def __init__(
            self,
            initial_callback: callable,
            auto_pilot_callback: callable,
            capture_callback: callable,
            tracking_callback: callable,
            pose_command_callback: callable,
            exit_callback: callable,
    ):
        super().__init__()

        self.initial_callback = initial_callback
        self.auto_pilot_callback = auto_pilot_callback
        self.capture_callback = capture_callback
        self.tracking_callback = tracking_callback
        self.pose_command_callback = pose_command_callback
        self.exit_callback = exit_callback

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Loading ...")

        self.initial_button = QPushButton("Connect")
        self.auto_pilot_button = QPushButton("Auto Pilot Mode")
        self.track_target_button = QPushButton("Tracking Mode")
        self.pose_command_button = QPushButton("Pose Command Mode")
        self.capture_button = QPushButton("Capture")
        self.exit_button = QPushButton("Exit")

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.capture_button)
        video_layout.addWidget(self.exit_button)

        option_layout = QVBoxLayout()
        option_layout.addWidget(self.initial_button)
        option_layout.addWidget(self.auto_pilot_button)
        option_layout.addWidget(self.track_target_button)
        option_layout.addWidget(self.pose_command_button)
        option_layout.addStretch()
        option_layout.setSpacing(10)

        main_layout.addLayout(video_layout, 3)
        main_layout.addLayout(option_layout, 1)

        self.setLayout(main_layout)

        self.exit_button.clicked.connect(self.exit_application)
        self.initial_button.clicked.connect(self.initial_callback)
        self.auto_pilot_button.clicked.connect(self.auto_pilot_callback)
        self.capture_button.clicked.connect(self.capture_callback)
        self.track_target_button.clicked.connect(self.tracking_callback)
        self.pose_command_button.clicked.connect(self.pose_command_callback)

        self.setWindowTitle("Tello Control")
        self.setGeometry(0, 0, 1080, 720)
        self.show()

    def exit_application(self):
        self.exit_callback()
        sys.exit()

    def update_video_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = MainWidget(
        initial_callback=lambda: print("Initial"),
        auto_pilot_callback=lambda: print("Auto Pilot"),
        capture_callback=lambda: print("Capture"),
        tracking_callback=lambda: print("Tracking"),
        pose_command_callback=lambda: print("Pose Command"),
        exit_callback=lambda: print("Exit")
    )
    sys.exit(app.exec_())
