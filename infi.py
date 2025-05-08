import sys
import pickle
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import mediapipe as mp

class HandGestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.timer = None
        self.initUI()
        self.cap = None 

    def initUI(self):
        self.setWindowTitle('Hand Gesture Recognition')
        self.setGeometry(100, 100, 640, 530)

        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.startButton = QPushButton('Start', self)
        self.startButton.clicked.connect(self.startDetection)
        self.layout.addWidget(self.startButton)

        self.endButton = QPushButton('End', self)
        self.endButton.clicked.connect(self.endDetection)
        self.endButton.setEnabled(False)
        self.layout.addWidget(self.endButton)

        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        self.model_dict = pickle.load(open('./model.pkl', 'rb'))
        self.model = self.model_dict['model']
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K' , 11:'L' , 12:'M' , 13:'N' , 14:'O' , 15:'P' , 16:'Q' , 17:'R' ,18:'S' , 19:'T' , 20:'U' , 21:'V' , 22:'W' , 23:'X' , 24:'Y' , 25:'Z' , 26:'get lost' ,  27:'BHUPENDRA JOGI' , 28:'HELLO' , 29:'I LOVE YOU' , 30:'SALUTE' , 31:'THANK YOU'}
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detectGesture)

    def startDetection(self):
        if self.cap is None or not self.cap.isOpened():  # Check if camera is not already open
            self.startButton.setEnabled(False)
            self.endButton.setEnabled(True)
            self.cap = cv2.VideoCapture(0)  # Open the camera
            self.timer.start(30)
    def endDetection(self):
        self.startButton.setEnabled(True)
        self.endButton.setEnabled(False)
        self.timer.stop()
        self.cap.release()  # Close the camera feed
        self.label.clear()  # Clear the label

    def detectGesture(self):
        ret, frame = self.cap.read()
        if ret:
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            # Convert the frame to a format suitable for displaying in PyQt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgb_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.label.setPixmap(QPixmap.fromImage(p))

def main():
    app = QApplication(sys.argv)
    window = HandGestureApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()