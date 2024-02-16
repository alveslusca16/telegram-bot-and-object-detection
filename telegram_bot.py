import telebot
import math
import time
from ultralytics import YOLO
import cv2
import cvzone
CHAVE_API = 'Token bot of Telegram here'
bot = telebot.TeleBot(CHAVE_API)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# For Webcam (0 is your default camera, 1 is your another camera)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


@bot.message_handler(commands=["start"])
def responder(mensagem):
    texto = """
    Bot iniciado!
    """
    print(mensagem)
    bot.reply_to(mensagem, texto)

    model = YOLO("../Yolo-Weights/yolov8l.pt")

    while True:
        sucess, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                w, h = x2-x1, y2-y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence

                conf = math.ceil((box.conf[0]*100))/100

                # Class Name

                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(
                    0, x1), max(35, y1)), scale=1, thickness=1)

                if classNames[cls] == "knife":
                    cv2.imwrite('imagens_detectadas/imgdetectada.png', img)
                    mensagem_suspeita = "Objeto Suspeito detectado!"
                    bot.send_message(mensagem.chat.id, mensagem_suspeita)
                    with open('imagens_detectadas/imgdetectada.png', 'rb') as photo:
                        bot.send_photo(mensagem.chat.id, photo)
                    time.sleep(20)


bot.polling()
cap.release()
