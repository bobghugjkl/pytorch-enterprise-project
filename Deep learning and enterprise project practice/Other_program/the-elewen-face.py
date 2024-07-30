"""如何获取拍到的图片"""
import cv2
import dlib
predictor_path = ".\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, cv_img = cap.read()
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    dets = detector(img, 0)
    shapes = []
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("q pressed")
        break
    else:
        for k, d in enumerate(dets):
            shape = predictor(img, d)
            for index, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                cv2.circle(img, pt_pos, 1, (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(index + 1),pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
            win.clear_overlay()
            win.set_image(img)
            if len(shapes) != 0:
                for i in range(len(shapes)):
                    win.add_overlay(shapes[i])
            win.add_overlay(dets)
cap.release()
cv2.destroyAllWindows()