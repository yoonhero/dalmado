import os
import cv2  # pip install opencv-python
import time
import numpy as np  # pip install numpy
import mediapipe as mp  # pip install mediapipe

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)


current_effect = None

current_effect_icons = {
    "eye": "./dalma/eye_left.png",
    "shade": None,
    "nose": "./dalma/nose.png",
    "cigar": None,
    "mustache": None,
    "mask": "./dalma/mouth.png"
}


def get_landmarks(image):
    landmarks = []
    height, width = image.shape[0:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mesh_results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            current = {}
            for i, landmark in enumerate(face_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                relative_x = int(x * width)
                relative_y = int(y * height)
                current[i + 1] = (relative_x, relative_y)
            landmarks.append(current)

    return landmarks


def get_effect_cordinates(landmarks):
    effect_cordinates = {
        "eye_left": (landmarks[30], (landmarks[158][0], landmarks[145][1])),
        "eye_right": (landmarks[287], (landmarks[260][0], landmarks[381][1])),
        "shade": (landmarks[71], (landmarks[294][0], landmarks[119][1])),
        "nose": ((landmarks[51][0], landmarks[4][1]), (landmarks[281][0], landmarks[3][1])),
        "cigar": (landmarks[16], (landmarks[273][0], landmarks[195][1])),
        "mustache": ((landmarks[148][0], landmarks[3][1]), ((landmarks[148][0]+(landmarks[3][0]-landmarks[148][0])*2), landmarks[41][1])),
        "mask": (landmarks[124], (landmarks[324][0], landmarks[153][1]))
    }

    return effect_cordinates


def remove_image_whitespace(image, blend, x, y, threshold=225):
    for i in range(blend.shape[0]):
        for j in range(blend.shape[1]):
            for k in range(3):
                if blend[i][j][k] > threshold:
                    blend[i][j][k] = image[i + y][j + x][k]


def add_effect(image, effect, icon_path, cordinates):
    item = cv2.imread(icon_path)
    pt1, pt2 = cordinates[effect]
    x, y, x_w, y_h = pt1[0], pt1[1], pt2[0], pt2[1]
    cropped = image[y:y_h, x:x_w, :]
    h, w, _ = cropped.shape
    item = cv2.resize(item, (w, h))
    blend = cv2.addWeighted(cropped, 0, item, 1.0, 0)

    return blend, x, y, x_w, y_h

# current_effect_icons[effect] = icon_path


def draw_face_effects(image, cordinates):
    for effect, icon_path in current_effect_icons.items():
        if effect == "eye":
            for effect in ["eye_left", "eye_right"]:
                if icon_path is not None:
                    blend, x, y, x_w, y_h = add_effect(
                        image, effect, icon_path, cordinates)
                    remove_image_whitespace(image, blend, x, y)
                    image[y:y_h, x:x_w, :] = blend
        else:
            if icon_path is not None:
                blend, x, y, x_w, y_h = add_effect(
                    image, effect, icon_path, cordinates)
                remove_image_whitespace(image, blend, x, y)
                image[y:y_h, x:x_w, :] = blend


def app(video_source):
    global current_effect

    source = cv2.VideoCapture(video_source)

    while True:
        ret, frame = source.read()
        if ret:
            current_time = time.time()
            height, width, _ = frame.shape
            image = cv2.resize(frame, (950, 650))

            landmarks = get_landmarks(image)
            faces = len(landmarks)

            if faces > 0:
                for l in landmarks:
                    cordinates = get_effect_cordinates(l)
                    draw_face_effects(image, cordinates)

            cv2.imshow("Live Face Effects", image)
            k = cv2.waitKey(1)

        else:
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app(0)
