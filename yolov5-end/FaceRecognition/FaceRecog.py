import face_recognition
import cv2
import numpy as np
import mediapipe as mp
import time

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
user_image = face_recognition.load_image_file("FaceRecognition/user.jpg")
# new_user_image = face_recognition.load_image_file("FaceRecognition/user.jpg")

# user_image = cv2.imread("data/user.jpg")
user_face_encoding = face_recognition.face_encodings(user_image)[0]
# new_user_face_encoding = face_recognition.face_encodings(user_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    # new_user_face_encoding,
    user_face_encoding
]
known_face_names = [
    # "New_User",
    "User"
]

# Initialize some variables
process_this_frame = True

w,h = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
T,f = 0,0

def faceRecognizer(frame):
    face_locations = []
    face_encodings = []
    face_names = []
    small_frame = frame

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if True:
        # Find all the faces and face encodings in the current frame of video
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # print("face_locations:",face_locations)
        # print("face_encodings:",face_encodings)

        face_names = []
        face_appeared = [0]
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # print("matches:",matches)
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                # if some faces has already be regared as this known face
                # compare the distances between two faces and the known face
                if (face_appeared[best_match_index] == 0) | (face_appeared[best_match_index] > face_distances[best_match_index]):
                    # compare the distances between two faces and the known face
                    name = known_face_names[best_match_index]
                    face_appeared[best_match_index] = face_distances[best_match_index]

            face_names.append(name)

    return face_locations, face_names