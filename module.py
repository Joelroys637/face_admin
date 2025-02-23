import os
import face_recognition
import cv2
import numpy as np
from PIL import Image
import streamlit as st

def capture_and_store_images(person_name, num_images=15):
    """
    Capture and store multiple images for a person.
    """
    # Create a directory for the person if it doesn't exist
    if not os.path.exists(person_name):
        os.makedirs(person_name)

    # Capture images using the webcam
    cap = cv2.VideoCapture(0)

    st.write(f"Capturing {num_images} images for {person_name}. Press 'q' to capture each image.")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        cv2.imshow('Capture Image', frame)

        # Save the image when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            image_path = os.path.join(person_name, f"{person_name}_{count}.jpg")
            cv2.imwrite(image_path, frame)
            st.write(f"Image {count + 1} captured and saved.")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    st.success(f"Captured {num_images} images for {person_name}.")

def load_known_faces(person_name):
    """
    Load known faces for a person from the stored images.
    """
    known_face_encodings = []
    known_face_names = []

    person_folder = person_name
    for filename in os.listdir(person_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(person_folder, filename)
            image = face_recognition.load_image_file(image_path)
            try:
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
            except IndexError:
                st.warning(f"No face found in {filename}. Skipping.")

    return known_face_encodings, known_face_names

def recognize_face(known_face_encodings, known_face_names):
    """
    Recognize a face from the webcam feed.
    """
    cap = cv2.VideoCapture(0)

    st.write("Press 'q' to capture an image for recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        cv2.imshow('Recognize Face', frame)

        # Capture an image for recognition when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    st.success(f"Recognized: {name}")
                else:
                    st.warning("Unknown face")

            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Face Recognition System")

    person_name = st.text_input("Enter the person's name:")
    if st.button("Capture Images"):
        capture_and_store_images(person_name)

    if st.button("Recognize Face"):
        known_face_encodings, known_face_names = load_known_faces(person_name)
        recognize_face(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
