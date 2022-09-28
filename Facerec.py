import face_recognition
import cv2
import os
import glob
import numpy as np
import time
import pickle 


with open('encodings', 'rb') as f: 
    faces = pickle.load(f)
    names=pickle.load(f)


class Facerec:
    def __init__(self):
        
        self.known_face_encodings = faces
        self.known_face_names = names

  
        self.frame_resizing = 1

    def load_encoding_images(self, images_path):
       
    
        images_path = glob.glob(os.path.join(images_path, "*.*"))

       

     
        for img_path in images_path:
            
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

           
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
      
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            if len(self.known_face_encodings)>=1:
                matches = face_recognition.compare_faces(self.known_face_encodings, img_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, img_encoding)

                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.3:
                    print(f"This face is already recorded as {self.known_face_names[best_match_index]} no need of {filename}")
                else:
                    faces.append(img_encoding)
                    names.append(filename)
                    self.known_face_encodings = faces
                    self.known_face_names = names
                    with open('encodings', 'wb') as f: 
                        pickle.dump(faces, f) 
                        pickle.dump(names,f)
            else:
                faces.append(img_encoding)
                names.append(filename)
                self.known_face_encodings = faces
                self.known_face_names = names
                with open('encodings', 'wb') as f: 
                    pickle.dump(faces, f) 
                    pickle.dump(names,f)
            break          
                

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
       
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_acc=[]
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            acc=0.0
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                acc=1-face_distances[best_match_index]
            face_names.append(name)
            face_acc.append(acc)


        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names,face_acc
       