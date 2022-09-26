# FaceRecognitionByWebcam


# How to use
First to recognise a face we have to input an image to the model   
To do so we can add the persons image in the data folder and run Loadfaces file    
Or to get the input directly by web cam just run Addface file    
It will take the webcam input and encode that face    
Now we are free to run faceRecognition file


# How it works
First after getting the image input by data folder or by webcam input, the facerec file load that data    
By face recognition module those loaded data will be encoded and stored in encodings file   
If repeated image is loaded here, facerec will detect the repeated image and neglet that to add to encodings file   
By that encodings face recognition will compare the distance metric and classify the faces    
