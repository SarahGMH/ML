import cv2 
import face_recognition
# Charger une image 
image = cv2.imread('me.jpg') 
# Afficher l'image 
cv2.imshow('Image', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

# Convertir en niveaux de gris 
gray_image = cv2.cvtColor(image, 
cv2.COLOR_BGR2GRAY) 
# Sauvegarder ou afficher 
cv2.imshow('Grayscale Image', gray_image) 
cv2.imwrite('gray_image.jpg', gray_image)

# Normalisation entre 0 et 1 
normalized_image = image / 255.0 
# Normalisation à une plage personnalisée 
norm_image = cv2.normalize(image, None, 
alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)



nown_image = face_recognition.load_image_file('file1.jpg')
nknown_image = face_recognition.load_image_file('file2.jpg')
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
results = face_recognition.compare_faces([biden_encoding], 
unknown_encoding)
print(str(results))
