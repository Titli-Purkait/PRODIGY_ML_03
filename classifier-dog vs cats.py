import cv2
import numpy as np
from sklearn import svm

def process_image(path, show=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    
    if show:
        cv2.imshow("Input Image", img)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()
    
    return img.flatten()

# Load and show images
cat_img = process_image("C:/Users/titli/OneDrive/Desktop/svm- cats vs dogs/cat jpg.1.jpg", show=True)
dog_img = process_image("C:/Users/titli/OneDrive/Desktop/svm- cats vs dogs/dog jpg.1.jpg", show=True)

X = np.array([cat_img, dog_img])
y = np.array([0, 1])  # 0 = Cat, 1 = Dog

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Choose which one to predict
prediction = clf.predict([dog_img])
print("Prediction:", "Dog " if prediction[0] == 1 else "Cat ")


