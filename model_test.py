from keras.models import load_model
#from keras.preprocessing import image
import numpy as np

import cv2
from keras.preprocessing.image import img_to_array
# dimensions of our images
img_width, img_height = 150, 150

test_data_dir = 'data/test'
# load the model we saved

model = load_model('modelTest.model')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
import os
print(model.summary())

test_labels = []
test_preds = []

for img_name in os.listdir(test_data_dir):
    # predicting images
    if 'cat' in img_name:
        test_labels.append(0)
    elif 'dog' in img_name:
        test_labels.append(1)


    img = cv2.imread(test_data_dir + '/' + img_name)
    if img is None:
        break
    orig = img.copy()
    # pre-process the image for classification
    image = cv2.resize(img, (img_width, img_height))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    classes = model.predict_classes(image, batch_size=10)
    pred = model.predict(image)
    print(pred)
    # print the classes, the images belong to
    print(classes)
    test_preds.append(classes[0])


    # build the label
    label1 = "Cat" if classes[0]==0 else "Dog"

    proba = pred[0][classes[0]]
    label = "{}: {:.2f}%".format(label1, proba * 100)

    # draw the label on the image
    output = cv2.resize(orig,(img_width,img_height))
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)
   # plt.show()

    


