from flask import Flask,request,render_template
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
model = tf.keras.models.load_model('dog-cat1.model')

app.config["IMAGE_UPLOADS"] = "./static"
print('helloooo starting')
X = []
IMG_SIZE = 50
@app.route('/')
def hello_world():
    return render_template('login.html')
database={'nachi':'123','james':'aac','karthik':'asdsf'}

@app.route('/upload-image',methods=['POST','GET'])
def upload_image():
    if request.method == 'POST':
        if request.files:
            image_path = request.files["image"] 
            print('helloooo')
            image_path.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
  
            #data1 = np.expand_dims(data, axis=0)
            #data1 = data1 * 1.0 / 255
            #filename = secure_filename(image.filename)
            #image.save(os.path.join(app.config["IMAGE_UPLOADS"], image))
            #img_array = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            #new_img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            #X.append(new_img_array)
            #feature = X/255.0
            #prediction = model.predict(feature)
            #if prediction == 1:
            #    output = 'DOG'
            #elif prediction == 0:
            #    output = 'CAT'
            img_array1  = image.load_img(image_path, target_size=(50, 50),grayscale=True)
            data1 = np.array(data).reshape(-1,IMG_SIZE,IMG_SIZE,1)
            #img_array1 = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            #new = cv2.resize(img_array1,(IMG_SIZE,IMG_SIZE))
            #new1 = new.reshape(-1,IMG_SIZE,IMG_SIZE,1)
             
            prediction = model.predict(data1)
           
            if prediction[0] == 1:
                predict_value = 'DOG'
            elif prediction[0] == 0:
                predict_value = 'CAT'
                
            print('im hello')
            #im1 = im1.save("geeks.jpg") 
            #print(filename)
            #img_arr = cv2.imread(filename)
            #plt.imshow(img_arr)
            #cv2.imshow('img_arr',img_arr)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #plt.show()
            predict_value = 'Dog'
            #file.save(os.path.join(app.config['IMAGE_UPLOADS'],filename))
            
            #image.save(secure_filename(image.filename))
            #image.save(os.path.join(app.config["IMAGE_UPLOADS"]), image.filename)
            print("image saved")
            return render_template('login.html', prediction_text='Animal is  $ {}'.format(predict_value))

if __name__ == '__main__':
    app.run()
    