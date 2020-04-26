from flask import Flask, render_template, redirect, request, flash, url_for
import tensorflow as tf
import os

UPLOAD_FOLDER = 'static/images/upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0
    return image

def predict(path):
    img_list = []
    img_list.append(tf.io.read_file(path))
    ds = tf.data.Dataset.from_tensor_slices(img_list)
    ds_processed = ds.map(preprocess_image)
    ds_processed = ds_processed.batch(1)
    model = tf.keras.models.load_model('catdog_classifier_Xception.h5')
    prediction = model.predict(ds_processed)
    return prediction

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('result',filename = filename))   
    return render_template('home.html')

@app.route('/result/<filename>')
def result(filename):
    prediction = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('result.html', prediction=prediction, filename=filename)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)
 