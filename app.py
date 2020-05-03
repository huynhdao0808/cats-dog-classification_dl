from flask import Flask, render_template, redirect, request, flash, url_for
import tensorflow as tf
import os
import sqlite3

conn = sqlite3.connect('predictions.db')
cur = conn.cursor()

UPLOAD_FOLDER = 'static/images/upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

class Prediction:
    def __init__(self, id, file_name, predict):
        self.id = id
        self.file_name = file_name
        self.predict = predict
        self.feed_back = 1

    def __repr__(self):
        return "ID: {}, file_name: {}, predict: {},  feedback: {}".format(self.id, self.file_name, self.predict, self.feed_back)

    def save_into_db(self):
        conn = sqlite3.connect('predictions.db')
        cur = conn.cursor()
        query = """
            INSERT INTO predictions (file_name, predict, feed_back)
            VALUES (?, ? , ?);
        """
        val = (self.file_name, self.predict, self.feed_back)
        try:
            cur.execute(query, val)
            self.id = cur.lastrowid
            conn.commit()
        except Exception as err:
            print('ERROR BY INSERT:', err)

def create_categories_table():
    query ="""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name VARCHAR(255),
        predict REAL,
        feed_back INT,
        create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    try:
        cur.execute(query)
    except Exception as err:
        print('ERROR BY CREATE TABLE', err)
 

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

def get_recent_pic():
    conn = sqlite3.connect('predictions.db')
    cur = conn.cursor()
    query = """
        SELECT  file_name, predict, feed_back, create_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 5
        """
    recent_pic = cur.execute(query)
    return recent_pic.fetchall()

create_categories_table()

@app.route('/',methods=['GET','POST'])
def index():
    recent_pic = get_recent_pic()
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
    return render_template('home.html', recent_pic=recent_pic)

@app.route('/result/<filename>')
def result(filename):
    prediction = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    Prediction(None,filename,float(prediction[0][0])).save_into_db()
    return render_template('result.html', prediction=prediction, filename=filename) 

@app.route('/howitworks')
def how_it_works():
    return render_template('how_it_works.html') 

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)
 