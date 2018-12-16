# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 


from werkzeug import secure_filename
import os,sys
import numpy as np
import cv2
import datetime
import time
import tensorflow as tf
from model import ResNetModel


app=Flask(__name__,static_url_path="",static_folder="static")
Bootstrap(app)
db=SQLAlchemy(app)

# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/filestorage.db'


checkpoint_dir="./checkpoint/"
batch_size=1
num_classes=5

model_path=tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
print(model_path)
label_dict={0:"Dr_0",1:"Dr_1",2:"Dr_2",3:"Dr_3",3:"Dr_4"}


# Saving Data To Database Storage
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	#data = db.Column(db.LargeBinary)


@app.route('/')
def index():
	return render_template('index.html')


@app.route("/dataupload",methods=["GET","POST"])
def dataupload():
    if request.method=="POST" and "image_data" in request.files:
        file=request.files["image_data"]
        filename=secure_filename(file.filename)

        file.save(os.path.join("static",filename))
        fullfile=os.path.join("static",filename)

        # For time
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        # EDA function
        img=cv2.imread(os.path.join("static",filename))
        df_shape=img.shape

        img=cv2.resize(img,(224,224))
        img=np.reshape(img,[1,224,224,3])

        # Placeholders
        x = tf.placeholder(tf.float32, [batch_size,224, 224, 3])
        is_training = tf.placeholder('bool', [])

        
        
        # Model
        model = ResNetModel(is_training, depth=101, num_classes=num_classes)
        model.inference(x)
        prediction=tf.argmax(model.prob,1)

       # saver=tf.train.Saver()
        with tf.Session() as sess:
           
            saver=tf.train.Saver()

            saver.restore(sess,model_path)
            idx=sess.run(prediction,feed_dict={x:img,is_training:False})
            label=label_dict[idx[0]]
        
        
        # Saving results of Uploaded files to Sqlite DB
        #newfile=FileContents(name=file.filename,modeldata=label)
        #db.session.add(newfile)
        #db.session.commit()

    return render_template('details.html',filename=filename,date=date,
		df_shape=df_shape,
                prediction=label,
                image_file=filename
		)



if __name__=="__main__":
    app.run(host="10.100.110.101",port=9527,debug=True)
