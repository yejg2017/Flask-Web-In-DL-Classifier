
# coding: utf-8

# ### evaluate the model using the pre-trained model

# In[10]:


import os, sys
import numpy as np
import tensorflow as tf

from model import ResNetModel
from tqdm import tqdm
sys.path.insert(0, './utils')
from preprocessor import BatchPreprocessor
import cv2


# In[3]:


checkpoint_dir="./checkpoint/"
batch_size=1
num_classes=5

model_path=tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
label_dict={0:"Dr_0",1:"Dr_1",2:"Dr_2",3:"Dr_3",3:"Dr_4"}


# In[6]:


# Placeholders
x = tf.placeholder(tf.float32, [batch_size,224, 224, 3])
y = tf.placeholder(tf.float32, [batch_size,num_classes])
is_training = tf.placeholder('bool', [])


# Model
model = ResNetModel(is_training, depth=101, num_classes=num_classes)
model.inference(x)
# Training accuracy of the model
#correct_pred = tf.equal(tf.argmax(model.prob, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediction=tf.argmax(model.prob,1)


# In[15]:


img_dir="/home/ye/Data/Image/ZocEye/origin/sick/"
img_list=[os.path.join(img_dir,img) for img in os.listdir(img_dir)]
np.random.shuffle(img_list)

x_sample=cv2.imread(img_list[1])
img_sample=cv2.resize(x_sample,(224,224))
img=np.reshape(img_sample,[1,224,224,3])


# In[25]:


saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,model_path)
    
    idx=sess.run(prediction,feed_dict={x:img,is_training:False})
    label=label_dict[idx[0]]
    print(label)


# In[22]:


idx[0]

