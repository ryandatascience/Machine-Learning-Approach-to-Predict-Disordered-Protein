#Date 11-29-2017
#Tensorflow CNN for disorder protein data
#Yu Liang
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
rng = np.random
tf.logging.set_verbosity(tf.logging.INFO)

def loaddata(filename):
    fi = open(filename)
    traindata = []
    trainlabel = []
    Dcount = 0
    Ocount = 0
    protein = []
    for line in iter(fi):
        lineelements = line.split()
        if int(lineelements[1]) == 1:
            if protein:
                
                if len(protein) <= 12000:
                    for x in range(12000-len(protein)):
                        protein.append(0)
                else:
                    protein = protein[:12000]
                traindata.append(protein)
                protein = []
                percentage = Dcount*1.0/(Ocount+Dcount)
                label = 1
                if(percentage<=0.25):
                    label = 1
                elif(percentage>0.25 and percentage<=0.5):
                    label = 2
                elif(percentage>0.5 and percentage<=0.8):
                    label = 3
                else:
                    label = 4
                trainlabel.append(label)
                
        iu = lineelements[3]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        iu = lineelements[4]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        iu = lineelements[5]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        iu = lineelements[6]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        iu = lineelements[7]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        iu = lineelements[8]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        iu = lineelements[9]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        iu = lineelements[10]
        if(iu == 'NA'):
            iu = 0.0
        else:
            iu = float(iu)
        protein.append(iu)
        if (lineelements[11] == 'D'):
            Dcount += 1
        else:
            Ocount += 1
    return(np.asarray(traindata,dtype = np.float16),np.asarray(trainlabel,dtype = np.float16))

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 120, 100, 1])
    
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[5, 5],
      padding="same",
        activation=tf.nn.relu)
        
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #conv2 = tf.layers.conv2d(
    #  inputs=pool1,
    #  filters=16,
    #  kernel_size=[5, 5],
    #  padding="same",
    #activation=tf.nn.relu)
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool1, [-1, 60 * 50 * 8])
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=4)
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    traindata,tlabel = loaddata(sys.argv[1])
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="./tmp/ProteinTensor1")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5000)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": traindata},
      y=tlabel,
      batch_size=10,
      num_epochs=None, shuffle=True)
    classifier.train(
      input_fn=train_input_fn,
      steps=20000,hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": traindata},
      y=tlabel,
      num_epochs=1,
      shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
