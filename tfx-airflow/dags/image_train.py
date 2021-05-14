from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
# import tensorflow_transform as tft # Step 4

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor

from tfx_bsl.tfxio import dataset_options

import image_module as ift
import tensorflow as tf
import tensorflow.keras as keras


from tfx.components.trainer.fn_args_utils import FnArgs
print(tf.__version__)

device = "gpu"

if device == "tpu":
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.config.experimental_connect_to_cluster(resolver)
  # This is the TPU initialization code that has to be at the beginning.
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)
else:
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

#Create model
params_default = {
    'lr' : 0.001,
    'beta_1' : 0.99,
    'beta_2' : 0.999,
    'epsilon' : 1e-08,
    'decay' : 0.01,
    'hidden_layers' : 1
}

# Define feature columns(Including feature engineered ones )
# These are the features which come from the TF Data pipeline
def create_feature_cols():
    #Keras format features
    image = tf.keras.Input(name='image', shape=(1,), dtype=tf.string)
    path  = tf.keras.Input(name='path', shape=(1,), dtype=tf.string)
    image_id  = tf.keras.Input(name='image_id', shape=(1,), dtype=tf.int64)
    id  = tf.keras.Input(name='id', shape=(1,), dtype=tf.int64)


    keras_dict_input = {'image': image, 'path': path,
                        'image_id': image_id,'id': id
                        }

    return({'K' : keras_dict_input})

def create_keras_model(feature_cols,  params = params_default):
    METRICS = [
            keras.metrics.RootMeanSquaredError(name='rmse')
    ]

    #Input layers
    input_feats = []
    for inp in feature_cols['K'].keys():
      input_feats.append(feature_cols['K'][inp])

    ##Input processing
    ##https://keras.io/examples/structured_data/structured_data_classification_from_scratch/
    ##https://github.com/tensorflow/community/blob/master/rfcs/20191212-keras-categorical-inputs.md

    
    image=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=20)(feature_cols['K']['image'])
    path=tf.keras.layers.experimental.preprocessing.Hashing(num_bins=20)(feature_cols['K']['path'])
    
    #Add feature engineered columns - LAMBDA layer
    ###Create MODEL
    ####Concatenate all features( Numerical input )
    x_input_numeric = tf.keras.layers.concatenate([
                    feature_cols['K']['image_id'],
                    feature_cols['K']['id']
                    # feature_cols['K']['width']
                    ])

    #DEEP - This Dense layer connects to input layer - Numeric Data
    x_numeric = tf.keras.layers.Dense(32, activation='relu', kernel_initializer="he_uniform")(x_input_numeric)
    x_numeric = tf.keras.layers.BatchNormalization()(x_numeric)

    ####Concatenate all Categorical features( Categorical converted )
    x_input_categ = tf.keras.layers.concatenate([image,path])
        
    
    #WIDE - This Dense layer connects to input layer - Categorical Data
    x_categ = tf.keras.layers.Dense(32, activation='relu', kernel_initializer="he_uniform")(x_input_categ)

    ####Concatenate both Wide and Deep layers
    x = tf.keras.layers.concatenate([x_categ, x_numeric])

    for l_ in range(params['hidden_layers']):
        x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer="he_uniform",
                                  activity_regularizer=tf.keras.regularizers.l2(0.00001))(x)
        x = tf.keras.layers.BatchNormalization()(x)

    #Final Layer
    out = tf.keras.layers.Dense(1, activation='relu')(x)
    model = tf.keras.Model(input_feats, out)

    #Set optimizer
    opt = tf.keras.optimizers.Adam(lr= params['lr'], beta_1=params['beta_1'], 
                                        beta_2=params['beta_2'], epsilon=params['epsilon'])

    #Compile model
    model.compile(loss='mean_squared_error',  optimizer=opt, metrics = METRICS)

    #Print Summary
    print(model.summary())
    return model

def keras_train_and_evaluate(model, train_dataset, validation_dataset, epochs=100):
  #Add callbacks
  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.00001, verbose = 1)
  
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

  #Train and Evaluate
  out = model.fit(train_dataset, 
                  validation_data = validation_dataset,
                  epochs=epochs,
                  # validation_steps = 3,   ###Keep this none for running evaluation on full EVAL data every epoch
                  steps_per_epoch = 100,   ###Has to be passed - Cant help it :) [ Number of batches per epoch ]
                  callbacks=[reduce_lr, #modelsave_callback, #tensorboard_callback, 
                             keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=True)]
                  )

  return model

def save_model(model, model_save_path):
  @tf.function
  def serving(image,path,image_id,id):
      ##Feature engineering( calculate distance )

      payload = {
          'image': image,
          'path': path,
          'image_id': image_id,
          'id' : id
      }
      
      ## Predict
      ##IF THERE IS AN ERROR IN NUMBER OF PARAMS PASSED HERE OR DATA TYPE THEN IT GIVES ERROR, "COULDN'T COMPUTE OUTPUT TENSOR"
      predictions = model(payload)
      return predictions

  serving = serving.get_concrete_function(image=tf.TensorSpec([None,], dtype= tf.string, name='image'), 
                                          path=tf.TensorSpec([None,], dtype= tf.string, name='path'),
                                          image_id=tf.TensorSpec([None,], dtype= tf.int64, name='image_id'),
                                          id=tf.TensorSpec([None,], dtype= tf.int64, name='id')

                                          )

  # version = "1"  #{'serving_default': call_output}
  tf.saved_model.save(
      model,
      model_save_path + "/",
      signatures=serving
  )

##Main function called by TFX
def run_fn(fn_args: FnArgs):
  #Create dataset input functions
  train_dataset = ift.make_input_fn(dir_uri = fn_args.train_files,
                      mode = tf.estimator.ModeKeys.TRAIN,
                      batch_size = 2)()

  validation_dataset = ift.make_input_fn(dir_uri = fn_args.eval_files,
                      mode = tf.estimator.ModeKeys.EVAL,
                      batch_size = 2)()

  #Create model
  m_ = create_keras_model(params = params_default, feature_cols = create_feature_cols())
  tf.keras.utils.plot_model(m_, show_shapes=True, rankdir="LR")

  #Train model
  m_ = keras_train_and_evaluate(m_, train_dataset, validation_dataset, fn_args.custom_config['epochs'])

  #Save model with custom signature
  save_model(m_, fn_args.serving_model_dir)

