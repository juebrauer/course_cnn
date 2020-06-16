import numpy as np
import cv2
import os
from os import listdir
from os.path import isdir, isfile, join
import random

print("Welcome to image_dataset class v1.0 by Juergen Brauer")



# -------------------------------------------------
# Helper class for an image dataset
# -------------------------------------------------

class image_dataset:
    """
    Provides access to a complete image dataset, as imagenette2 or imagewoof
    """
        
    #
    # Traverse all subfolders
    # of the specified root foolder and
    # generate a Python list of the following form
    #
    # [ ["data/bikes/jfksdj43.jpg", "bikes",
    #   ["data/cars/bvcnm401.jpg", "cars"],
    #   ...
    # ]
    #
    def __init__(self,
                 name,
                 root_folder,
                 img_size,
                 inputs_are_for_VGG16=False,
                 dev_mode=False,
                 nr_samples_per_class=-1):
        """
        Generate a Python list <all_dataset_items>
        of available images
        """
        
        self.name = name
        
        self.img_size = img_size
        
        self.inputs_are_for_VGG16 = inputs_are_for_VGG16
        
        self.all_dataset_items = []
       
        self.class_names = [d for d in listdir(root_folder)
                            if isdir(os.path.join(root_folder,d))]
        
        self.nr_classes = len(self.class_names)
        
        self.mini_batch_size = 128

        self.nr_samples_per_class = nr_samples_per_class

        print("Under root folder\n\t{0}\n"
              "I have found the following {1} subfolders/classes:\n"
              .format(root_folder, self.nr_classes))
        print(self.class_names)        
        
        
        # For each subfolder ...
        total_nr_images_that_we_could_use = 0
        for class_id, class_name in enumerate(self.class_names):
            
            subfolder_name = root_folder + "/" + class_name + "/"
            
            filenames = [subfolder_name + f
                         for f in listdir(subfolder_name)
                         if isfile(join(subfolder_name, f))]
            
            print("{} files in subfolder {}"
                  .format(len(filenames), subfolder_name) )

            total_nr_images_that_we_could_use += len(filenames)
            
            # For each image filename in current subfolder ...
            example_imgs_for_current_class = 0
            for filename in filenames:
                
                teacher_vec = np.zeros( self.nr_classes )
                teacher_vec[class_id] = 1.0
                
                self.all_dataset_items.append(
                    [filename,
                     class_id,
                     class_name,
                     teacher_vec] )    
                
                example_imgs_for_current_class += 1
                
                # do only store
                # <nr_train_samples_per_class>
                # example images per class?
                if nr_samples_per_class != -1 and\
                   example_imgs_for_current_class >= nr_samples_per_class:
                    # we have collected enough training samples
                    break

            print("For class {0} I have {1} sample images in the list."
                  .format(class_name, example_imgs_for_current_class))
                
        
        self.nr_images = len(self.all_dataset_items)
        print("In total there are {0} images that we could use."
              .format(total_nr_images_that_we_could_use))
        print("We will use {0} of them.".format(self.nr_images))
        
        print("\nHere are the first 3 entries in the list:")
        for i,entry in enumerate(self.all_dataset_items[:3]):
            print("Entry #{0}:".format(i))
            for subentry in entry:
                print("\t{0}".format(subentry))
                
        self.original_order = self.all_dataset_items.copy()
        
        # currently we assume that we work with color images!
        self.nr_color_channels = 3
        
    
    
    def load_single_image(self, absolute_filename):
        """
        Load the specified image
        and preprocess it
        """
        
        img_orig = cv2.imread(absolute_filename)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)                
        img_orig = cv2.resize(img_orig,
                              self.img_size,
                              interpolation=cv2.INTER_AREA)
        
        if self.inputs_are_for_VGG16:                    
            x = img_orig.astype(float)
            x = np.expand_dims(x, axis=0)
            #print("x has shape", x.shape)                
            #print("x has mean", np.mean(x))        
            # From the VGG paper:
            # "The only pre-processing we do is subtracting the mean RGB value,
            # computed on the training set, from each pixel."
            #
            # see imagenet_utils.py
            #
            x = preprocess_input(x)
            #print("x has mean", np.mean(x))   
            img_preprocessed = x.reshape((self.img_size[0],self.img_size[1],3))
        else:            
            img_preprocessed = img_orig * (1.0 / 255.0)
        
        return img_orig, img_preprocessed
        
        
       
    def get_image_by_index(self, idx):
        """
        Return the image from the dataset
        with the specified index idx
        """
        
        image_filename  = self.all_dataset_items[idx][0]
        class_id        = self.all_dataset_items[idx][1]
        class_name      = self.all_dataset_items[idx][2]
        teacher_vec     = self.all_dataset_items[idx][3]
        
        img_orig, img_preprocessed = self.load_single_image(image_filename)
        
        return img_orig, img_preprocessed, class_id, class_name, teacher_vec
    
    
    def get_random_image(self):        
        """
        Return some random image
        """
        
        rnd_idx = np.random.randint(0, self.nr_images)
        return self.get_image_by_index( rnd_idx )
    
    
    def shuffle(self):
        """
        Shuffles the order of the entries
        in the all_training_items Python list
        """
        random.shuffle( self.all_dataset_items )

        
    def reset_original_order(self):
        """
        Sets the order of the images in the list
        to a defined initial state
        """
        self.all_dataset_items = self.original_order.copy()

    # ---------------------------------------

    def set_mini_batch_size(self, mini_batch_size):
        """
        Set the size of a single mini batch.
        
        What is a mini batch?
        For faster training a (C)NN, we divide
        the dataset into small mini batches and
        compute an update of the model after
        each mini batch presentation and error
        computation.
        """
        self.mini_batch_size = mini_batch_size
        
        
    def get_nr_mini_batches(self):
        """
        Given the current batch size 
        computs how many mini batches we have to process for the
        specified dataset in order to divide it into
        a complete set of mini batches
        """

        nr_mini_batches = (int) (self.nr_images / self.mini_batch_size)
        
        # If the number of images is not an integer multiple
        # of the batch size, we need an additional mini batch
        # more
        if self.nr_images % self.mini_batch_size != 0:            
            nr_mini_batches +=1
                
        return nr_mini_batches


    def get_image_mini_batch(self, mini_batch_idx):
        """
        Returns the idx-th specified image batch.
        """

        # 1. how many mini-batches are there at all?
        nr_mini_batches = self.get_nr_mini_batches()

        
        # 2. what is the size of the last batch?    
        last_mini_batch_size = self.mini_batch_size
        division_rest = self.nr_images % self.mini_batch_size
        if division_rest != 0:
            last_mini_batch_size = division_rest

            
        # 3. what is the final size of the current mini batch
        #    with index <batch_idx>?
        final_mini_batch_size = self.mini_batch_size   
        if mini_batch_idx == nr_mini_batches-1:
            # this is the last mini batch!
            final_mini_batch_size = last_mini_batch_size

            
        # 4. prepare input X and output Y matrices
        height = self.img_size[0]
        width  = self.img_size[1]        
        X = np.zeros( (final_mini_batch_size,height,width,self.nr_color_channels) )
        Y = np.zeros( (final_mini_batch_size,self.nr_classes) )

        
        # 5. prepare the batch
        for mini_batch_offset in range(0, final_mini_batch_size):

            # 5.1 compute index of image to retrieve
            img_idx = mini_batch_idx*self.mini_batch_size + mini_batch_offset

            # 5.2 get that image
            img_orig, img_processed, class_id, class_name, teacher_vec = \
                 self.get_image_by_index( img_idx )

            # 5.3
            # put the 3d image into the 4D array
            # since Keras wants as input for
            # the training method fit() a 4D array            
            X[mini_batch_offset,:,:,:] = img_processed

            # the desired output is a 2D array
            Y[mini_batch_offset,:] = teacher_vec

            
        # 6. return the input and output batch matrices
        return X,Y
    
    

    
    
    
    
# -------------------------------------------------
# Helper function for creating CNN of different
# sizes
# -------------------------------------------------


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow import keras    
    
def create_cnn_model(nr_outputs,
                     input_shape,
                     learn_rate = None,
                     model_name="inc_nr_filters"):
    """
    Here we create the desired CNN model using the Keras API and return it
    """

    K.clear_session()

    model = models.Sequential()

    if model_name == "inc-nr-filters":

        #1+2
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        #3+4
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #5+6
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #7+8
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        #9+10
        model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        #11+12
        model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))        
        
    elif model_name == "same-nr-filters":
        
        #1+2
        model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        #3+4
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #5+6
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        #7+8
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        #9+10
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        #11+12
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        

    # add MLP
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu', name="fc1"))
    model.add(layers.Dense(nr_outputs, activation='softmax', name="output"))


    # for all models: use the same optimizer, loss and learn rate:
    if learn_rate == None:
        # learn rate not specified by user,
        # use default learn rate
        my_optimizer = keras.optimizers.SGD()
    else:
        # learn rate specified by user,
        # use this desired learn rate
        my_optimizer = keras.optimizers.SGD(learning_rate=learn_rate)
    
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

    # return the model just built
    return model




# -------------------------------------------------
# Helper function for training a CNN (one epoch)
# -------------------------------------------------

from datetime import datetime

def train_cnn_one_epoch(your_cnn,
                        your_train_ds,
                        show_progress=True):
    """
    Given the specified model <your_cnn> and
    the specified dataset <your_train_ds>
    train the model for exactly one epoch,
    i.e. such that each training images is shown
    exactly one to the model
    """
    
    time_start_epoch = datetime.now()
        
    # 1. make sure, we use another compilation of
    #    mini batches in this training epoch
    your_train_ds.shuffle()
    
    
    # 2. compute how many mini batches we have to train
    nr_mini_batches = your_train_ds.get_nr_mini_batches()
    
    
    # 3. for all mini batches to train ...
    img_idx = 0
    nr_images_trained = 0
    for mini_batch_idx in range(0,nr_mini_batches):

        # 3.1 get the next mini batch of training images and labels
        X,Y = your_train_ds.get_image_mini_batch(mini_batch_idx)

        # 3.2 how train the model with that mini batch
        your_cnn.fit(X,Y,verbose=0)
        
        # 3.3 output how far we are
        nr_images_trained += X.shape[0]
        if show_progress and mini_batch_idx % 10 == 0:
            print("train_cnn_one_epoch: "
                  "finished training batch {0} of {1}. Trained images so far: {2}"
                   .format(mini_batch_idx+1,
                           nr_mini_batches,
                           nr_images_trained))

    time_end_epoch = datetime.now()
    time_delta_epoch = time_end_epoch - time_start_epoch
    print("train_cnn_one_epoch: "
          "time needed for training this epoch: {0}"
           .format(time_delta_epoch))
    



# -------------------------------------------------
# Helper function for testing a CNN
# -------------------------------------------------
            
import numpy as np

def test_cnn(your_cnn,
             your_test_ds,
             show_infos=True):
    """
    Given the specified model <your_cnn> and
    the specified test dataset <your_test_ds>
    test how good the model can predict the
    right class for each of the test images
    in the test dataset
    """
    
    if show_infos:
        print("test_cnn: testing model on dataset: {0}".format(your_test_ds.name))
    
    # 1. compute how many mini-batches to test
    nr_mini_batches = your_test_ds.get_nr_mini_batches()
    print("test_cnn: "
          "there are {0} testing images. So for a batch size of "
          "{1} we have to test {2} batches."
          .format(your_test_ds.nr_images,
                  your_test_ds.mini_batch_size,
                  nr_mini_batches))
       
    
    # 2. for all mini batches to test ...
    correct = 0
    nr_images_tested = 0
    for mini_batch_idx in range(0,nr_mini_batches):

        # 2.1
        # get the next batch of test images
        X,Y = your_test_ds.get_image_mini_batch(mini_batch_idx)
                
        # 2.2
        # classify the test images now!
        neuron_outputs = your_cnn.predict(X)
        
        # 2.3 
        # for debugging: log neuron outputs?
        # helps to check, whether the outputs are still valid
        # or e.g. nan values
        show_neuron_outputs_sometimes = False
        if show_neuron_outputs_sometimes and mini_batch_idx % 5 == 0:
            print("test_cnn: neuron_outputs.shape={}"
                  .format( neuron_outputs.shape ))
            print("test_cnn: neuron_outputs={}"
                  .format( neuron_outputs ))
                    
        # 2.4
        # compare predicted and actual class indices
        nr_imgs_tested = Y.shape[0]
        for test_img_idx in range(0, nr_imgs_tested):
            
            # get predicted class index
            predicted_class_idx = np.argmax(neuron_outputs[test_img_idx].reshape(-1))
            
            # get actual class index
            actual_class_idx = np.argmax(Y[test_img_idx].reshape(-1))
        
            # was the prediction correct?
            if predicted_class_idx == actual_class_idx:
                   correct +=1
            
        # 2.5
        # output how far we are with the test
        nr_images_tested += X.shape[0]
        if show_infos and mini_batch_idx % 10 == 0:
            print("test_cnn: tested mini batch {0} of {1}. Tested images so far: {2}"
                   .format(mini_batch_idx+1,
                           nr_mini_batches,
                           nr_images_tested))
            
        
    # 3. calculate classification rate
    correct_rate = float(correct) / float(your_test_ds.nr_images)
        
    print("test_cnn: "
          "correctly classified: {0} of {1} images of dataset '{2}':"
          " --> classification rate: {3:.2f}"
          .format(correct,
                  your_test_ds.nr_images,
                  your_test_ds.name,
                  correct_rate))
    
    
    # 4. return correct classification rate
    return correct_rate



# -------------------------------------------------
# Helper function for checking whether there are
# GPUs available or not on the current system
# -------------------------------------------------

import tensorflow as tf

def gpu_check():
    """
    Check whether we have GPUs available or not

    Note: for checking on a (Linux) computer with NVIDIA GPUs,
    whether they are really used during training, enter:

        watch -n1.0 nvidia-smi

    """
    # TF 2.1
    # list_gpus_available = tf.config.list_physical_devices('GPU')

    # TF 2.0
    list_gpus_available = tf.config.experimental.list_physical_devices('GPU')
    print("The following GPUs are available: {0}".format(list_gpus_available) )    
    print("Nr of GPUs available: {0}".format(len(list_gpus_available)) )

    
    
    
# -------------------------------------------------
# Helper function for controlling whether we use
# the same or different start weights
# -------------------------------------------------    
    

def initialize_pseudo_random_number_generators(seed_value):
    """
    For repeating experiments
    with the same or different start weights
    we control the pseudo-random number generators
    used by TensorFlow and Keras for
    initializing the weights
    """

    import random
    random.seed(seed_value)

    import numpy as np
    np.random.seed(seed_value)

    import tensorflow as tf
    tf.random.set_seed( seed_value )
    
 
 

# -------------------------------------------------
# Helper function for retrieving weight values
# from a CNN layer
# ------------------------------------------------- 
def get_weights_from_conv_layer(your_model, conv_layer_name, show_info=False):
    """
    Returns the weights and biases from the specified
    filter layer
    """
    
    for lay in your_model.layers:
        
        if lay.name == conv_layer_name:
        
            # get the list of weights from that layer
            #
            # note:
            # get_weights() returns a list with two elements:
            # list element #0: filter weights
            # list element #1: bias weights
            filter_weights = lay.get_weights()[0]
            bias_weights   = lay.get_weights()[1]
            
            # show some information about this entities?
            if show_info:
                print("filter_weights has shape: {0}".format(filter_weights.shape))
                print("bias_weights has shape: {0}".format(bias_weights.shape))

                print("filter_weights has type: {0}".format(type(filter_weights)))
                print("bias_weights has type: {0}".format(type(bias_weights)))
        
            return filter_weights, bias_weights
        
        
    return "Sorry, specified layer {0} not found!".format(conv_layer_name)




# -------------------------------------------------
# Helper function for training a CNN (many epochs)
# -------------------------------------------------

import random

def train_cnn_complete(your_cnn,
                       your_train_ds,
                       your_test_ds,
                       check_for_progress_min_cl_rate=False,
                       same_shuffling = False,
                       stop_epochnr = None,
                       stop_classification_rate_train = None,
                       show_progress=True,
                       stop_training_if_no_progress = False,
                       check_for_progress_epoch_nr = 5
                       ):
    """
    Given the specified model <your_cnn> and
    the specified dataset <your_train_ds>
    train the model till some stopping criterion is met
    """
    
    print("\n\n")
    print("-----------------------------------------------")
    print("train_cnn_complete: starting to train the model")
    print("-----------------------------------------------")
    
    time_start_training = datetime.now()
    print("train_cnn_complete: training start time is {0}".format(time_start_training))
    
    # 1. reset start order of training samples
    your_train_ds.reset_original_order()

    
    # 2. if the user wants the same order of
    #    mini-batches between training different models
    #    we reproduce the same order again and again
    #    by setting the random seed now to a defined
    #    value
    if same_shuffling == True:
        random.seed(0)
    
    
    # 3. count how many epochs we have already trained
    nr_epochs_trained = 0
    
    
    # 4. during training we will store classification
    #    rates for training and testing dataset
    history = { "cl_rate_train": [],
                "cl_rate_test" : [] }
    
    # 5. compute classification rate on
    #    training and testing data BEFORE training ... 
    if True:
	    cl_rate_train = test_cnn(your_cnn, your_train_ds)
	    cl_rate_test  = test_cnn(your_cnn, your_test_ds)        
	    
	    # 6. ... and store both rates
	    history["cl_rate_train"].append( cl_rate_train )
	    history["cl_rate_test"].append( cl_rate_test )
    
    
    # 7. train an epoch in each loop
    continue_training = True
    training_aborted_due_to_no_progress = False
    while continue_training:

        time_start_epoch = datetime.now()
        
        # 7.1
        # shuffle the training data for the next
        # training epoch
        your_train_ds.shuffle()
        
        # 7.2
        # train one epoch
        print("\n")
        print("********************************************************")
        print("train_cnn_complete: starting training epoch {0}"
              .format(nr_epochs_trained+1))        
        train_cnn_one_epoch(your_cnn,
                            your_train_ds,
                            show_progress=True)
        print("********************************************************")
        print("\n")

        # 7.3
        # compute classification rate on
        # training and testing data
        if True:
           cl_rate_train = test_cnn(your_cnn, your_train_ds)
           cl_rate_test  = test_cnn(your_cnn, your_test_ds)
        
           # 7.4
           # store both rates
           history["cl_rate_train"].append( cl_rate_train )
           history["cl_rate_test"].append( cl_rate_test )
        
        # 7.5
        # one epoch trained more
        nr_epochs_trained += 1
        
        # 7.6
        # show progress
        if show_progress:            
            print("train_cnn_complete: "
                  "training epoch {0} finished."
                  .format(nr_epochs_trained))
            print("train_cnn_complete: "
                  "classification rates: train={0:.2f}, test={1:.2f}"
                  .format(cl_rate_train, cl_rate_test))
        
        
        # 7.7
        # one of the stopping criteria met?
        
        # maximum nr of epochs reached?
        if stop_epochnr != None:
            if nr_epochs_trained == stop_epochnr:
                continue_training = False
                
        # classification threshold for training data reached?
        if stop_classification_rate_train != None:
            if cl_rate_train >= stop_classification_rate_train:
                training_aborted_due_to_no_progress = True
                continue_training = False

        time_end_epoch = datetime.now()
        time_delta_epoch = time_end_epoch - time_start_epoch
        print("train_cnn_complete: time needed for one epoch training "
              "and testing: {0}".format(time_delta_epoch))

        # make sure we invest not too much time in
        # training models, when there is no progress
        if stop_training_if_no_progress:
            # is it time to check whether there is progress?
            if nr_epochs_trained >= check_for_progress_epoch_nr:
                # yes, it is time to check whether there is enough
                # progress!

                # is there enough progress?
                if cl_rate_train < check_for_progress_min_cl_rate:
                    # training seems not to work!
                    # --> stop training now to be able to proceed
                    #     with a new model
                    print("train_cnn_complete: Aborting training since there seems"
                          "to be no progress. Even after {0} epochs, the"
                          "classification rate on the training data is still: {1}"
                          .format(nr_epochs_trained, cl_rate_train))
                    continue_training = False


    time_end_training = datetime.now()
    time_delta_training = time_end_training - time_start_training
    print("\n\n")
    print("-----------------------------------------------")
    print("train_cnn_complete: "
          "time needed for training the complete model: {0}"
           .format(time_delta_training))    
    print("-----------------------------------------------")

    print("train_cnn_complete: training end time is {0}".format(time_end_training))

    # 8. save information in history dictionary,
    #    whether we aborted the training or not
    history["training_aborted_due_to_no_progress"] = training_aborted_due_to_no_progress

    # 9. return data about the training history
    return history
    
    

# -------------------------------------------------
# Helper functions to save/load the training
# history of a model
# -------------------------------------------------

import pickle


def save_history(history, fname):
    """
    Using pickle, the training history information
    is saved into the specified file
    """
    fobj = open(fname, "wb")
    pickle.dump(history, fobj)
    fobj.close()
    
    
    
def load_history(fname):
    """
    Using pickle, the training history information
    is loaded from the specified file
    """
    fobj = open(fname, "rb")
    history = pickle.load(fobj)
    fobj.close()
    return history



# -------------------------------------------------
# Helper function to prepare an experiment output
# folder
# -------------------------------------------------

from pathlib import Path


def prepare_output_folder(folder_name):
    """
    Creates the specified folder, if it does not yet exist
    """
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    
    
def deactivate_conv_layers_for_training(model,
                                        show_layer_information=True):
    """
    If we conduct an experiment with a random feature hierarchy,
    we use this helper method to set only the FC layers to
    trainable and keep the rest of the model layers randomly and
    untrained.
    """

    if show_layer_information:
        print("Layers of the model BEFORE defining which to train:")
        for layer in model.layers:
            print("{}: trainable={}"
                   .format(layer.name, layer.trainable))

    
    for layer in model.layers:
        if "conv" in layer.name:
            layer.trainable = False
    
    ############################################
    # only the specified layers will be trained!
    ############################################
    #for layer_name in list_of_layer_names_to_train:    
    #    l = model.get_layer( layer_name )
    #    l.trainable = True

    if show_layer_information:
        print("Layers of the model AFTER defining which to train:")
        for layer in model.layers:
            print("{}: trainable={}"
                   .format(layer.name, layer.trainable))

    model.compile(optimizer=keras.optimizers.SGD(), loss='categorical_crossentropy')
    #model.compile(optimizer='adam', loss='categorical_crossentropy')
