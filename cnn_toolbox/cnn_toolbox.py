import numpy as np
import cv2
import os
from os import listdir
from os.path import isdir, isfile, join
import random

print("Welcome to image_dataset class v1.0 by Juergen Brauer")


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
                 inputs_are_for_VGG16=False):
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

        print("Under root folder\n\t{0}\n"
              "I have found the following {1} subfolders/classes:\n"
              .format(root_folder, self.nr_classes))
        print(self.class_names)        
        
        
        # For each subfolder ...
        for class_id, class_name in enumerate(self.class_names):
            
            subfolder_name = root_folder + "/" + class_name + "/"
            
            filenames = [subfolder_name + f
                         for f in listdir(subfolder_name)
                         if isfile(join(subfolder_name, f))]
            
            print("{} files in subfolder {}"
                  .format(len(filenames), subfolder_name) )
            
            # For each image filename in current subfolder ...
            for filename in filenames:
                
                teacher_vec = np.zeros( self.nr_classes )
                teacher_vec[class_id] = 1.0
                
                self.all_dataset_items.append(
                    [filename,
                     class_id,
                     class_name,
                     teacher_vec] )
        
        self.nr_images = len(self.all_dataset_items)
        print("In total there are {} images"
              .format(self.nr_images))
        
        print("\nHere are the first 3 entries in the list:")
        for i,entry in enumerate(self.all_dataset_items[:3]):
            print("Entry #{0}:".format(i))
            for subentry in entry:
                print("\t{0}".format(subentry))
                
        self.original_order = self.all_dataset_items.copy()
        
    
    
    def load_single_image(self, absolute_filename):
        """
        Load the specified image
        and preprocess it
        """
        
        img = cv2.imread(absolute_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                
        img = cv2.resize(img,
                         self.img_size,
                         interpolation=cv2.INTER_AREA)
        
        if self.inputs_are_for_VGG16:                    
            x = img.astype(float)
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
            img_preprocessed = img * (1.0 / 255.0)
        
        return img, img_preprocessed
        
        
       
    def get_image_by_index(self, idx):
        """
        Return the image from the dataset
        with the specified index idx
        """
        
        image_filename  = self.all_dataset_items[idx][0]
        class_id        = self.all_dataset_items[idx][1]
        class_name      = self.all_dataset_items[idx][2]
        teacher_vec     = self.all_dataset_items[idx][3]
        
        img, img_preprocessed = self.load_single_image(image_filename)
        
        return img, img_preprocessed, class_id, class_name, teacher_vec
    
    
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

        
    def set_defined_start_order(self):
        """
        Sets the order of the images in the list
        to a defined initial state
        """
        self.all_dataset_items = self.original_order.copy()
        random.seed( 0 )
        self.shuffle()




def get_nr_batches(the_ds):
    """
    Given the batch size specified by PARAM_BATCH_SIZE
    computes how many batches we have to process for the
    specified dataset
    """
    
    nr_batches = (int) (the_ds.nr_images / PARAM_BATCH_SIZE)
    if the_ds.nr_images % PARAM_BATCH_SIZE != 0:
        nr_batches +=1
    return nr_batches


def get_image_batch(the_ds, batch_idx):
    """
    Helper functions for image batching
    Given a dataset and a batch size `PARAM_BATCH_SIZE`, the image batcher
    function returns the i-th specified image batch.
    """

    nr_batches = get_nr_batches(the_ds)
    
    # 1. what is the size of the last batch?    
    last_batch_size = PARAM_BATCH_SIZE
    if the_ds.nr_images % PARAM_BATCH_SIZE != 0:
        last_batch_size = the_ds.nr_images % PARAM_BATCH_SIZE
    
    # 2. what is the final size of the current batch
    #    with index <batch_idx>?
    final_batch_size = PARAM_BATCH_SIZE    
    if batch_idx == nr_batches-1:
        # this is the last batch!
        final_batch_size = last_batch_size
        
    # 3. prepare input X and output Y matrices
    height      = PARAM_INPUT_SHAPE[0]
    width       = PARAM_INPUT_SHAPE[1]
    nr_channels = PARAM_INPUT_SHAPE[2]
    X = np.zeros( (final_batch_size,height,width,nr_channels) )
    Y = np.zeros( (final_batch_size,the_ds.nr_classes) )
    
    # 4. prepare the batch
    for batch_offset in range(0, final_batch_size):
        
        # 4.1 compute index of image to retrieve
        img_idx = batch_idx*PARAM_BATCH_SIZE + batch_offset

        # 4.2 get that image
        img, img_processed, class_id, class_name, teacher_vec = \
             the_ds.get_image_by_index( img_idx )

        # 4.3
        # put the 3d image into the 4D array
        # since Keras wants as input for
        # the training method fit() a 4D array            
        X[batch_offset,:,:,:] = img_processed

        # the desired output is a 2D array
        Y[batch_offset,:] = teacher_vec
        
    # 5. return the input and output batch matrices
    return X,Y