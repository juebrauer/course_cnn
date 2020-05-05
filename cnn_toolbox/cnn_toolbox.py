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
        
        self.mini_batch_size = 32

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

        
    def set_defined_start_order(self):
        """
        Sets the order of the images in the list
        to a defined initial state
        """
        self.all_dataset_items = self.original_order.copy()
        random.seed( 0 )
        self.shuffle()

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