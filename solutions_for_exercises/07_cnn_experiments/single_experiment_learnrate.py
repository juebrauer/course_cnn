import sys
sys.path.append("../../cnn_toolbox")
print(sys.path)

##################################################################
# 1. get the experiment parameters from the command line arguments
##################################################################
import sys
print("Here are all command line arguments: {0}".format( sys.argv ) )
exp_name = sys.argv[1]
dataset_name = sys.argv[2]
learn_rate = float(sys.argv[3])


#gpu_nr = int(sys.argv[2])
#print("Model will be trained on GPU #{0}".format(gpu_nr))
#device_name = "/gpu:{0}".format(gpu_nr)


##################################################################
# 2. check whether we have GPUs on this system available for CNN training
##################################################################
from cnn_toolbox import gpu_check
gpu_check()



##################################################################
# 3. prepare datasets for training and testing
##################################################################
img_shape = (224,224,3)


from cnn_toolbox import image_dataset

if dataset_name=="imagenette":
    #root_folder = "/media/juebrauer/Seagate Expansion Drive/datasets/01_images/18_imagenette2/320px/"
    root_folder = "/home/juebrauer/data_jb/datasets/imagenette2/320px/"
elif dataset_name=="imagewoof":
    #root_folder = "/media/juebrauer/Seagate Expansion Drive/datasets/01_images/19_imagewoof/320px/"
    root_folder = "/home/juebrauer/data_jb/datasets/imagewoof/320px/"

root_folder_train = root_folder + "train/"
root_folder_test = root_folder + "val/"


ds_train = image_dataset(name=dataset_name+"-train",
                         root_folder=root_folder_train,
                         img_size=(img_shape[0],img_shape[1])
                         )

# for the test dataset: use all the available test images!
# for this nr_train_samples_per_class = -1
ds_test = image_dataset(name=dataset_name+"-test",
                        root_folder=root_folder_test,
                        img_size=(img_shape[0],img_shape[1])
                        )



##################################################################
# 4. create a CNN, train it, save training history
#    i.e. classification results on training and testing dataset
#    after each epoch
##################################################################

from cnn_toolbox import initialize_pseudo_random_number_generators,\
                        create_cnn_model,\
                        get_weights_from_conv_layer,\
                        train_cnn_complete,\
                        prepare_output_folder,\
                        save_history

# 4.1 set defined start value for random number generators?
#initialize_pseudo_random_number_generators( rnd_seed_value )

# 4.2 create the CNN
import tensorflow as tf
#with tf.device(device_name):
model = create_cnn_model(model_name = cnn_name,
                         input_shape = img_shape,                         
                         nr_outputs = ds_train.nr_classes
                         )
model.summary()

# 4.3 plausiblity check whether the weights are really different
#     for the case, a different random seed was used
filter_weights, bias_weights = get_weights_from_conv_layer(model, "conv2d", show_info=True)
print("Here are filter 0 weights:")
print(filter_weights[:,:,:,0])

# 4.4 train the CNN completely
#with tf.device(device_name):
history = train_cnn_complete(your_cnn=model,
                             your_train_ds=ds_train,
                             your_test_ds=ds_test,
                             stop_epochnr=50,
                             learn_rate = learn_rate)

# 4.5 save training history for further later analysis
if history["training_aborted_due_to_no_progress"] == False:
    output_folder = "saved_model_histories"
    prepare_output_folder(output_folder)
    fname = "{0}/{1}.history".format( output_folder, exp_name)
    save_history(history, fname)
