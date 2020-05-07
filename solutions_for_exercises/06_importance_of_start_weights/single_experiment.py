import sys
sys.path.append("../../cnn_toolbox")
print(sys.path)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# 1. get the random seed value
import sys
print("Here are all command line arguments: {0}".format( sys.argv ) )
rnd_seed_value = sys.argv[1]
print("I will now conduct experiment with random seed: {0}".format( rnd_seed_value ) )


# 2. check whether we have GPUs on this system available for CNN training
from cnn_toolbox import gpu_check
gpu_check()


# 3. prepare datasets for training and testing
img_shape = (224,224,3)


from cnn_toolbox import image_dataset

root_folder = "/media/juebrauer/Seagate Expansion Drive/datasets/01_images/18_imagenette2/320px/"
root_folder_train = root_folder + "train/"
root_folder_test  = root_folder + "val/"

ds_train = image_dataset(name="imagenette2-train",
                         root_folder=root_folder_train,
                         img_size=(img_shape[0],img_shape[1]))

ds_test = image_dataset(name="imagenette2-test",
                        root_folder=root_folder_test,
                        img_size=(img_shape[0],img_shape[1]))


# 4. create a CNN, train it, save training history
#    i.e. classification results on training and testing dataset
#    after each epoch

from cnn_toolbox import create_cnn_model,\
                        train_cnn_complete,\
                        prepare_output_folder,\
                        save_history

# 4.1 create the CNN
model = create_cnn_model(model_name = "same_nr_filters",
                         input_shape = img_shape,
                         nr_outputs = ds_train.nr_classes)

# 4.2 train the CNN completely
history = train_cnn_complete(model,
                             ds_train,
                             ds_test,
                             stop_epochnr=2)

# 4.3 save training history for further later analysis
output_folder = "saved_model_histories"
prepare_output_folder(output_folder)
fname = "{0}/model_seed{1:0>3}.history".format( output_folder, rnd_seed_value )
save_history(history, fname)


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")