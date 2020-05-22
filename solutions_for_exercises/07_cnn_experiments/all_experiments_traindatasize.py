import os

print("all_experiments.py started.")

dataset_name = "imagenette"

exp_nr = 0
for run in [1,2,3]:
    for cnn_name in ["same-nr-filters", "inc-nr-filters"]:
        for nr_train_imgs_per_class in [100,200,300,400,500,600,700,800]:
                exp_nr +=1
                exp_name = "{0:0>3}".format(exp_nr)

                cmd = "python3 single_experiment_traindatasize.py {0} {1} {2} {3} > logfile_{4}.txt"\
                      .format(exp_name, cnn_name, dataset_name, nr_train_imgs_per_class, exp_name)

                os.system( cmd )


print("all_experiments.py finished.")