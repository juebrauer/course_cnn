import os

print("all_experiments_random_weights.py started.")


for dataset_name in ["imagenette", "imagewoof"]:

    for exp_nr in range(1,2):

        for use_rnd_weights in [True]:

            exp_name = "{0:0>4}".format(exp_nr)

            cmd = "python3 single_experiment_random_weights.py {0} {1} {2} > tmp_results/logfile_{3}.txt"\
                  .format(exp_name, use_rnd_weights, dataset_name, exp_name)

            os.system( cmd )


print("all_experiments_random_weights.py started.")
