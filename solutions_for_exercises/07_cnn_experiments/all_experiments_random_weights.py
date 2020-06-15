import os

print("all_experiments_random_weights.py started.")

exp_nr = 40

exp_nr = 0
for dataset_name in ["imagenette", "imagewoof"]:

    for run in range(1,11):

        for use_rnd_weights in [False]:
            
            exp_nr +=1

            exp_nr += 1

            exp_name = "{0:0>4}".format(exp_nr)

            cmd = "python3 single_experiment_random_weights.py {0} {1} {2} > tmp_results/logfile_{3}.txt"\
                  .format(exp_name, use_rnd_weights, dataset_name, exp_name)

            os.system( cmd )


print("all_experiments_random_weights.py finished.")
