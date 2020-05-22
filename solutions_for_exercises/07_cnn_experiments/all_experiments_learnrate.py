import os

print("all_experiments_learnrate.py started.")

dataset_name = "imagenette"

exp_nr = 0
for run in [1,2,3,4,5]:
    for learn_rate in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            exp_nr +=1
            exp_name = "{0:0>4}".format(exp_nr)

            cmd = "python3 single_experiment_learnrate.py {0} {1} {2} > logfile_{4}.txt"\
                  .format(exp_name, dataset_name, learn_rate, exp_name)

            os.system( cmd )


print("all_experiments_learnrate.py finished.")