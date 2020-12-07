import numpy as np

import config as cf
from ants import main, plot_path

NUM_AVERAGE = 5
NAME = "compare"
NUM_STEPS = 2000
INIT_ANTS = 200
MAX_ANTS = 200

# standard prior
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
prior = 0
for o in range(cf.NUM_OBSERVATIONS):
    C[o] = prior
    prior += PRIOR_TICK

if __name__ == "__main__":

    for _ in range(NUM_AVERAGE):
        INIT_ANTS = 10
        MAX_ANTS = 10

        num_round_trips, paths, coeff = main(
            num_steps=NUM_STEPS,
            init_ants=INIT_ANTS,
            max_ants=MAX_ANTS,
            C=C,
            save=False,
            switch=False,
            name=NAME,
        )
        print(f"num_round_trips_10 {num_round_trips} / coeff_10 {coeff/ MAX_ANTS}")
        f = open(f"imgs/{NAME}.txt", "w")
        f.write(f"num_round_trips_10 {num_round_trips} / coeff_10 {coeff/ MAX_ANTS}")
        f.close()

        INIT_ANTS = 100
        MAX_ANTS = 100

        num_round_trips, paths, coeff = main(
            num_steps=NUM_STEPS,
            init_ants=INIT_ANTS,
            max_ants=MAX_ANTS,
            C=C,
            save=False,
            switch=False,
            name=NAME,
        )
        print(f"num_round_trips_100 {num_round_trips} / coeff_100 {coeff/ MAX_ANTS}")
        f = open(f"imgs/{NAME}.txt", "w")
        f.write(f"num_round_trips_100 {num_round_trips} / coeff_100 {coeff/ MAX_ANTS}")
        f.close()

        INIT_ANTS = 1000
        MAX_ANTS = 1000

        num_round_trips, paths, coeff = main(
            num_steps=NUM_STEPS,
            init_ants=INIT_ANTS,
            max_ants=MAX_ANTS,
            C=C,
            save=False,
            switch=False,
            name=NAME,
        )
        print(f"num_round_trips_1000 {num_round_trips} / coeff_1000 {coeff/ MAX_ANTS}")
        f = open(f"imgs/{NAME}.txt", "w")
        f.write(f"num_round_trips_1000 {num_round_trips} / coeff_1000 {coeff/ MAX_ANTS}")
        f.close()