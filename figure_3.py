import numpy as np
from pathlib import Path 

import config as cf
from ants import main

Path("imgs").mkdir(parents=True, exist_ok=True)

NUM_AVERAGE = 5
NAME = "compare_strict"
NUM_STEPS = 2000
INIT_ANTS = 200
MAX_ANTS = 200

C = np.array([[0, 0, 0, 0, 0, 1, 2, 3, 4, 5]])


def run():
    num_round_trips, paths, coeff = main(
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=True,
        name=NAME,
    )
    print(f"num_round_trips_{MAX_ANTS} {num_round_trips} / coeff {coeff}")
    f = open(f"imgs/{NAME}.txt", "a+")
    f.write(f"num_round_trips_{MAX_ANTS} {num_round_trips} / coeff {coeff}\n")
    f.close()


if __name__ == "__main__":

    for _ in range(NUM_AVERAGE):
        NAME = "compare_strict_10"
        INIT_ANTS = 10
        MAX_ANTS = 10
        run()

        NAME = "compare_strict_30"
        INIT_ANTS = 30
        MAX_ANTS = 30
        run()

        NAME = "compare_strict_50"
        INIT_ANTS = 50
        MAX_ANTS = 50
        run()

        NAME = "compare_strict_70"
        INIT_ANTS = 70
        MAX_ANTS = 70
        run()

