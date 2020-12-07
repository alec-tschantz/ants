import numpy as np

import config as cf
from ants import main, plot_path


NUM_STEPS = 2000
INIT_ANTS = 200
MAX_ANTS = 200

if __name__ == "__main__":
    NAME = "strict"
    C = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    num_round_trips, paths, coeff = main(
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=False,
        name=NAME,
    )
    print(f"num_round_trips {num_round_trips} / coeff {coeff/ MAX_ANTS}")
    f = open(f"imgs/{NAME}.txt", "w")
    f.write(f"num_round_trips {num_round_trips} / coeff {coeff/ MAX_ANTS}")
    f.close()

    NAME = "flat"
    C = np.array([[0.1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1]])
    num_round_trips, paths, coeff = main(
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=False,
        name=NAME,
    )
    print(f"num_round_trips {num_round_trips} / coeff {coeff/ MAX_ANTS}")
    f = open(f"imgs/{NAME}.txt", "w")
    f.write(f"num_round_trips {num_round_trips} / coeff {coeff/ MAX_ANTS}")
    f.close()

