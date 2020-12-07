import numpy as np

import config as cf
from ants import main, plot_path

NAME = "switch"
NUM_STEPS = 4000
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
    num_round_trips, paths, coeff = main(
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=True,
        name=NAME,
    )
    print(f"num_round_trips {num_round_trips} / coeff {coeff/ MAX_ANTS}")
    f = open(f"imgs/{NAME}.txt", "w")
    f.write(f"num_round_trips {num_round_trips} / coeff {coeff/ MAX_ANTS}")
    f.close()

