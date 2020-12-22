import numpy as np

import matplotlib.pyplot as plt


round_trips_10 = [3, 1, 2, 1, 2]
round_trips_100 = [44, 47, 38, 37, 39]
round_trips_1000 = [417, 396, 411, 408, 412]

norm_round_trips_10 = [el / 10 for el in round_trips_10]
norm_round_trips_100 = [el / 100 for el in round_trips_100]
norm_round_trips_1000 = [el / 1000 for el in round_trips_1000]

mean_10 = np.mean(np.array(norm_round_trips_10))
mean_100 = np.mean(np.array(norm_round_trips_100))
mean_1000 = np.mean(np.array(norm_round_trips_1000))

coeffs_10 = [25612, 18855, 27846]
coeffs_100 = [20443, 26061, 24249]
coeffs_1000 = [26083, 26121, 25432]

coeff_10 = np.mean(np.array(coeffs_10))
coeff_100 = np.mean(np.array(coeffs_100))
coeff_1000 = np.mean(np.array(coeffs_1000))

plt.bar(
    range(3),
    [mean_10, mean_100, mean_1000],
    color=["r", "g", "b"],
    alpha=0.7,
    tick_label=["10", "100", "1000"],
)
plt.ylabel("Normalised round trips")
plt.xlabel("Number of ants")
plt.savefig("imgs/round_trips.png")
plt.close("all")

plt.plot(range(3), [mean_10, mean_100, mean_1000], "ro-")
plt.ylabel("Normalised round trips")
plt.xlabel("Number of ants")
plt.xticks(range(3), ("10", "100", "1000"))
plt.savefig("imgs/round_trips_line.png")
plt.close("all")

plt.plot(range(3), [coeff_10, coeff_100, coeff_1000], "ro-")
plt.ylabel("Distance coefficient")
plt.xlabel("Number of ants")
plt.xticks(range(3), ("10", "100", "1000"))
plt.savefig("imgs/coeff_line.png")

