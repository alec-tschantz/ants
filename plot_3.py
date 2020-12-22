import numpy as np

import matplotlib.pyplot as plt


round_trips_10 = [4, 4, 4, 5, 5]
round_trips_30 = [11, 6, 8, 4, 10]
round_trips_50 = [15, 15, 24, 10, 27]
round_trips_70 = [23, 21, 9, 23, 19]


#norm_round_trips_10 = [el / 10 for el in round_trips_10]
#norm_round_trips_30 = [el / 30 for el in round_trips_30]
#norm_round_trips_50 = [el / 50 for el in round_trips_50]
#norm_round_trips_70 = [el / 70 for el in round_trips_70]

norm_round_trips_10 = round_trips_10
norm_round_trips_30 = round_trips_30
norm_round_trips_50 = round_trips_50
norm_round_trips_70 = round_trips_70

mean_10 = np.mean(np.array(norm_round_trips_10))
mean_30 = np.mean(np.array(norm_round_trips_30))
mean_50 = np.mean(np.array(norm_round_trips_50))
mean_70 = np.mean(np.array(norm_round_trips_70))

coeffs_10 = [259129, 252195, 252507, 243852, 243448]
coeffs_30 = [791697, 664590, 811597, 769337, 756244]
coeffs_50 = [1300997, 1262762, 1339007, 1343278, 1285012]
coeffs_70 = [1801414, 1596067, 1762367, 1872975, 1786465]

coeff_10 = np.mean(np.array(coeffs_10))
coeffs_30 = np.mean(np.array(coeffs_30))
coeffs_50 = np.mean(np.array(coeffs_50))
coeffs_70 = np.mean(np.array(coeffs_70))

plt.bar(
    range(4),
    [mean_10, mean_30, mean_50, mean_70],
    color=["r", "g", "b", "y"],
    alpha=0.7,
    tick_label=["10", "30, ", "50", "70"],
)
plt.ylabel("Number of round trips")
plt.xlabel("Number of ants")
plt.savefig("imgs/round_trips.png")
plt.close("all")

plt.plot(range(4), [mean_10, mean_30, mean_50, mean_70], "ro-")
plt.ylabel("Number of round trips")
plt.xlabel("Number of ants")
plt.xticks(range(4), ("10", "30", "50", "70"))
plt.savefig("imgs/round_trips_line.png")
plt.close("all")

plt.plot(range(4), [coeff_10, coeffs_30, coeffs_50, coeffs_70], "ro-")
plt.ylabel("Distance coefficient")
plt.xlabel("Number of ants")
plt.xticks(range(4), ("10", "30", "50", "70"))
plt.savefig("imgs/coeff_line.png")

