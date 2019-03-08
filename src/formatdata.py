import numpy as np
import pandas as pd

ratings = np.array([[8, 4, 0, 0, 4], [0, 0, 8, 10, 4], [8, 10, 0, 0, 6], [10, 10, 8, 10, 10], [0, 0, 0, 0, 0], [2, 0, 4, 0, 6], [8, 6, 4, 0, 0], [0, 0, 6, 4, 0], [0, 6, 0, 4, 10], [0, 4, 6, 8, 8]])
print(ratings)
did_rate = (ratings != 0) * 1
print(did_rate)
