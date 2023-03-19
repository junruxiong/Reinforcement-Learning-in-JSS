

class Params:
    def __init__(self):
        self.num_m = 5
        self.num_j = 5
        self.num_t = 5
        self.time_matrix = np.random.randint(1, 10, size=(self.num_m, self.num_j))
