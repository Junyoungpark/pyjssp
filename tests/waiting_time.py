from pyjssp.simulators import Simulator
import numpy as np
import random

np.random.seed(1)
random.seed(1)

s = Simulator(3, 3, verbose=True)

s.transit()

s.plot_graph()

s.flush_trivial_ops()


g, _, _ = s.observe()
