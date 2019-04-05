import time
from pprint import pprint
import numpy as np


def pprint_graph(graph):
    print("Node information")
    for n in graph.nodes():
        print('{}:'.format(n))
        pprint(graph.nodes[n])

    print("\n Edge information")
    for e in graph.edges():
        print('{}:'.format(e))
        pprint(graph.edges[e])


def check_equal(l):
    cond = (len(set(l.ravel().tolist())) == 1)
    return cond


def instance_generate():
    num_machine = np.random.randint(5, 10)
    num_job = np.random.randint(num_machine, 10)

    return num_machine, num_job


def doable_ops_dr(machine):
    doable_ops = []
    for op in machine.remain_ops:
        prev_start = op.prev_op is None
        if prev_start:
            doable_ops.append(op)
        else:
            prev_done = op.prev_op.node_status == 1  # DONE NODE SIG
            if prev_done:
                doable_ops.append(op)
    return doable_ops


class Timer:
    def __init__(self, name=None):
        if name is None:
            self.name = 'Operation'
        else:
            self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print('{} : {} sec'.format(self.name, self.interval))

