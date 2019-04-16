from pyjssp.simulators import Simulator

if __name__ == "__main__":
    s = Simulator(5, 5)
    g, r, done = s.observe()

    for n in g.nodes:
        print(n, g.nodes[n])