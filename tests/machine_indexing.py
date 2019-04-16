from pyjssp.simulators import Simulator

if __name__ == "__main__":
    s = Simulator(5, 5, delay=False)
    print(s.get_doable_ops())
    s.plot_graph()