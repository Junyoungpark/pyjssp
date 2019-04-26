import random
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pyjssp.jobShopSamplers import jssp_sampling
from pyjssp.operationHelpers import (JobManager,
                                     NodeProcessingTimeJobManager,
                                     get_edge_color_map,
                                     get_node_color_map)
from pyjssp.machineHelpers import (MachineManager,
                                   NodeProcessingTimeMachineManager)
from pyjssp.configs import (N_SEP, SEP, NEW)


class Simulator:
    def __init__(self,
                 num_machines,
                 num_jobs,
                 name=None,
                 machine_matrix=None,
                 processing_time_matrix=None,
                 embedding_dim=16,
                 use_surrogate_index=True,
                 delay=False,
                 verbose=False):

        if machine_matrix is None or processing_time_matrix is None:
            ms, prts = self._sample_jssp_graph(num_machines, num_jobs)
            self.machine_matrix = ms.astype(int)
            self.processing_time_matrix = prts.astype(float)
        else:
            self.machine_matrix = machine_matrix.astype(int)
            self.processing_time_matrix = processing_time_matrix.astype(float)

        if name is None:
            self.name = '{} machine {} job'.format(num_machines, num_jobs)
        else:
            self.name = name

        self._machine_set = list(set(self.machine_matrix.flatten().tolist()))
        self.num_machine = len(self._machine_set)
        self.embedding_dim = embedding_dim
        self.num_jobs = self.processing_time_matrix.shape[0]
        self.num_steps = self.processing_time_matrix.shape[1]
        self.use_surrogate_index = use_surrogate_index
        self.delay = delay
        self.verbose = verbose
        self.reset_simulator()
        # simulation procedure : global_time +=1 -> do_processing -> transit

    def reset_simulator(self):
        self.job_manager = JobManager(self.machine_matrix,
                                      self.processing_time_matrix,
                                      embedding_dim=self.embedding_dim,
                                      use_surrogate_index=self.use_surrogate_index)
        self.machine_manager = MachineManager(self.machine_matrix, self.delay, self.verbose)
        self.global_time = 0  # -1 matters a lot

    def transit(self, action=None):
        if action is None:
            # Perform random action
            machine = random.choice(self.machine_manager.get_available_machines())
            op_id = random.choice(machine.doable_ops_id)
            job_id, step_id = self.job_manager.sur_index_dict[op_id]
            operation = self.job_manager[job_id][step_id]
            action = operation
            machine.transit(self.global_time, action)
        else:
            if self.use_surrogate_index:
                if action in self.job_manager.sur_index_dict.keys():
                    job_id, step_id = self.job_manager.sur_index_dict[action]
                else:
                    raise RuntimeError("Input action is not valid")
            else:
                job_id, step_id = action

            operation = self.job_manager[job_id][step_id]
            machine_id = operation.machine_id
            machine = self.machine_manager[machine_id]
            action = operation
            machine.transit(self.global_time, action)

    def get_available_machines(self, shuffle_machine=True):
        return self.machine_manager.get_available_machines(shuffle_machine)

    def get_doable_ops_in_dict(self, machine_id=None, shuffle_machine=True):
        if machine_id is None:
            doable_dict = {}
            if self.get_available_machines():
                for m in self.get_available_machines(shuffle_machine):
                    _id = m.machine_id
                    _ops = m.doable_ops_id
                    doable_dict[_id] = _ops
            ret = doable_dict
        else:
            available_machines = [m.machine_id for m in self.get_available_machines()]
            if machine_id in available_machines:
                ret = self.machine_manager[machine_id].doable_ops_id
            else:
                raise RuntimeWarning("Access to the not available machine {}. Return is None".format(machine_id))
        return ret

    def get_doable_ops_in_list(self, machine_id=None, shuffle_machine=True):
        doable_dict = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        do_ops = []
        for _, v in doable_dict.items():
            do_ops += v
        return do_ops

    def get_doable_ops(self, machine_id=None, return_list=False, shuffle_machine=True):
        if return_list:
            ret = self.get_doable_ops_in_list(machine_id, shuffle_machine)
        else:
            ret = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        return ret

    def observe(self, reward='makespan', return_doable=True):
        # A simple wrapper for JobManager's observe function
        # and return current time step reward r
        # check all jobs are done or not, then return done = True or False

        jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
        # check jobs_done contains only True or False
        if np.prod(jobs_done) == 1:
            done = True
        else:
            done = False
        if reward == 'makespan':
            if done:
                r = -self.global_time
            else:
                r = 0
        # return reward as total sum of queues for all machines
        elif reward == 'utilization':
            t_cost = self.machine_manager.cal_total_cost()
            r = -t_cost

        g = self.job_manager.observe()

        if return_doable:
            if self.use_surrogate_index:
                do_ops_list = self.get_doable_ops(return_list=True)
                for n in g.nodes:
                    if n in do_ops_list:
                        job_id, op_id = self.job_manager.sur_index_dict[n]
                        m_id = self.job_manager[job_id][op_id].machine_id
                        g.nodes[n]['doable'] = True
                        g.nodes[n]['machine'] = m_id
                    else:
                        g.nodes[n]['doable'] = False
                        g.nodes[n]['machine'] = 0

        return g, r, done

    def plot_graph(self, draw=True,
                   node_type_color_dict=None,
                   edge_type_color_dict=None,
                   half_width=None,
                   half_height=None,
                   **kwargs):
        
        g = self.job_manager.observe()
        node_colors = get_node_color_map(g, node_type_color_dict)
        edge_colors = get_edge_color_map(g, edge_type_color_dict)
        
        if half_width is None:
            half_width = 30
        if half_height is None:
            half_height = 10
        
        num_horizontals = self.num_steps + 1
        num_verticals = self.num_jobs + 1 
        
        def xidx2coord(x):
            return np.linspace(-half_width, half_width, num_horizontals)[x]

        def yidx2coord(y):
            return np.linspace(-half_height, half_height, num_verticals)[y]
        
        pos_dict = OrderedDict()
        for n in g.nodes:
            if self.use_surrogate_index:
                y, x = self.job_manager.sur_index_dict[n]
                pos_dict[n] = np.array((xidx2coord(x), yidx2coord(y)))
            else:
                pos_dict[n] = np.array((xidx2coord(n[1]), yidx2coord(n[0])))
        
        if kwargs is None:
            kwargs['figsize'] = (10, 5)
            kwargs['dpi'] = 300
        
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1, 1, 1)

        nx.draw(g, pos_dict,
                node_color=node_colors,
                edge_color=edge_colors,
                with_labels=True,
                ax=ax)
        if draw:
            plt.show()
        else:
            return fig, ax

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        # Draw a gantt chart
        self.job_manager.draw_gantt_chart(path, benchmark_name, max_x)

    @staticmethod
    def _sample_jssp_graph(m, n):
        if not m % N_SEP == 0:
            m = int(N_SEP * (m // N_SEP))
            if m < N_SEP:
                m = N_SEP
        if not n % N_SEP == 0:
            n = int(N_SEP * (n // N_SEP))
            if n < N_SEP:
                n = N_SEP
        if m > n:
            raise RuntimeError(" m should be smaller or equal to n ")

        return jssp_sampling(m, n, 5, 100)
        # return jssp_sampling(m, n, 1, 5)

    @classmethod
    def from_path(cls, jssp_path, **kwargs):
        with open(jssp_path) as f:
            ms = []  # machines
            prts = []  # processing times
            for l in f:
                l_split = l.split(SEP)
                m = l_split[0::2]
                prt = l_split[1::2]
                if NEW in prt[-1]:
                    prt[-1] = prt[-1].split(NEW)[0]
                ms.append(np.array(m, dtype=int))
                prts.append(np.array(prt, dtype=float))

        ms = np.stack(ms)
        prts = np.stack(prts)
        num_job, num_machine = ms.shape
        name = jssp_path.split('/')[-1].replace('.txt', '')

        return cls(num_machines=num_machine,
                   num_jobs=num_job,
                   name=name,
                   machine_matrix=ms,
                   processing_time_matrix=prts,
                   **kwargs)

    @classmethod
    def from_TA_path(cls, pt_path, m_path, **kwargs):
        with open(pt_path) as f1:
            prts = []
            for l in f1:
                l_split = l.split(SEP)
                prt = [e for e in l_split if e != '']
                if NEW in prt[-1]:
                    prt[-1] = prt[-1].split(NEW)[0]
                prts.append(np.array(prt, dtype=float))

        with open(m_path) as f2:
            ms = []
            for l in f2:
                l_split = l.split(SEP)
                m = [e for e in l_split if e != '']
                if NEW in m[-1]:
                    m[-1] = m[-1].split(NEW)[0]
                ms.append(np.array(m, dtype=int))

        ms = np.stack(ms)-1
        prts = np.stack(prts)
        num_job, num_machine = ms.shape
        name = pt_path.split('/')[-1].replace('_PT.txt', '')

        return cls(num_machines=num_machine,
                   num_jobs=num_job,
                   name=name,
                   machine_matrix=ms,
                   processing_time_matrix=prts,
                   **kwargs)


class NodeProcessingTimeSimulator(Simulator):

    def reset_simulator(self):
        self.job_manager = NodeProcessingTimeJobManager(self.machine_matrix,
                                                        self.processing_time_matrix,
                                                        embedding_dim=self.embedding_dim,
                                                        use_surrogate_index=self.use_surrogate_index)
        self.machine_manager = NodeProcessingTimeMachineManager(self.machine_matrix,
                                                                self.delay,
                                                                self.verbose)
        self.global_time = 0  # -1 matters a lot