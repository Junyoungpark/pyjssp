import numpy as np

from pyjssp.jobShopSamplers import jssp_sampling
from pyjssp.operationHelpers import JobManager
from pyjssp.machineHelpers import MachineManager
from pyjssp.configs import (N_SEP, SEP, NEW)


class Simulator:
    def __init__(self,
                 num_machines,
                 num_jobs,
                 name=None,
                 machine_matrix=None,
                 processing_time_matrix=None,
                 embedding_dim=16):

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
        self.reset_simulator()
        # simulation procedure : global_time +=1 -> do_processing -> transit

    def reset_simulator(self):
        self.job_manager = JobManager(self.machine_matrix,
                                      self.processing_time_matrix,
                                      embedding_dim=self.embedding_dim)
        self.machine_manager = MachineManager(self.machine_matrix)
        self.global_time = 0  # -1 matters a lot

    def transit(self, action):
        """
        :param action: (2 dim tuple); (job_id, step_id)
        """
        job_id, step_id = action
        operation = self.job_manager[job_id][step_id]
        machine_id = operation.machine_id
        machine = self.machine_manager[machine_id]
        action = operation
        machine.transit(self.global_time, action)

    def get_available_machines(self):
        return self.machine_manager.get_available_machines()

    def observe(self, reward='makespan'):
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

        return self.job_manager.observe(), r, done

    def plot_graph(self):
        # A simple wrapper for JobManager's plot_graph function
        self.job_manager.plot_graph()

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
    def from_path(cls, jssp_path):
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
                   processing_time_matrix=prts)

    @classmethod
    def from_TA_path(cls, pt_path, m_path):
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
                   processing_time_matrix=prts)

