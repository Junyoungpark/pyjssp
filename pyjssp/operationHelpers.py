import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from collections import OrderedDict

from plotly.offline import plot

from pyjssp.configs import (NOT_START_NODE_SIG,
                            PROCESSING_NODE_SIG,
                            DONE_NODE_SIG,
                            DELAYED_NODE_SIG,
                            DUMMY_NODE_SIG,
                            CONJUNCTIVE_TYPE,
                            DISJUNCTIVE_TYPE,
                            FORWARD,
                            BACKWARD)


def get_edge_color_map(g, edge_type_color_dict=None):
    if edge_type_color_dict is None:
        edge_type_color_dict = OrderedDict()
        edge_type_color_dict[CONJUNCTIVE_TYPE] = 'k'
        edge_type_color_dict[DISJUNCTIVE_TYPE] = '#F08080'

    colors = []
    for e in g.edges:
        edge_type = g.edges[e]['type']
        colors.append(edge_type_color_dict[edge_type])
    return colors


def calc_positions(g, half_width=None, half_height=None):
    if half_width is None:
        half_width = 30
    if half_height is None:
        half_height = 10

    min_idx = min(g.nodes)
    max_idx = max(g.nodes)

    num_horizontals = max_idx[1] - min_idx[1] + 1
    num_verticals = max_idx[0] - min_idx[1] + 1

    def xidx2coord(x):
        return np.linspace(-half_width, half_width, num_horizontals)[x]

    def yidx2coord(y):
        return np.linspace(-half_height, half_height, num_verticals)[y]

    pos_dict = OrderedDict()
    for n in g.nodes:
        pos_dict[n] = np.array((xidx2coord(n[1]), yidx2coord(n[0])))
    return pos_dict


def get_node_color_map(g, node_type_color_dict=None):
    if node_type_color_dict is None:
        node_type_color_dict = OrderedDict()
        node_type_color_dict[NOT_START_NODE_SIG] = '#F0E68C'
        node_type_color_dict[PROCESSING_NODE_SIG] = '#ADFF2F'
        node_type_color_dict[DELAYED_NODE_SIG] = '#829DC9'
        node_type_color_dict[DONE_NODE_SIG] = '#E9E9E9'
        node_type_color_dict[DUMMY_NODE_SIG] = '#FFFFFF'

    colors = []
    for n in g.nodes:
        node_type = g.nodes[n]['type']
        colors.append(node_type_color_dict[node_type])
    return colors


class JobManager:
    def __init__(self,
                 machine_matrix,
                 processing_time_matrix,
                 embedding_dim=16,
                 use_surrogate_index=True):

        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = Job(job_i, m, pr_t, embedding_dim)

        # Constructing disjunctive edges
        machine_index = list(set(machine_matrix.flatten().tolist()))
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            for job_id1, step_id1 in zip(job_ids, step_ids):
                op1 = self.jobs[job_id1][step_id1]
                ops = []
                for job_id2, step_id2 in zip(job_ids, step_ids):
                    if (job_id1 == job_id2) and (step_id1 == step_id2):
                        continue  # skip itself
                    else:
                        ops.append(self.jobs[job_id2][step_id2])
                op1.disjunctive_ops = ops
        
        self.use_surrogate_index = use_surrogate_index
        
        if self.use_surrogate_index:
            # Constructing surrogate indices:
            num_ops = 0
            self.sur_index_dict = dict()
            for job_id, job in self.jobs.items():
                for op in job.ops:
                    op.sur_id = num_ops
                    self.sur_index_dict[num_ops] = op._id
                    num_ops += 1

    def __call__(self, index):
        return self.jobs[index]

    def __getitem__(self, index):
        return self.jobs[index]

    def observe(self):
        """
        :return: Current time stamp job-shop graph
        """
        
        g = nx.OrderedDiGraph()
        for job_id, job in self.jobs.items():
            for op in job.ops:
                not_start_cond = not (op == job.ops[0])
                not_end_cond = not isinstance(op, EndOperation)

                g.add_node(op.id, **op.x)

                if not_end_cond:  # Construct forward flow conjunctive edges only
                    g.add_edge(op.id, op.next_op.id,
                               processing_time=op.processing_time,
                               type=CONJUNCTIVE_TYPE,
                               direction=FORWARD)

                    for disj_op in op.disjunctive_ops:
                        g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)

                if not_start_cond:
                    g.add_edge(op.id, op.prev_op.id,
                               processing_time=-1 * op.prev_op.processing_time,
                               type=CONJUNCTIVE_TYPE,
                               direction=BACKWARD)
        return g

    def plot_graph(self, draw=True,
                   node_type_color_dict=None,
                   edge_type_color_dict=None,
                   half_width=None,
                   half_height=None,
                   **kwargs):

        g = self.observe()
        node_colors = get_node_color_map(g, node_type_color_dict)
        edge_colors = get_edge_color_map(g, edge_type_color_dict)
        pos = calc_positions(g, half_width, half_height)

        if kwargs is None:
            kwargs['figsize'] = (10, 5)
            kwargs['dpi'] = 300

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1, 1, 1)

        nx.draw(g, pos,
                node_color=node_colors,
                edge_color=edge_colors,
                with_labels=True,
                ax=ax)
        if draw:
            plt.show()
        else:
            return fig, ax

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        gantt_info = []
        for _, job in self.jobs.items():
            for op in job.ops:
                if not isinstance(op, DummyOperation):
                    temp = OrderedDict()
                    temp['Task'] = "Machine" + str(op.machine_id)
                    temp['Start'] = op.start_time
                    temp['Finish'] = op.end_time
                    temp['Resource'] = "Job" + str(op.job_id)
                    gantt_info.append(temp)
        gantt_info = sorted(gantt_info, key=lambda k: k['Task'])
        color = OrderedDict()
        for g in gantt_info:
            _r = random.randrange(0, 255, 1)
            _g = random.randrange(0, 255, 1)
            _b = random.randrange(0, 255, 1)
            rgb = 'rgb({}, {}, {})'.format(_r, _g, _b)
            color[g['Resource']] = rgb
        fig = ff.create_gantt(gantt_info, colors=color, show_colorbar=True, group_tasks=True, index_col='Resource',
                              title=benchmark_name + ' gantt chart', showgrid_x=True, showgrid_y=True)
        fig['layout']['xaxis'].update({'type': None})
        fig['layout']['xaxis'].update({'range': [0, max_x]})
        fig['layout']['xaxis'].update({'title': 'time'})

        plot(fig, filename=path)


class NodeProcessingTimeJobManager(JobManager):

    def __init__(self,
                 machine_matrix,
                 processing_time_matrix,
                 embedding_dim=16,
                 use_surrogate_index=True):

        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = NodeProcessingTimeJob(job_i, m, pr_t, embedding_dim)

        # Constructing disjunctive edges
        machine_index = list(set(machine_matrix.flatten().tolist()))
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            for job_id1, step_id1 in zip(job_ids, step_ids):
                op1 = self.jobs[job_id1][step_id1]
                ops = []
                for job_id2, step_id2 in zip(job_ids, step_ids):
                    if (job_id1 == job_id2) and (step_id1 == step_id2):
                        continue  # skip itself
                    else:
                        ops.append(self.jobs[job_id2][step_id2])
                op1.disjunctive_ops = ops

        self.use_surrogate_index = use_surrogate_index

        if self.use_surrogate_index:
            # Constructing surrogate indices:
            num_ops = 0
            self.sur_index_dict = dict()
            for job_id, job in self.jobs.items():
                for op in job.ops:
                    op.sur_id = num_ops
                    self.sur_index_dict[num_ops] = op._id
                    num_ops += 1

    def observe(self):
        """
        :return: Current time stamp job-shop graph
        """

        g = nx.OrderedDiGraph()
        for job_id, job in self.jobs.items():
            for op in job.ops:
                not_start_cond = not (op == job.ops[0])
                not_end_cond = not isinstance(op, EndOperation)

                g.add_node(op.id, **op.x)

                if not_end_cond:  # Construct forward flow conjunctive edges only
                    g.add_edge(op.id, op.next_op.id,
                               distance=(op.next_op.complete_ratio-op.complete_ratio),
                               type=CONJUNCTIVE_TYPE,
                               direction=FORWARD)

                    for disj_op in op.disjunctive_ops:
                        g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)

                if not_start_cond:
                    g.add_edge(op.id, op.prev_op.id,
                               distance=-(op.complete_ratio - op.prev_op.complete_ratio),
                               type=CONJUNCTIVE_TYPE,
                               direction=BACKWARD)

        return g


class Job:
    def __init__(self, job_id, machine_order, processing_time_order, embedding_dim):
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            op = Operation(job_id=job_id, step_id=step_id, machine_id=m_id,
                           prev_op=None,
                           processing_time=pr_t,
                           complete_ratio=cum_pr_t/self.processing_time,
                           job=self)
            cum_pr_t += pr_t
            self.ops.append(op)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]

        # instantiate DUMMY END node
        _prev_op = self.ops[-1]
        self.ops.append(EndOperation(job_id=job_id,
                                     step_id=_prev_op.step_id + 1,
                                     embedding_dim=embedding_dim))
        self.ops[-1].prev_op = _prev_op
        self.num_sequence = len(self.ops) - 1

        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i+1]

    def __getitem__(self, index):
        return self.ops[index]

    # To check job is done or not using last operation's node status
    @property
    def job_done(self):
        if self.ops[-2].node_status == DONE_NODE_SIG:
            return True
        else:
            return False

    # To check the number of remaining operations
    @property
    def remaining_ops(self):
        c = 0
        for op in self.ops:
            if op.node_status == DONE_NODE_SIG and op.node_status != DUMMY_NODE_SIG:
                c += 1
        return c


class NodeProcessingTimeJob(Job):

    def __init__(self, job_id, machine_order, processing_time_order, embedding_dim):
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            op = NodeProcessingTimeOperation(job_id=job_id,
                                             step_id=step_id,
                                             machine_id=m_id,
                                             prev_op=None,
                                             processing_time=pr_t,
                                             complete_ratio=cum_pr_t / self.processing_time,
                                             job=self)
            cum_pr_t += pr_t
            self.ops.append(op)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]

        # instantiate DUMMY END node
        _prev_op = self.ops[-1]
        self.ops.append(NodeProcessingTimeEndOperation(job_id=job_id,
                                                       step_id=_prev_op.step_id + 1,
                                                       embedding_dim=embedding_dim))
        self.ops[-1].prev_op = _prev_op
        self.num_sequence = len(self.ops) - 1

        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i + 1]


class DummyOperation:
    def __init__(self,
                 job_id,
                 step_id,
                 embedding_dim):
        self.job_id = job_id
        self.step_id = step_id
        self._id = (job_id, step_id)
        self.machine_id = 'NA'
        self.processing_time = 0
        self.embedding_dim = embedding_dim
        self.built = False
        self.type = DUMMY_NODE_SIG
        self._x = {'type': self.type}
        self.node_status = DUMMY_NODE_SIG
        self.remaining_time = 0
    
    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id


class StartOperation(DummyOperation):

    def __init__(self, job_id, embedding_dim):
        super().__init__(job_id=job_id, step_id=-1, embedding_dim=embedding_dim)
        self.complete_ratio = 0.0
        self._next_op = None

    @property
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, op):
        self._next_op = op
        self.built = True

    @property
    def x(self):
        ret = self._x
        ret['complete_ratio'] = self.complete_ratio
        return ret


class EndOperation(DummyOperation):

    def __init__(self, job_id, step_id, embedding_dim):
        super().__init__(job_id=job_id, step_id=step_id, embedding_dim=embedding_dim)
        self.remaining_time = -1.0
        self.complete_ratio = 1.0
        self._prev_op = None

    @property
    def prev_op(self):
        return self._prev_op

    @prev_op.setter
    def prev_op(self, op):
        self._prev_op = op
        self.built = True

    @property
    def x(self):
        ret = self._x
        ret['complete_ratio'] = self.complete_ratio
        ret['remain_time'] = self.remaining_time
        return ret


class NodeProcessingTimeEndOperation(EndOperation):

    @property
    def x(self):
        ret = self._x
        ret['processing_time'] = self.processing_time
        ret['remain_time'] = self.remaining_time
        return ret


class Operation:

    def __init__(self,
                 job_id,
                 step_id,
                 machine_id,
                 complete_ratio,
                 prev_op,
                 processing_time,
                 job,
                 next_op=None,
                 disjunctive_ops=None):

        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = NOT_START_NODE_SIG
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.delayed_time = 0
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        self.start_time = None
        self.end_time = None

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    def __str__(self):
        return "job {} step {}".format(self.job_id, self.step_id)

    def processible(self):
        prev_none = self.prev_op is None
        if self.prev_op is not None:
            prev_done = self.prev_op.node_status is DONE_NODE_SIG
        else:
            prev_done = False
        return prev_done or prev_none
    
    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id

    @property
    def disjunctive_ops(self):
        return self._disjunctive_ops

    @disjunctive_ops.setter
    def disjunctive_ops(self, disj_ops):
        for ops in disj_ops:
            if not isinstance(ops, Operation):
                raise RuntimeError("Given {} is not Operation instance".format(ops))
        self._disjunctive_ops = disj_ops
        self.disjunctive_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, next_op):
        self._next_op = next_op
        self.next_op_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def x(self):  # return node attribute
        not_start_cond = (self.node_status == NOT_START_NODE_SIG)
        delayed_cond = (self.node_status == DELAYED_NODE_SIG)
        processing_cond = (self.node_status == PROCESSING_NODE_SIG)
        done_cond = (self.node_status == DONE_NODE_SIG)

        if not_start_cond:
            _x = OrderedDict()
            _x["complete_ratio"] = self.complete_ratio
            _x["type"] = self.node_status
            _x["remain_time"] = -1
        elif processing_cond or done_cond or delayed_cond:
            _x = OrderedDict()
            _x["complete_ratio"] = self.complete_ratio
            _x["type"] = self.node_status
            _x["remain_time"] = self.remaining_time
        else:
            raise RuntimeError("Not supporting node type")
        return _x


class NodeProcessingTimeOperation(Operation):

    def __init__(self,
                 job_id,
                 step_id,
                 machine_id,
                 complete_ratio,
                 prev_op,
                 processing_time,
                 job,
                 next_op=None,
                 disjunctive_ops=None):

        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = NOT_START_NODE_SIG
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        self.start_time = None
        self.end_time = None

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    @property
    def x(self):  # return node attribute
        not_start_cond = (self.node_status == NOT_START_NODE_SIG)
        delayed_cond = (self.node_status == DELAYED_NODE_SIG)
        processing_cond = (self.node_status == PROCESSING_NODE_SIG)
        done_cond = (self.node_status == DONE_NODE_SIG)

        if not_start_cond:
            _x = OrderedDict()
            _x["processing_time"] = self.processing_time
            _x["type"] = self.node_status
            _x["remain_time"] = -1
        elif processing_cond or done_cond or delayed_cond:
            _x = OrderedDict()
            _x["processing_time"] = self.processing_time
            _x["type"] = self.node_status
            _x["remain_time"] = self.remaining_time
        else:
            raise RuntimeError("Not supporting node type")
        return _x