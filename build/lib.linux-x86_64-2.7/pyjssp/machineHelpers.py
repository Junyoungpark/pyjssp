import random
from collections import OrderedDict
import numpy as np
from pyjssp.operationHelpers import Operation, NodeProcessingTimeOperation
from pyjssp.configs import (PROCESSING_NODE_SIG,
                            DONE_NODE_SIG,
                            DELAYED_NODE_SIG)


class MachineManager:
    def __init__(self,
                 machine_matrix,
                 delay=True,
                 verbose=False):

        machine_matrix = machine_matrix.astype(int)

        # Parse machine indices
        machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(Operation.get_op(job_id, step_id))
            m_id += 1  # To make machine index starts from 1
            self.machines[m_id] = Machine(m_id, possible_ops, delay, verbose)

    def do_processing(self, t):
        for _, machine in self.machines.items():
            machine.do_processing(t)

    def load_op(self, machine_id, op, t):
        self.machines[machine_id].load_op(op, t)

    def __getitem__(self, index):
        return self.machines[index]

    def get_available_machines(self, shuffle_machine=True):
        m_list = []
        for _, m in self.machines.items():
            if m.available():
                m_list.append(m)

        if shuffle_machine:
            m_list = random.sample(m_list, len(m_list))

        return m_list

    # calculate the length of queues for all machines
    def cal_total_cost(self):
        c = 0
        for _, m in self.machines.items():
            c += len(m.doable_ops_no_delay)
        return c

    # update all cost functions of machines
    def update_cost_function(self, cost):
        for _, m in self.machines.items():
            m.cost += cost

    def get_machines(self):
        return [m for _, m in self.machines.items()]

    def all_delayed(self):
        return np.product([m.delayed_op is not None for _, m in self.machines.items()])

    def fab_stuck(self):
        # All machines are not available and All machines are delayed.
        all_machines_not_available_cond = not self.get_available_machines()
        all_machines_delayed_cond = self.all_delayed()
        return all_machines_not_available_cond and all_machines_delayed_cond


class NodeProcessingTimeMachineManager(MachineManager):

    def __init__(self,
                 machine_matrix,
                 delay=True,
                 verbose=False):

        machine_matrix = machine_matrix.astype(int)

        # Parse machine indices
        machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(NodeProcessingTimeOperation.get_op(job_id, step_id))
            m_id += 1  # To make machine index starts from 1
            self.machines[m_id] = Machine(m_id, possible_ops, delay, verbose)


class Machine:
    def __init__(self, machine_id, possible_ops, delay, verbose):
        self.machine_id = machine_id
        self.possible_ops = possible_ops
        self.remain_ops = possible_ops
        self.current_op = None
        self.delayed_op = None
        self.prev_op = None
        self.remaining_time = 0
        self.done_ops = []
        self.num_done_ops = 0
        self.cost = 0
        self.delay = delay
        self.verbose = verbose

    def __str__(self):
        return "Machine {}".format(self.machine_id)

    def available(self):
        future_work_exist_cond = self.doable_ops(delay=self.delay)
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        ret = future_work_exist_cond and currently_not_processing_cond and not_wait_for_delayed_cond
        return ret

    def wait_for_delayed(self):
        wait_for_delayed_cond = self.delayed_op is not None
        ret = wait_for_delayed_cond
        if wait_for_delayed_cond:
            delayed_op_ready_cond = self.delayed_op.prev_op.node_status == DONE_NODE_SIG
            ret = ret and not delayed_op_ready_cond
        return ret

    def doable_ops(self, delay=True):
        # doable_ops are subset of remain_ops.
        # some ops are doable when the prev_op is 'done' or 'processing' or 'start'
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE_SIG
                prev_process = op.prev_op.node_status == PROCESSING_NODE_SIG

                if delay:
                    cond = prev_done or prev_process
                else:
                    cond = prev_done

                if cond:
                    doable_ops.append(op)
                else:
                    pass

        return doable_ops

    @property
    def doable_ops_id(self):
        doable_ops_id = []
        doable_ops = self.doable_ops(delay=self.delay)
        for op in doable_ops:
            doable_ops_id.append(op.id)

        return doable_ops_id

    @property
    def doable_ops_no_delay(self):
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == 1  # DONE NODE SIG
                if prev_done:
                    doable_ops.append(op)
        return doable_ops

    def work_done(self):
        return not self.remain_ops

    def load_op(self, t, op):

        # Procedures for double-checkings
        # If machine waits for the delayed job is done:
        if self.wait_for_delayed():
            raise RuntimeError("Machine {} waits for the delayed job {} but load {}".format(self.machine_id,
                                                                                  print(self.delayed_op), print(op)))

        # ignore input when the machine is not available
        if not self.available():
            raise RuntimeError("Machine {} is not available".format(self.machine_id))

        # ignore when input op's previous op is not done yet:
        if not op.processible():
            raise RuntimeError("Operation {} is not processible yet".format(print(op)))

        if op not in self.possible_ops:
            raise RuntimeError("Machine {} can't perform ops {}{}".format(self.machine_id,
                                                                          op.job_id,
                                                                          op.step_id))

        # Essential condition for checking whether input is delayed
        # if delayed then, flush dealed_op attr
        if op == self.delayed_op:
            if self.verbose:
                print("[DELAYED OP LOADED] / MACHINE {} / {} / at {}".format(self.machine_id, op, t))
            self.delayed_op = None

        else:
            if self.verbose:
                print("[LOAD] / Machine {} / {} on at {}".format(self.machine_id, op, t))

        # Update operation's attributes
        op.node_status = PROCESSING_NODE_SIG
        op.remaining_time = op.processing_time
        op.start_time = t

        # Update machine's attributes
        self.current_op = op
        self.remaining_time = op.processing_time
        self.remain_ops.remove(self.current_op)
        # reset cost because this machine loaded a new operation.
        self.cost = 0

    def unload(self, t):
        if self.verbose:
            print("[UNLOAD] / Machine {} / Op {} / t = {}".format(self.machine_id, self.current_op, t))
        self.current_op.node_status = DONE_NODE_SIG
        self.current_op.end_time = t
        self.done_ops.append(self.current_op)
        self.num_done_ops += 1
        self.prev_op = self.current_op
        self.current_op = None
        self.remaining_time = 0

    def do_processing(self, t):
        if self.remaining_time > 0:  # When machine do some operation
            self.current_op.remaining_time -= 1
            self.remaining_time -= 1

            if self.current_op.remaining_time <= 0:
                if self.current_op.remaining_time < 0:
                    raise RuntimeWarning("Negative remaining time observed")
                if self.verbose:
                    print("[OP DONE] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.current_op, t))
                self.unload(t)

    def transit(self, t, a):
        if self.available():  # Machine is ready to process.
            if a.processible():  # selected action is ready to be loaded right now.
                self.load_op(t, a)
            else:  # When input operation turns out to be 'delayed'
                a.node_status = DELAYED_NODE_SIG
                self.delayed_op = a
                self.delayed_op.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.current_op = None  # MACHINE is now waiting for delayed ops
                if self.verbose:
                    print("[DELAYED OP CHOSEN] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.delayed_op, t))
        else:
            raise RuntimeError("Access to not available machine")
