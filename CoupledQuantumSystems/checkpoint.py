import pickle
import qutip
import dynamiqs as dq
from dynamiqs.time_qarray import CallableTimeQArray
import jax.numpy as jnp
import numpy as np
import os
import inspect
import sys
from CoupledQuantumSystems.drive import DriveTerm

class Checkpoint:
    '''
    Checkpoint data should be store as qutip data, instead of data in VRAM (dynamiqs.Result).
    '''
    next_t_idx: int
    qt_result: qutip.solver.Result
    
    def __init__(self):
        self.next_t_idx = 0
        self.qt_result = qutip.solver.Result()

    def convert_from_dq_result(self, dq_result: dq.result.Result):
        self.qt_result.solver = 'dynamiqs'
        self.qt_result.times = np.array(dq_result.tsave)
        self.qt_result.expect = np.array(dq_result.expects)
        self.qt_result.states = dq.to_qutip(dq_result.states)
        self.next_t_idx = len(dq_result.tsave)-1

    def concatenate_with_new_dq_result_segment(self, dq_result: dq.result.Result):
        # We assume the first recorded time step in this new segment is not the last recorded time step in the previous segment, (setting  t_save[0] = t0 + dt, where dq.Options(t0 = dq_result.tsave[0])
        self.qt_result.expect = np.concatenate([self.qt_result.expect, np.array(dq_result.expects)], axis=1)
        self.qt_result.states.extend(dq.to_qutip(dq_result.states))
        self.qt_result.times = np.concatenate([self.qt_result.times, np.array(dq_result.tsave)])

        self.next_t_idx = self.next_t_idx + len(dq_result.tsave)

class CheckpointingJob:
    '''
    This class is used to save and load the data that describe the system and partial evolution results.
    Essentially, it breaks down the evolution into segments, and save the partial results as checkpoints, then it automatically exists, and wait for HTCondor to reschedule the next segment.
    '''
    def __init__(self, name: str):
        # Initialize with name, automatically load the system and checkpoint. If successful, then it's ready to run a segment.
        self.name = name
        self.system_file_name = f'{name}_system.pkl'
        self.qutip_result_file_name = f'{name}_result.pkl'
        self.checkpoint_file_load = 'checkpoint.atomic'

        # Load the system and checkpoint
        if os.path.exists(self.system_file_name):
            with open(self.system_file_name, 'rb') as f:
                data = pickle.load(f)
            self.set_system(**data)
            self.system_loaded = True
        else:
            self.system_loaded = False # This is the first time to run the job, no checkpoint to load, the user should call self.set_system() in this first segment

        if os.path.exists(self.checkpoint_file_load):
            with open(self.checkpoint_file_load, 'rb') as f:
                self.checkpoint = pickle.load(f)
            self.first_segment = False
            # print(f'Checkpoint loaded, starting from t_idx = {self.checkpoint.next_t_idx}')
        else:
            self.checkpoint = Checkpoint()
            self.first_segment = True
            # print('No checkpoint loaded, starting from t_idx = 0')

    def set_system(self,
                static_hamiltonian: dq.QArray,
                drive_terms: list[DriveTerm],
                rho0: dq.QArray,
                tsave: jnp.ndarray,
                jump_ops: list[dq.QArray],
                exp_ops: list[dq.QArray],
                method: dq.method.Method = dq.method.Tsit5(max_steps=int(1e9)),
                len_t_segment_per_chunk: int = 5):
        '''
        Disassemble the hamiltonian into serializable components.
        '''
        assert len(jump_ops) == 0 or len(jump_ops) == 1, "len(jump_ops)>1 lead to batching, not supported here"
        self.static_hamiltonian = static_hamiltonian
        self.drive_terms = drive_terms
        self.rho0 = rho0
        self.tsave = tsave
        self.jump_ops = jump_ops
        self.exp_ops = exp_ops
        self.method = method
        self.len_t_segment_per_chunk = len_t_segment_per_chunk

    def run_segment(self):
        next_segment_start = self.checkpoint.next_t_idx + self.len_t_segment_per_chunk
        if next_segment_start > len(self.tsave):
            next_segment_start = len(self.tsave)

        segment_t_save = self.tsave[self.checkpoint.next_t_idx:next_segment_start+1]

        def _H(t:float)->CallableTimeQArray:
            _H = dq.asqarray(self.static_hamiltonian)
            for term in self.drive_terms:
                _H += dq.asqarray(term.driven_op)* term.jax_wrapper()(t, term.get_pulse_shape_args_with_id())
            return _H

        H =  dq.timecallable(_H, discontinuity_ts = None)
        # print(f"starting from t_idx = {segment_t_save[0].item()}, ending at t_idx = {segment_t_save[-1].item()}")
        segment_result = dq.mesolve(
            H = H,
            rho0 = self.rho0,
            jump_ops = self.jump_ops,
            tsave = segment_t_save[1:] if not self.first_segment else segment_t_save, # The first time step is already in the previous checkpoint
            exp_ops = self.exp_ops,
            method = self.method,
            options = dq.Options(
                progress_meter = False,
                t0 = segment_t_save[0].item() # Converts from device array to regular float
            )
        )
        if self.first_segment:
            self.checkpoint.convert_from_dq_result(segment_result)
        else:
            self.checkpoint.concatenate_with_new_dq_result_segment(segment_result)
        if next_segment_start < len(self.tsave):
            # Step 1: save the segment_result to a temporary checkpoint file
            temp_checkpoint = f'{self.name}_tdx{next_segment_start}.tmp'
            with open(temp_checkpoint, 'wb') as f:
                pickle.dump(self.checkpoint, f)
                # Ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())
            # print(f'Checkpoint saved temporarily, ended at t_idx = {next_segment_start}')

            # Step 2: store the system
            if not self.system_loaded:
                params = list(inspect.signature(self.set_system).parameters.keys())
                data = {param: getattr(self, param) for param in params}
                with open(self.system_file_name, 'wb') as f:
                    pickle.dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())

            # Step 3: atomically rename the checkpoint file
            os.replace(temp_checkpoint, self.checkpoint_file_load)

            # Step 4: exit the job
            sys.exit(85)
        else:
            with open(self.qutip_result_file_name, 'wb') as f:
                pickle.dump(self.checkpoint.qt_result,f)
            # print(f"result written to {self.qutip_result_file_name}")
            sys.exit(0)