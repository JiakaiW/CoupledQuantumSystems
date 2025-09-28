"""Checkpointing utilities for quantum system evolution.

This module provides tools for saving and loading quantum system evolution states,
allowing for resumable long-running simulations. It's meant for simulation on GPU and storage as qutip objects.
"""

import pickle
import qutip
import numpy as np
import os
import inspect
import sys
from ..dynamics import DriveTerm
from typing import List, Optional, TYPE_CHECKING, Any

try:
    import dynamiqs as dq
    from dynamiqs.time_qarray import CallableTimeQArray
    import jax.numpy as jnp
    if TYPE_CHECKING:
        from dynamiqs.result import Result as DynamiqsResult
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    dq = None
    CallableTimeQArray = None
    jnp = None
    if TYPE_CHECKING:
        DynamiqsResult = Any
        DynamiqsMethod = Any
        DynamiqsQArray = Any
    else:
        DynamiqsMethod = None
        DynamiqsQArray = None

class Checkpoint:
    """Stores the state of a quantum system evolution.

    This class maintains the state of a quantum system evolution, including
    the current time index, quantum states, and expectation values. It provides
    methods to convert between dynamiqs and QuTiP result formats and to
    concatenate new evolution segments.

    Attributes:
        next_t_idx (int): Index of the next time step to evolve.
        qt_result (qutip.solver.Result): QuTiP result object containing the
            evolution history.
    """

    def __init__(self):
        """Initialize an empty checkpoint."""
        if not JAX_AVAILABLE:
            raise ImportError("JAX and dynamiqs are required for checkpoint functionality. Please install them using 'pip install CoupledQuantumSystems[jax]'")
        self.next_t_idx = 0
        self.qt_result = qutip.solver.Result()

    def convert_from_dq_result(self, dq_result: 'DynamiqsResult'):
        """Convert a dynamiqs result to QuTiP format and store it.

        Args:
            dq_result (DynamiqsResult): Dynamiqs evolution result to convert.

        Note:
            This method sets the solver name to 'dynamiqs' and converts all
            states and expectation values to QuTiP format.
        """
        self.qt_result.solver = 'dynamiqs'
        self.qt_result.times = np.array(dq_result.tsave)
        self.qt_result.expect = np.array(dq_result.expects)
        self.qt_result.states = dq.to_qutip(dq_result.states)
        self.next_t_idx = len(dq_result.tsave)-1

    def concatenate_with_new_dq_result_segment(self, dq_result: 'DynamiqsResult'):
        """Concatenate a new evolution segment to the existing checkpoint. We assume the first recorded time step in this new segment is not the last recorded time step in the previous segment, 
        (setting  t_save[0] = t0 + dt, where dq.Options(t0 = dq_result.tsave[0])

        Args:
            dq_result ('DynamiqsResult'): New evolution segment to append.

        Note:
            This method assumes the first time step in the new segment is
            not the last time step in the previous segment. It concatenates
            the times, states, and expectation values arrays.
        """
        self.qt_result.expect = np.concatenate([self.qt_result.expect, np.array(dq_result.expects)], axis=1)
        self.qt_result.states.extend(dq.to_qutip(dq_result.states))
        self.qt_result.times = np.concatenate([self.qt_result.times, np.array(dq_result.tsave)])

        self.next_t_idx = self.next_t_idx + len(dq_result.tsave)

class CheckpointingJob:
    """Manages checkpointed quantum system evolution jobs.

    This class handles the saving and loading of quantum system evolution states,
    breaking down long evolutions into segments that can be resumed after
    interruption. It is designed to work with HTCondor job scheduling.

    Attributes:
        name (str): Name of the job.
        system_file_name (str): Name of the file storing system parameters.
        qutip_result_file_name (str): Name of the file for final results.
        checkpoint_file_load (str): Name of the checkpoint file.
        system_loaded (bool): Whether the system parameters have been loaded.
        first_segment (bool): Whether this is the first evolution segment.
        checkpoint (Checkpoint): Current checkpoint state.
    """

    def __init__(self, name: str):
        """Initialize a checkpointing job.

        Args:
            name (str): Name of the job, used for file naming.

        Note:
            This method attempts to load existing system parameters and
            checkpoint data if available. If no checkpoint exists, it
            initializes a new checkpoint.
        """
        
        # Initialize with name, automatically load the system and checkpoint. If successful, then it's ready to run a segment.
        self.name = name
        self.system_file_name = f'system.pkl'
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
            print(f'Checkpoint loaded, starting from t_idx = {self.checkpoint.next_t_idx}')
        else:
            self.checkpoint = Checkpoint()
            self.first_segment = True
            print('No checkpoint loaded, starting from t_idx = 0')

    def set_system(self,
                static_hamiltonian: 'DynamiqsQArray',
                drive_terms: list[DriveTerm],
                rho0: 'DynamiqsQArray',
                tsave: 'jnp.ndarray',
                jump_ops: list['DynamiqsQArray'],
                exp_ops: list['DynamiqsQArray'],
                method: Optional['DynamiqsMethod'] = None,
                len_t_segment_per_chunk: int = 5):
        """Set up the quantum system for evolution.

        Args:
            static_hamiltonian (dq.QArray): Static part of the Hamiltonian.
            drive_terms (list[DriveTerm]): List of drive terms for the Hamiltonian.
            rho0 (dq.QArray): Initial state of the system.
            tsave (jnp.ndarray): Time points at which to save the state.
            jump_ops (list[dq.QArray]): List of jump operators for the master equation.
            exp_ops (list[dq.QArray]): List of operators for expectation values.
            method (dq.method.Method, optional): Integration method to use.
                If None, defaults to Tsit5 with max_steps=1e9.
            len_t_segment_per_chunk (int, optional): Length of each evolution segment.
                Defaults to 5.

        Note:
            The jump_ops list must be empty or contain exactly one operator,
            as multiple jump operators would require batching which is not
            supported in this implementation.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX and dynamiqs are required for checkpoint functionality. Please install them.")

        if method is None:
            self.method = dq.method.Tsit5(max_steps=int(1e9))
        else:
            self.method = method
            
        assert len(jump_ops) == 0 or len(jump_ops) == 1, "len(jump_ops)>1 lead to batching, not supported here"
        self.static_hamiltonian = static_hamiltonian
        self.drive_terms = drive_terms
        self.rho0 = rho0
        self.tsave = tsave
        self.jump_ops = jump_ops
        self.exp_ops = exp_ops
        self.len_t_segment_per_chunk = len_t_segment_per_chunk

    def run_segment(self,progress_meter):
        """Run a single evolution segment and handle checkpointing.

        This method:
        1. Determines the time range for the next segment
        2. Constructs the time-dependent Hamiltonian
        3. Runs the evolution for the segment
        4. Updates the checkpoint
        5. Saves the checkpoint if more segments remain
        6. Saves the final result if this is the last segment

        Note:
            If more segments remain after this one, the method will exit with
            code 85 to signal HTCondor to reschedule the job. If this is the
            final segment, it will save the complete result and exit with code 0.
        """
        last_idx = len(self.tsave) - 1

        next_segment_start = self.checkpoint.next_t_idx + self.len_t_segment_per_chunk
        if next_segment_start > last_idx:
            next_segment_start = last_idx

        segment_t_save = self.tsave[self.checkpoint.next_t_idx:next_segment_start+1]

        def _H(t:float)->CallableTimeQArray:
            _H = dq.asqarray(self.static_hamiltonian)
            for term in self.drive_terms:
                _H += dq.asqarray(term.driven_op)* term.jax_wrapper()(t, term.get_pulse_shape_args_with_id())
            return _H

        H =  dq.timecallable(_H, discontinuity_ts = None)
        print(f"starting from t value = {segment_t_save[0].item()}, ending at t value = {segment_t_save[-1].item()}")
        segment_result = dq.mesolve(
            H = H,
            rho0 = self.rho0 if self.first_segment else dq.asqarray(self.checkpoint.qt_result.states[-1]),
            jump_ops = self.jump_ops,
            tsave = segment_t_save[1:] if not self.first_segment else segment_t_save, # The first time step is already in the previous checkpoint
            exp_ops = self.exp_ops,
            method = self.method,
            options = dq.Options(
                progress_meter = progress_meter,
                t0 = segment_t_save[0].item() # Converts from device array to regular float
            )
        )
        if self.first_segment:
            self.checkpoint.convert_from_dq_result(segment_result)
        else:
            self.checkpoint.concatenate_with_new_dq_result_segment(segment_result)
        if next_segment_start < last_idx:
            # Step 1: save the segment_result to a temporary checkpoint file
            temp_checkpoint = f'{self.name}_tdx{next_segment_start}.tmp'
            with open(temp_checkpoint, 'wb') as f:
                pickle.dump(self.checkpoint, f)
                # Ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())
            print(f'Checkpoint saved temporarily, ended at t_idx = {next_segment_start}')

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
            print(f"result written to {self.qutip_result_file_name}")
            sys.exit(0)