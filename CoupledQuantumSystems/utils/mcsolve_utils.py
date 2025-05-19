import qutip
import numpy as np
import pickle
import zipfile
import os
from typing import List,Union
from tqdm import tqdm
import gzip

from ..dynamics import DriveTerm
from .mcsolve_on_node import packed_mcsolve_problem

def pack_mcsolve_chunks(
                    y0: qutip.Qobj,
                    tlist: np.array,
                    static_hamiltonian: qutip.Qobj,
                    drive_terms: List[DriveTerm],
                    c_ops: Union[None,List[qutip.Qobj]] = None,
                    e_ops:Union[None,List[qutip.Qobj]] = None,

                    ntraj = 500,
                    existing_chunk_num: int = 0,
                    chunk_size = 4):

    seeds = list(np.random.randint(0, 2**32,
                        size=ntraj,
                        dtype=np.int64))

    chunk_id = existing_chunk_num

    for i in range(0, ntraj, chunk_size):
        chunk_seeds = seeds[i:i + chunk_size]
        problem = packed_mcsolve_problem(
            y0=y0,
            tlist=tlist,
            static_hamiltonian = static_hamiltonian,
            drive_terms = drive_terms,
            c_ops=c_ops,
            e_ops = e_ops,
            chunk_seeds=chunk_seeds,
        )

        with open(f"{chunk_id}.pkl", "wb") as f:
            pickle.dump(problem, f)
        chunk_id += 1
    existing_chunk_num = chunk_id
    return existing_chunk_num

def pack_pkl_files_to_zip(zip_filename="mcsolve_input.zip"):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir('.'):
            name, ext = os.path.splitext(filename)
            if ext == '.pkl' and name.isdigit():
                zipf.write(filename)
                os.remove(filename)

def merge_results(zip_files):
    # Used to merge mcsolve results on HTC condor
    averaged_dm_array = None
    tlist = None
    ntraj = 0
    expect = None

    for zip_file in tqdm(zip_files,desc='progress'):
        if not os.path.exists(zip_file):
            print(f"File {zip_file} does not exist. Skipping...")
            continue

        with gzip.GzipFile(zip_file, "rb") as f:
            result = pickle.load(f)

        if tlist is None:
            tlist = result.times
        ntraj += result.ntraj

        if expect is None:
            expect = np.array(result.expect) * result.ntraj
        else:
            expect += np.array(result.expect)  * result.ntraj

        # Convert states to density matrices and sum them up
        states_array = np.array([[state.full() for state in traj] for traj in result.states])
        # The following line averages over n trajectories of kets
        # n is traj index, t is time index, r is row index, c is column index, i and j are the row and column index of the conjugated ket
        summed_dm_array = np.einsum('ntrc,ntij->tri', states_array, states_array.conj()) 
        if averaged_dm_array is None:
            averaged_dm_array = summed_dm_array
        else:
            averaged_dm_array += summed_dm_array

    averaged_dm_array /= ntraj

    # Convert the final averaged density matrices back to Qobj
    averaged_dms = [qutip.Qobj(dm) for dm in averaged_dm_array]

    final_result = qutip.solver.Result()
    final_result.states = averaged_dms
    final_result.times = tlist
    final_result.ntraj = ntraj
    final_result.expect = expect/ntraj

    # TODO:I can't get the individual expectations when one job contain more than one trajectory. (it' pre-averaged)
    return final_result

