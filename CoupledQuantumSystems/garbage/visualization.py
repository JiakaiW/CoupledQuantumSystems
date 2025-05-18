# from bidict import bidict
# import ipywidgets as widgets
# from jax import jit, vmap
# import jax.numpy as jnp
# from matplotlib.colors import LogNorm
# import matplotlib.gridspec as gridspec
# import numpy as np
# import matplotlib.pyplot as plt
# import qutip
# import scqubits
# from tqdm import tqdm

# from CoupledQuantumSystems.qobj_manip import *


# ############################################################################
# #
# #
# # Functions about visualizing data or results
# #
# #
# ############################################################################


# def compute_expectation(ket_or_dm, operator):
#     # Check if the input is a ket or a density matrix
#     if ket_or_dm.shape[-1] == 1:  # Input is a ket
#         return (jnp.linalg.multi_dot([jnp.conj(ket_or_dm).T, operator, ket_or_dm]))[0][0]
#     else:  # Input is a density matrix
#         return jnp.trace(jnp.dot(operator, ket_or_dm))
        
# def get_vectorized_compute_expectation_function():
#     # Vectorize the function over the kets
#     vmapped_function = vmap(compute_expectation, in_axes=(0, None))
#     return  jit(vmapped_function)

# def plot_population(results,qubit_level,osc_level,product_to_dressed,a,w_d,tlist,fourier=False,fix_ylim = True,plot_only_pn_alpha = False):
#     product_states = [(ql,ol) for ql in range(qubit_level) for ol in range(osc_level)]
#     idxs = [product_to_dressed[(s1, s2)] for (s1, s2) in product_states]
#     tot_dims = qubit_level*osc_level

#     nlevels = len(results)


#     a_op = jnp.array(a.full())
#     pn_op = jnp.array((a.dag()*a).full())

#     # Vectorize the function compute_expectation over the kets
#     vectorized_compute_expectation = vmap(compute_expectation, in_axes=(0, None))
#     vectorized_compute_expectation = jit(vectorized_compute_expectation)

#     for i in range(nlevels):
#         if hasattr(results[i], 'y'):
#             states = jnp.array(results[i].y)  # assuming y contains JAX arrays or density matrices
#         elif hasattr(results[i], 'states'):
#             states = jnp.stack([jnp.array(q.full()) for q in results[i].states])  # assuming states contains QObj or density matrices

#         results[i].expect = []
#         if not plot_only_pn_alpha:
#             for idx in idxs:
#                 dressed_state = jnp.zeros(tot_dims).at[idx].set(1).reshape(-1, 1)
#                 dressed_state_op = jnp.outer(dressed_state, jnp.conj(dressed_state).T)
#                 expectations = vectorized_compute_expectation(states, dressed_state_op)
#                 results[i].expect.append(np.array(expectations))
#         alpha_expect = vectorized_compute_expectation(states, a_op)
#         pns_expect = vectorized_compute_expectation(states, pn_op)
#         results[i].expect.append(np.array(alpha_expect))
#         results[i].expect.append(np.array(pns_expect))

#     if fourier == True:
#         first_dominant_freq =find_dominant_frequency(results[0].expect[-2],tlist)
#     else:
#         first_dominant_freq = w_d


#     fig, axes = plt.subplots(4,nlevels, figsize=(9, 6))

#     for i in range(nlevels):
#         if not plot_only_pn_alpha:
#             qubit_state_population = [np.zeros(shape=len(tlist))]*qubit_level
#             for idx, product_state in enumerate(product_states):
#                 ql = product_state[0]
#                 qubit_state_population[ql] += np.array(results[i].expect[idx],dtype=np.float64)
#             for ql in range(nlevels):
#                 axes[0][i].plot(tlist, qubit_state_population[ql], label=r"$\overline{|%s\rangle}$" % (f"{ql}"))
        

#         alpha = results[i].expect[-2]*np.exp(-1j * 2 * np.pi * np.array(first_dominant_freq) * np.array(tlist))

#         # Coherent state eigenval
#         real = alpha.real
#         imag = alpha.imag
#         axes[1][i].plot(tlist,imag , label=r"imag alpha")
#         axes[2][i].plot(tlist, real, label=r"real alpha")
#         axes[3][i].plot(-imag, real, label=r"imag alpha VS real alpha")
        
#         # Photon number
#         axes[0][i].plot(tlist, results[i].expect[-1], label=r"photon number")

#     if fix_ylim: 
#         axes[0][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.5, 0.5))
#         axes[1][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.3, 0.5))
#         axes[2][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.3, 0.5))
#         axes[3][nlevels-1].legend(loc='center', ncol=1, bbox_to_anchor=(1.4, 0.5))
#         plt.ylabel("population")
#         plt.xlabel("t (ns)")
#         for row in [0,1,2,3]:
#             max_x_range,min_x_range,max_y_range,min_y_range = 0,0,0,0
#             for col in range(nlevels):
#                 ymin, ymax = axes[row][col].get_ylim()
#                 xmin, xmax = axes[row][col].get_xlim()
#                 if ymax > max_y_range:
#                     max_y_range = ymax
#                 if ymin < min_y_range:
#                     min_y_range = ymin
#                 if xmax > max_x_range:
#                     max_x_range = xmax
#                 if xmin < min_x_range:
#                     min_x_range = xmin
#             for col in range(nlevels):
#                 axes[row][col].set_ylim(min_y_range, max_y_range)
#                 axes[row][col].set_xlim(min_x_range,max_x_range)
#                 # Set the third row y range equal x range
#                 if row == 3:
#                     axes[row][col].set_ylim(min(min_x_range,min_y_range), max(max_x_range,max_y_range))
#                     axes[row][col].set_xlim(min(min_x_range,min_y_range),max(max_x_range,max_y_range))
#     # plt.yscale('log')
#     for ax in axes.flat:
#         ax.minorticks_on()
#         ax.grid(True)
#         ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
#     for col in axes[3]:
#         col.set_aspect('equal', 'box')
#     plt.show()


# def plot_specturum(qubit, resonator, hilbertspace, num_levels = 20,
#                     flagged_transitions = [[[0,0],[0,1]],[[1,0],[1,1]],[[2,0],[2,1]]]):
#     product_to_dressed = generate_single_mapping(hilbertspace.hamiltonian())
#     energy_text_size = 8
#     # clear_output(wait=True)
    
#     fig, old_ax = qubit.plot_wavefunction(which = [0,1,2,3,4,5,6,7,8,9,10,11])
#     left, bottom, width, height = 1, 0, 1, 1  
#     ax = fig.add_axes([left, bottom, width, height])
#     fig.set_size_inches(8, 8)

#     product_to_dressed = bidict(product_to_dressed)
#     qls = [product[0] for product in [product_to_dressed.inv[l] for l in range(num_levels)]]
#     rls = [product[1] for product in [product_to_dressed.inv[l] for l in range(num_levels)]]
#     max_qubit_level = max(qls) +1 
#     max_resonator_level = max(rls) +1
#     qubit_ori_energies = qubit.eigenvals(max_qubit_level)
#     resonator_ori_energies = resonator.eigenvals(max_resonator_level)

    
#     max_rl_for_ql = [0] *3
#     for l in range(num_levels):
#         (ql,rl) = product_to_dressed.inv[l]
#         original = (qubit_ori_energies[ql] + resonator_ori_energies[rl])#* 2 * np.pi
#         x1,x2 = ql-0.25,ql+0.25
#         ax.plot([x1, x2], [original, original], linewidth=1, color='red')
#         ax.text(ql, original, f"{original:.3f}", fontsize=energy_text_size, ha='center', va='bottom')

#         dressed_state_index = product_to_dressed[(ql,rl)]
#         dressed = hilbertspace.energy_by_dressed_index(dressed_state_index)#* 2 * np.pi
#         ax.plot([x1, x2], [dressed, dressed], linewidth=1, color='green')
#         ax.text(ql, dressed, f"{dressed:.3f}", fontsize=energy_text_size, ha='center', va='top')

#         if ql in [0,1,2]:
#             if rl > max_rl_for_ql[ql]:
#                 max_rl_for_ql[ql]=rl

#     flagged_transitions = []
#     for ql in [0,1,2]:
#         for i in range(max_rl_for_ql[ql] ):
#             flagged_transitions.append([[ql,i],[ql,i+1]])
#     for transition in flagged_transitions:
#         state1, state2 = transition[0],transition[1]
#         dressed_index1 = product_to_dressed[(state1[0],state1[1])]
#         dressed_index2 = product_to_dressed[(state2[0],state2[1])]
#         if dressed_index1!= None and dressed_index2!= None:
#             energy1 = hilbertspace.energy_by_dressed_index(dressed_index1)#* 2 * np.pi
#             energy2 = hilbertspace.energy_by_dressed_index(dressed_index2)#* 2 * np.pi
#             ax.plot([state1[0], max_qubit_level], [energy2, energy2], linewidth=1, color='green')
#             ax.plot([state1[0], state2[0]], [energy1, energy2], linewidth=1, color='green')
#             ax.text((state1[0]+ state2[0])/2, (energy1+ energy2)/2, f"{energy2-energy1:.3f}", fontsize=energy_text_size, ha='center', va='top')
#         else:
#             print("dressed_state_index contain None")
#     plt.show()


# def plot_heatmap(result, time_index, product_to_dressed, qubit_levels, oscillator_levels,norm ):
#     if hasattr(result, 'states'):
#         dm = result.states[time_index]
#     elif hasattr(result, 'y'):
#         dm = result.y[time_index]
#     if hasattr(result, 'states_pad_back_custom'):
#         dm = result.states_pad_back_custom[time_index]
#     if dm.shape[1] == 1:
#         dm = qutip.ket2dm(dm)
#     dm = qutip.Qobj(dm.full())
#     dm = 0.5 * (dm + dm.dag())
#     dm = dm / dm.tr()
    
#     # dm = pad_back_function(dm)
#     grid = np.zeros(( qubit_levels,oscillator_levels))

#     for qubit_level in range(qubit_levels):
#         for oscillator_level in range(oscillator_levels):
#             product_state = (qubit_level, oscillator_level)
#             dressed_state = product_to_dressed[product_state]
#             if dressed_state < dm.dims[0][0]:
#                 # Create a basis state corresponding to the dressed state
#                 basis_state = qutip.basis(dm.dims[0][0], dressed_state)
#                 # Calculate the expectation value
#                 expectation_value = qutip.expect(basis_state * basis_state.dag(), dm)
#             else:
#                 expectation_value = 0
#             grid[ qubit_level,oscillator_level] = expectation_value
#     grid[grid < 1e-11] = 1e-11
#     plt.imshow(grid, cmap='viridis', origin='lower', norm=norm)
#     plt.colorbar(label='Expectation Value')
#     plt.xlabel('Oscillator Level')
#     plt.ylabel('Qubit Level')
#     plt.title(f'Expectation Values at t = {result.times[time_index]}')
#     plt.show()

# def interactive_heatmap(result, product_to_dressed, qubit_levels, oscillator_levels,norm = LogNorm()):
#     if hasattr(result, 'times'):
#         times = result.times
#     elif hasattr(result, 't'):
#         times = result.t
#     time_slider = widgets.IntSlider(
#         value=0,
#         min=0,
#         max=len(times) - 1,
#         step=1,
#         description='Time Index:',
#         continuous_update=False
#     )
    
#     widgets.interact(lambda time_index: plot_heatmap(result, time_index, product_to_dressed, qubit_levels, oscillator_levels,norm),
#                      time_index=time_slider)
    




# def get_shift_accurate(ele,omega_i, omega_j, omega_r):
#     return abs(ele)**2 / (omega_j-omega_i-omega_r) - abs(ele)**2 / (omega_i-omega_j-omega_r)



# def get_EJ_Er_sweep_data(EJ_values, 
#          Er_values,
#          EC,
#          EL,
#          computational_state = [0,1],
#          leakage_state = 2,
#     ):

#     qubit_level = 25
    
#     Z1 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])
#     Z2 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])

#     highest_level_to_transition_from = 16
#     transitions_to_0 = [[] for _ in range(highest_level_to_transition_from)]
#     transitions_to_1 = [[] for _ in range(highest_level_to_transition_from)]
#     transitions_to_2 = [[] for _ in range(highest_level_to_transition_from)]

#     # for every EJ
#     for i in tqdm(range(len(EJ_values)), desc="sweeping"):
#         qbt = scqubits.Fluxonium(EJ=EJ_values[i],EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
#         num_evals = qubit_level
#         evals = qbt.eigenvals(num_evals)
#         elements = qbt.matrixelement_table('n_operator',evals_count = num_evals)
        
#         # record the transitions to plot reference curves
#         for l in range(highest_level_to_transition_from):
#             transitions_to_0[l].append(abs(evals[l]-evals[0]))
#             transitions_to_1[l].append(abs(evals[l]-evals[1]))
#             transitions_to_2[l].append(abs(evals[l]-evals[2]))

#         # get estimated dispersive shifts
#         for j in range(len(Er_values)):
#             Er = Er_values[j]
#             shifts = [
#                 sum([get_shift_accurate(elements[0,ql2], evals[0], evals[ql2], Er) for ql2 in range(num_evals)] ),
#                 sum([get_shift_accurate(elements[1,ql2], evals[1], evals[ql2], Er) for ql2 in range(num_evals)]),
#                 sum([get_shift_accurate(elements[2,ql2], evals[2], evals[ql2], Er) for ql2 in range(num_evals)] )
#             ]

#             Z1[j, i] = abs(shifts[computational_state[1]]-shifts[computational_state[0]])
#             Z2[j, i] = abs(shifts[leakage_state]-shifts[computational_state[1]])
#     return (transitions_to_0,
#             transitions_to_1,
#             transitions_to_2,
#             Z1,
#             Z2)


# def get_EJ_Er_sweep_data_diagonalization(EJ_values, 
#          Er_values,
#          EC,
#          EL,
#          g,
#          computational_state = [0,1],
#          leakage_state = 2,
#     ):

#     qubit_level = 20
    
#     Z1 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])
#     Z2 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])

#     highest_level_to_transition_from = 16
#     transitions_to_0 = [[] for _ in range(highest_level_to_transition_from)]
#     transitions_to_1 = [[] for _ in range(highest_level_to_transition_from)]
#     transitions_to_2 = [[] for _ in range(highest_level_to_transition_from)]

#     # for every EJ
#     for i in tqdm(range(len(EJ_values)), desc="sweeping"):
#         qbt = scqubits.Fluxonium(EJ=EJ_values[i],EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
#         num_evals = qubit_level
#         evals = qbt.eigenvals(num_evals)
#         elements = qbt.matrixelement_table('n_operator',evals_count = num_evals)
        
#         # record the transitions to plot reference curves
#         for l in range(highest_level_to_transition_from):
#             transitions_to_0[l].append(abs(evals[l]-evals[0]))
#             transitions_to_1[l].append(abs(evals[l]-evals[1]))
#             transitions_to_2[l].append(abs(evals[l]-evals[2]))

#         # get estimated dispersive shifts
#         for j in range(len(Er_values)):
#             Er = Er_values[j]
#             # shifts = [
#             #     sum([get_shift_accurate(elements[0,ql2], evals[0], evals[ql2], Er) for ql2 in range(num_evals)] ),
#             #     sum([get_shift_accurate(elements[1,ql2], evals[1], evals[ql2], Er) for ql2 in range(num_evals)]),
#             #     sum([get_shift_accurate(elements[2,ql2], evals[2], evals[ql2], Er) for ql2 in range(num_evals)] )
#             # ]
#             try:
#                 osc = scqubits.Oscillator(
#                 E_osc=Er,
#                 truncated_dim=6
#                 )
#                 hilbertspace = scqubits.HilbertSpace([qbt, osc])
#                 hilbertspace.add_interaction(
#                     g_strength=g,
#                     op1=qbt.n_operator,
#                     op2=osc.creation_operator,
#                     add_hc=True
#                 )
#                 hilbertspace.generate_lookup()
#                 chi0 = transition_frequency(hilbertspace,hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))) - Er
#                 chi1 = transition_frequency(hilbertspace,hilbertspace.dressed_index((1,0)),hilbertspace.dressed_index((1,1))) - Er
#                 chi2 = transition_frequency(hilbertspace,hilbertspace.dressed_index((2,0)),hilbertspace.dressed_index((2,1))) - Er
#                 shifts = [
#                     chi0,
#                     chi1,
#                     chi2
#                 ]

#                 Z2[j, i] = abs(shifts[leakage_state]-shifts[computational_state[1]])
#                 Z1[j, i] = abs(shifts[computational_state[1]]-shifts[computational_state[0]])
#             except:
#                 Z2[j, i] = np.nan   
#                 Z1[j, i] = np.nan
#     return (transitions_to_0,
#             transitions_to_1,
#             transitions_to_2,
#             Z1,
#             Z2)



# def get_EJ_Er_sweep_data_diagonalization_three_outcome(EJ_values, 
#          Er_values,
#          EC,
#          EL,
#          g = 0.18,
#          computational_state = [0,1],
#          leakage_states = [0,3],
#     ):

#     qubit_level = 20
    
#     Z1 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])
#     Z2 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])
#     Z3 = np.zeros_like(np.meshgrid(EJ_values, Er_values)[0])


#     highest_level_to_transition_from = 16
#     transitions_to_0 = [[] for _ in range(highest_level_to_transition_from)]
#     transitions_to_1 = [[] for _ in range(highest_level_to_transition_from)]
#     transitions_to_2 = [[] for _ in range(highest_level_to_transition_from)]
#     transitions_to_3 = [[] for _ in range(highest_level_to_transition_from)]


#     # for every EJ
#     for i in tqdm(range(len(EJ_values)), desc="sweeping"):
#         qbt = scqubits.Fluxonium(EJ=EJ_values[i],EC=EC,EL=EL,flux=0,cutoff=110,truncated_dim=qubit_level)
#         num_evals = qubit_level
#         evals = qbt.eigenvals(num_evals)
        
#         # record the transitions to plot reference curves
#         for l in range(highest_level_to_transition_from):
#             transitions_to_0[l].append(abs(evals[l]-evals[0]))
#             transitions_to_1[l].append(abs(evals[l]-evals[1]))
#             transitions_to_2[l].append(abs(evals[l]-evals[2]))
#             transitions_to_3[l].append(abs(evals[l]-evals[3]))

#         # get estimated dispersive shifts
#         for j in range(len(Er_values)):
#             Er = Er_values[j]
#             try:
#                 osc = scqubits.Oscillator(
#                 E_osc=Er,
#                 truncated_dim=14
#                 )
#                 hilbertspace = scqubits.HilbertSpace([qbt, osc])
#                 hilbertspace.add_interaction(
#                     g_strength=g,
#                     op1=qbt.n_operator,
#                     op2=osc.creation_operator,
#                     add_hc=True
#                 )
#                 hilbertspace.generate_lookup()
#                 chi0 = transition_frequency(hilbertspace,hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))) - Er
#                 chi1 = transition_frequency(hilbertspace,hilbertspace.dressed_index((1,0)),hilbertspace.dressed_index((1,1))) - Er
#                 chi2 = transition_frequency(hilbertspace,hilbertspace.dressed_index((2,0)),hilbertspace.dressed_index((2,1))) - Er
#                 chi3 = transition_frequency(hilbertspace,hilbertspace.dressed_index((3,0)),hilbertspace.dressed_index((3,1))) - Er
#                 shifts = [
#                     chi0,
#                     chi1,
#                     chi2,
#                     chi3
#                 ]
#                 Z1[j, i] = abs(shifts[computational_state[1]]-shifts[computational_state[0]])
#                 Z2[j, i] = abs(shifts[leakage_states[0]]-shifts[computational_state[1]])
#                 Z3[j, i] = abs(shifts[leakage_states[1]]-shifts[computational_state[1]])
#             except:
#                 Z1[j, i] = np.nan
#                 Z2[j, i] = np.nan   
#                 Z3[j, i] = np.nan
#     return (transitions_to_0,
#             transitions_to_1,
#             transitions_to_2,
#             transitions_to_3,
#             Z1,
#             Z2,
#             Z3)




# def plot_EJ_Er_sweep(
#         EJ_values, 
#         Er_values,
#         transitions_to_0,
#         transitions_to_1,
#         transitions_to_2,
#         Z1,
#         Z2,
#         computational_state = [0,1],
#          leakage_state = 2,
#         legend = False,    
#         norm1= LogNorm(vmin=1e-5,vmax=1e-4),
#         norm2 = LogNorm(vmin=1e-2,vmax=1),
#         big_pic = False,
# ):
#     X, Y = np.meshgrid(EJ_values, Er_values)
#     # Plotting
#     if not big_pic:
#         fig = plt.figure(figsize=(2*(3+3/8), 
#                             (3+3/8)/1.8))
#     else:
#         fig = plt.figure(figsize=(2*(3+3/8)*3, 
#                             (3+3/8)/1.8*3))
#     gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1], wspace=0.4)

#     # ax1 = plt.subplot(gs[0])
#     # plt.text(-0.25, 1, '(a)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
#     # plt.plot(EJ_values, transition_21, '-', linewidth=2,color = 'black')
#     # plt.xlabel('EJ')
#     # plt.ylabel(r'$\omega_{12}$')
    
#     transitions = [transitions_to_0, transitions_to_1, transitions_to_2]
#     line_styles = [
#         (0,(1, 1)),
#         (0,(3, 3)),
#         (0,(3,2,1,2)),
#         (0, (1,1,1,1,3,3)),
#         (0, (1,1,1,1,1,1,3,3)),
#         (0, (1,1,2,2,1,1,4,5)),
#         (0, (5,5)),
#         (0, (5,1,5,1))
#     ]
#     line_style_counter = [[],[],[]]
#     next_line_style_idx = 0
#     def plot_transition_curves(initial_level):
#         nonlocal next_line_style_idx
#         if line_style_counter[initial_level] == []:
#             for level, list_of_transitions in enumerate(transitions[initial_level]):
#                 if level % 2 == 1 - (initial_level%2) and np.max(list_of_transitions) > Er_values[0] and np.min(list_of_transitions) < Er_values[-1]:
#                     plt.plot(EJ_values, list_of_transitions, linestyle = line_styles[next_line_style_idx], linewidth=2,color = 'black',label = rf'$\omega_{{{initial_level},{level}}}$')
#                     line_style_counter[initial_level].append(next_line_style_idx)
#                     next_line_style_idx += 1
#                     next_line_style_idx = next_line_style_idx % len(line_styles)
#         else:
#             local_counter = 0
#             for level, list_of_transitions in enumerate(transitions[initial_level]):
#                 if level % 2 == 1 - (initial_level%2) and np.max(list_of_transitions) > Er_values[0] and np.min(list_of_transitions) < Er_values[-1]:
#                     try:
#                         plt.plot(EJ_values, list_of_transitions, linestyle = line_styles[line_style_counter[initial_level][local_counter]], linewidth=2,color = 'black',label = rf'$\omega_{{{initial_level},{level}}}$')
#                     except:
#                         print(line_styles[line_style_counter[initial_level][local_counter]])
#                     local_counter += 1

#     ################################################################################
#     # Heatmap about the diff of shift from the two computational states
#     ################################################################################
#     ax0 = plt.subplot(gs[0])
#     plt.text(0.05, 1, '(a)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
#     z1_plot = plt.pcolormesh(X, Y, Z1, shading='auto', cmap='inferno', norm=norm1)
#     plt.colorbar(z1_plot)


#     plot_transition_curves(computational_state[0])
#     plot_transition_curves(computational_state[1])


#     if legend:
#         plt.legend(loc='lower left')
#     plt.xlabel('EJ')
#     plt.ylabel('Er')
#     ax0.set_xlim([EJ_values[0], EJ_values[-1]])
#     ax0.set_ylim([Er_values[0], Er_values[-1]])
#     ax0.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
#     ax0.minorticks_on()
#     ax0.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)



#     ################################################################################
#     # Heatmap about the diff of shift from the first computational state and the leakage state
#     ################################################################################
#     ax1 = plt.subplot(gs[1])
#     plt.text(0.05, 1, '(b)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
#     z2_plot = plt.pcolormesh(X, Y, Z2, shading='auto', cmap='inferno', norm=norm2)
#     plt.colorbar(z2_plot)

#     plot_transition_curves(computational_state[0])
#     plot_transition_curves(leakage_state)

#     if legend:
#         plt.legend(loc='lower left')
#     plt.xlabel('EJ')
#     # plt.ylabel('Er')
#     ax1.set_xlim([EJ_values[0], EJ_values[-1]])
#     ax1.set_ylim([Er_values[0], Er_values[-1]])
#     ax1.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
#     ax1.minorticks_on()
#     ax1.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
#     # ax3.set_xticks([]) 
#     # ax3.set_yticks([])


#     ################################################################################
#     # Additional subplot for overlay
#     ################################################################################
#     ax2 = plt.subplot(gs[2])
#     plt.text(0.05, 1, '(c)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', color='black')
#     # Overlay of Z1 and Z2
#     plt.pcolormesh(X, Y, Z1, shading='auto', cmap='inferno', alpha=0.5, norm=norm1)
#     plt.pcolormesh(X, Y, Z2, shading='auto', cmap='inferno', alpha=0.5, norm=norm2)

#     plot_transition_curves(computational_state[0])
#     plot_transition_curves(computational_state[1])
#     plot_transition_curves(leakage_state)

#     plt.xlabel('EJ')
#     # plt.ylabel('Er')
#     ax2.set_xlim([EJ_values[0], EJ_values[-1]])
#     ax2.set_ylim([Er_values[0], Er_values[-1]])
#     ax2.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
#     ax2.minorticks_on()
#     ax2.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)

#     plt.tight_layout()

#     return fig, (ax0, ax1, ax2)
