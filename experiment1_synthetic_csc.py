"""
Experiment 1: Univariate CSC on Synthetic Data
================================================

This script reproduces a simplified version of Figure 3 from:
"Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals"
(Dupré La Tour et al., NeurIPS 2018)

It demonstrates convolutional sparse coding (CSC) on simulated signals
with known waveform patterns, allowing us to verify that the algorithm
correctly recovers the ground truth atoms.

Parameters match the paper:
- N = 100 trials
- L = 64 (atom length)
- T = 512 (signal length)
- K = 2 atoms
"""

import numpy as np
import matplotlib.pyplot as plt
from alphacsc.simulate import simulate_data
from alphacsc import learn_d_z

# =============================================================================
# Parameters (matching paper Section 4, Figure 3)
# =============================================================================
n_trials = 100      # N: number of signals
n_times = 512       # T: signal length
n_times_atom = 64   # L: atom length
n_atoms = 2         # K: number of atoms to learn
n_iter = 50         # number of alternating minimization iterations
reg = 0.1           # regularization parameter (lambda)

random_state = 42   # for reproducibility

# =============================================================================
# Generate Synthetic Data
# =============================================================================
print("Generating synthetic data...")
print(f"  - {n_trials} trials")
print(f"  - Signal length: {n_times}")
print(f"  - Atom length: {n_times_atom}")
print(f"  - Number of atoms: {n_atoms}")

# simulate_data generates:
# - X: signals (n_trials, n_times)
# - ds_true: ground truth atoms (n_atoms, n_times_atom)
# - z_true: ground truth activations
X, ds_true, z_true = simulate_data(
    n_trials, n_times, n_times_atom, n_atoms,
    random_state=random_state
)

# Add small Gaussian noise
rng = np.random.RandomState(random_state)
noise_level = 0.01
X += noise_level * rng.randn(*X.shape)

print(f"  - Data shape: {X.shape}")
print(f"  - True atoms shape: {ds_true.shape}")

# =============================================================================
# Run Convolutional Sparse Coding
# =============================================================================
print("\nRunning Convolutional Sparse Coding...")
print(f"  - Regularization: {reg}")
print(f"  - Iterations: {n_iter}")

pobj, times, d_hat, z_hat, reg_used = learn_d_z(
    X,
    n_atoms,
    n_times_atom,
    reg=reg,
    n_iter=n_iter,
    solver_d_kwargs=dict(factr=100),
    random_state=random_state,
    n_jobs=1,
    verbose=1
)

print(f"\nLearned atoms shape: {d_hat.shape}")
print(f"Final objective value: {pobj[-1]:.6f}")

# =============================================================================
# Visualize Results
# =============================================================================
print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Ground truth atoms
ax = axes[0, 0]
t = np.arange(n_times_atom)
for k in range(n_atoms):
    ax.plot(t, ds_true[k], label=f'Atom {k+1}', linewidth=2)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('Ground Truth Atoms')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Learned atoms
ax = axes[0, 1]
for k in range(n_atoms):
    ax.plot(t, d_hat[k], label=f'Atom {k+1}', linewidth=2)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('Learned Atoms (CSC)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Comparison (overlay)
ax = axes[1, 0]
colors = plt.cm.tab10.colors
for k in range(n_atoms):
    ax.plot(t, ds_true[k], '--', color=colors[k],
            label=f'True {k+1}', linewidth=2, alpha=0.7)
    # Try both polarities to handle sign ambiguity
    corr_pos = np.corrcoef(ds_true[k], d_hat[k])[0, 1]
    corr_neg = np.corrcoef(ds_true[k], -d_hat[k])[0, 1]
    if corr_neg > corr_pos:
        ax.plot(t, -d_hat[k], '-', color=colors[k],
                label=f'Learned {k+1}', linewidth=2)
    else:
        ax.plot(t, d_hat[k], '-', color=colors[k],
                label=f'Learned {k+1}', linewidth=2)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('Comparison: True (dashed) vs Learned (solid)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Convergence
ax = axes[1, 1]
ax.semilogy(pobj, linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Function')
ax.set_title('CSC Convergence')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/axel/TimeS_project/experiment1_results.png', dpi=150)
print("Saved: experiment1_results.png")
plt.show()

# =============================================================================
# Quantitative Evaluation
# =============================================================================
print("\n" + "="*50)
print("Quantitative Evaluation")
print("="*50)

# Compute correlation between true and learned atoms
# Handle permutation and sign ambiguity
from itertools import permutations

best_corr = -1
best_perm = None
best_signs = None

for perm in permutations(range(n_atoms)):
    for signs in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        total_corr = 0
        for k in range(n_atoms):
            corr = np.corrcoef(ds_true[k], signs[k] * d_hat[perm[k]])[0, 1]
            total_corr += corr
        if total_corr > best_corr:
            best_corr = total_corr
            best_perm = perm
            best_signs = signs

print(f"\nBest atom matching (permutation): {best_perm}")
print(f"Best signs: {best_signs}")
print(f"\nCorrelation between true and learned atoms:")
for k in range(n_atoms):
    corr = np.corrcoef(ds_true[k], best_signs[k] * d_hat[best_perm[k]])[0, 1]
    print(f"  Atom {k+1}: {corr:.4f}")

print("\nExperiment 1 completed successfully!")
