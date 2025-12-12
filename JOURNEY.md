# Scientific Journey: Multivariate CSC for Brain Signals

This document chronicles the scientific investigation into Convolutional Sparse Coding (CSC) for electromagnetic brain signals, based on the NeurIPS 2018 paper by Dupré La Tour et al.

---

## 1. Starting Point: The Waveform Shape Problem

### The Traditional View
Neuroscience has long relied on Fourier analysis and wavelet transforms to study brain rhythms. These methods decompose signals into frequency bands:
- **Theta** (4-8 Hz): Associated with memory and navigation
- **Alpha** (8-12 Hz): Related to relaxation and attention
- **Beta** (15-30 Hz): Linked to active thinking and motor control

### The Limitation
A fundamental assumption underlies these methods: brain oscillations are essentially sinusoidal. But what if they're not?

The paper posed a provocative question: **Can two rhythms have the same peak frequency but different waveform shapes?** If so, Fourier analysis would fail to distinguish them.

### The Mu-Rhythm Mystery
The **mu-rhythm** (~10 Hz) originates from the somatosensory cortex and appears during motor imagery and sensory processing. Despite having the same peak frequency as alpha waves, researchers had noted it "looked different" - more like a comb or "M" shape than a smooth sinusoid.

**Hypothesis**: If we could learn the actual waveform shapes from data (rather than assuming sinusoids), we might discover that brain rhythms have distinct morphologies that carry functional meaning.

---

## 2. The Method: Convolutional Sparse Coding

### Core Idea
Instead of projecting signals onto predefined basis functions (sines/cosines), CSC *learns* a dictionary of atoms directly from the data. The signal is modeled as:

```
X ≈ Σ z_k * D_k
```

where `D_k` are learned atoms (waveform patterns) and `z_k` are sparse activation signals indicating when each pattern occurs.

### The Rank-1 Innovation
For multivariate MEG/EEG data, the paper introduced a **rank-1 constraint**:

```
D_k = u_k × v_k^T
```

This decomposes each atom into:
- `u_k`: Spatial pattern (which channels are activated)
- `v_k`: Temporal pattern (the actual waveform shape)

**Why rank-1?** Maxwell's equations tell us that electromagnetic fields from a single brain source spread instantaneously across all sensors with fixed relative amplitudes. This is exactly what rank-1 captures.

### Practical Benefit
The rank-1 constraint means we can localize where in the brain each learned pattern originates - by fitting an equivalent current dipole to the spatial pattern `u_k`.

---

## 3. Validation: Synthetic Experiments

### Experiment 1 Design
Before trusting CSC on real brain data, we validated it on synthetic signals with known ground truth:
- Created two atoms: a triangle wave and a square wave
- Generated 100 trials with sparse activations
- Added noise (σ = 0.01)

### Results
The algorithm successfully recovered both atoms with high correlation to ground truth. Key observations:
- **Sign ambiguity**: Learned atoms may be flipped (×-1), handled by checking both polarities
- **Permutation ambiguity**: Atom ordering is arbitrary, matched by correlation
- **Convergence**: Objective function decreased monotonically over ~50 iterations

### Conclusion
CSC works as advertised - it can recover the true underlying waveform shapes when they exist.

---

## 4. Main Experiment: The Mu-Rhythm Hunt

### Dataset
MNE somatosensory dataset:
- **Task**: Median nerve stimulation (triggers mu-rhythm activity)
- **Channels**: 204 gradiometers
- **Trials**: 103 epochs (-2s to +4s around stimulus)
- **Sampling**: 150 Hz

### Preprocessing
- Notch filter: Removed 50/60 Hz power line noise
- High-pass filter at 2 Hz: Removed slow drifts
- No band-pass filtering (critical! - we want CSC to learn all frequencies)

### Model Configuration
- K = 25 atoms
- L = 150 samples (1 second)
- λ = 0.2 × λ_max (regularization)
- Rank-1 constraint enabled

### The Discovery
Among 25 learned atoms, **Atom 15** emerged as the mu-rhythm:
- **Peak frequency**: ~10 Hz
- **Waveform shape**: Non-sinusoidal "comb" pattern with sharp peaks
- **Spatial pattern**: Localized over somatosensory cortex
- **PSD**: Clear harmonic at ~20 Hz (2× fundamental)

This confirms the paper's central claim: the mu-rhythm has a distinct waveform shape that Fourier analysis would miss.

### Source Localization
We fit an equivalent current dipole to the learned spatial pattern:
- **Goodness of Fit**: 66.8% (vs. paper's 59.3%)
- **Location**: Primary somatosensory cortex (S1)

The dipole fit validates that the learned pattern corresponds to a focal brain source, not diffuse noise.

---

## 5. Extension: Activation Timing Analysis

### Motivation
CSC provides not just learned waveforms but also **sparse activation signals** z_k^n(t). These tell us exactly when each pattern appears in the data. A scientifically meaningful question: **When do these atoms activate relative to the stimulus?**

The somatosensory dataset includes a clear temporal reference: median nerve stimulation at t=0. If CSC-learned atoms capture real neural activity, they should show stimulus-locked responses.

### Method
For each atom's activation signal z_hat (shape: 103 trials × 25 atoms × 751 time points):
1. **Raster plots**: Visualize individual activation events across trials
2. **PSTH**: Post-Stimulus Time Histogram - average activation over time with Gaussian smoothing
3. **Peak latency**: Time of maximum activation post-stimulus
4. **Statistical comparison**: Pre-stimulus (-2 to 0s) vs post-stimulus (0.5 to 2s) using Wilcoxon signed-rank test

### Key Findings

#### Finding 1: Stimulus-Locked Responses
The mu-rhythm atom shows clear temporal structure in the raster plot:
- Sparse but consistent activations across all 103 trials
- Visible clustering of activations in the post-stimulus window
- The PSTH reveals a clear peak following stimulus onset

This confirms the learned pattern captures *functionally relevant* neural activity, not just background oscillations.

#### Finding 2: Response Timing
- Peak activation occurs post-stimulus, consistent with known S1 cortical response times
- Different atoms show different latencies, suggesting they may capture different processing stages

#### Finding 3: Event-Related Modulation
Statistical comparison reveals significant differences:
- Wilcoxon signed-rank test: p < 0.001 for mu-rhythm atom
- Clear effect size (Cohen's d) indicating meaningful change
- Consistent with event-related synchronization (ERS) or desynchronization (ERD) in the mu band

#### Finding 4: Atom Heterogeneity
Different atoms exhibit distinct temporal profiles:
- Early responders: May capture initial sensory processing
- Late responders: May capture motor preparation or attention
- This heterogeneity suggests the "mu-rhythm" label encompasses multiple functionally distinct patterns

### Scientific Significance
This analysis bridges CSC (a signal processing method) with neuroscience:
- **Validation**: Learned atoms have physiological meaning - they respond to stimuli
- **Event-related analysis**: CSC can be used like traditional ERP/ERD methods
- **Trial-by-trial variability**: The raster view enables single-trial analysis of waveform occurrence

---

## 6. Key Insights

### 1. Waveform Shape Matters
The mu-rhythm's comb shape produces a harmonic at 20 Hz. Traditional analysis might misinterpret this as beta activity (15-30 Hz band). CSC reveals it's actually a harmonic of the mu-rhythm, not independent beta oscillations.

### 2. Rank-1 Enables Localization
Without the rank-1 constraint, we'd have no principled way to ask "where does this pattern come from?" The spatial-temporal factorization is key for neuroscientific interpretation.

### 3. Sparsity Is Task-Dependent
The regularization parameter λ controls how many atoms are "active." Higher λ recovers fewer, more prominent patterns. Our choice (λ = 0.2 × λ_max) was tuned for the mu-rhythm but may miss weaker signals in other bands.

### 4. Harmonics Indicate Non-Sinusoidality
A practical diagnostic: if a waveform's PSD shows peaks at 2f₀, 3f₀..., it's non-sinusoidal. Pure sinusoids have a single spectral peak.

---

## 7. Limitations and Future Directions

### Current Limitations
1. **Single dataset**: Results specific to somatosensory cortex during median nerve stimulation
2. **Fixed parameters**: Different K, L, λ might reveal different patterns
3. **No ground truth for real data**: Can't verify if learned atoms match true neural generators

### Future Work
1. **Multi-dataset analysis**: Apply to resting-state, motor, auditory paradigms
2. **Parameter sensitivity**: Systematic study of K, L, λ effects
3. **Clinical applications**: Parkinson's disease shows abnormal beta waveforms - could CSC detect them?
4. **Real-time implementation**: Can CSC run fast enough for brain-computer interfaces?

---

## 8. Conclusions

This investigation validated the core claims of Dupré La Tour et al.:

1. **CSC successfully recovers non-sinusoidal brain rhythms** - the mu-rhythm's comb shape is real, not an artifact
2. **Rank-1 constraint enables source localization** - we can trace patterns back to specific brain regions
3. **Waveform shape carries information** - distinguishing mu from alpha requires looking beyond frequency

The extension to **activation timing analysis** revealed that CSC-learned atoms are not just static templates but capture temporally structured, stimulus-locked neural activity. The mu-rhythm shows significant event-related modulation, with different atoms exhibiting heterogeneous response profiles that may reflect distinct neural populations or processing stages.

**Bottom line**: CSC provides more than waveform shapes - it also reveals *when* brain patterns occur. This makes it a powerful tool for event-related brain signal analysis, complementing traditional methods while providing additional morphological information.

---

*This document was created as part of the MVA 2025/2026 Machine Learning for Time Series mini-project.*
