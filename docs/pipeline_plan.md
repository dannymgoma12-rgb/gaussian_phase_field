# LLM-informed Prior + Physics-based Refinement Pipeline

> **Paper Title (working)**: *Text-Guided Brittle Fracture Parameter Estimation via 3D Gaussian Splatting*
>
> **Target**: SIGGRAPH / SCA

---

## 1. Overview

텍스트로 재질을 알려주면 LLM이 초기 물성값(E, Gc, v)을 추정하고,
실제 파괴 영상과 비교하여 MPM + 3DGS 시뮬레이션으로 finite difference 기반 정밀 보정한다.

```
                    "ceramic mug"
                         |
                    [ LLM Prior ]
                         |
                   E0, Gc0, v0  (initial guess)
                         |
          +--------------+--------------+
          |    Inverse Optimization Loop |
          |                              |
          |  for i in range(max_iter):   |
          |    1. Forward sim (E,Gc,v)   |
          |       -> rendered frames     |
          |    2. Loss(rendered, target)  |
          |       SSIM + L1              |
          |    3. Finite diff gradient   |
          |       6 forward sims/iter    |
          |    4. Gradient descent       |
          |       (log-space for E,Gc)   |
          |    5. Converged? -> stop     |
          +--------------+--------------+
                         |
                  E*, Gc*, v*  (estimated)
```

---

## 2. Forward Pipeline (existing, working)

```
Config (E, Gc, v)
  -> Mesh -> Particles (MPM space [0,1]^3)
  -> CorotatedPhaseFieldElasticity (stress from F)
  -> MPM P2G2P loop (substeps per frame)
  -> AT2 Phase Field (damage evolution, uses Gc)
  -> Damage Mapper (volume -> surface)
  -> Gaussian Updater (position + crack visualization)
  -> 3DGS Rasterizer (render to image)
  -> Output frames
```

### Key Material Parameter Flow
| Parameter | Where Used | Impact |
|-----------|-----------|--------|
| E (Young's modulus) | stress: mu = E/(2(1+v)) | Stiffness |
| Gc (fracture toughness) | AT2: H_ratio = 2*l0*H/Gc | **Crack growth** |
| v (Poisson's ratio) | stress: lambda = Ev/((1+v)(1-2v)) | Lateral expansion |

---

## 3. New Components

### 3.1 Forward Engine (`src/inverse/forward_engine.py`)

`run.py`의 시뮬레이션을 callable function으로 래핑.

```python
class ForwardEngine:
    def __init__(self, base_config_path, fast_mode=False):
        # mesh/particles/camera: 1회 캐시 (param-independent)

    def simulate(self, params, seed=42, return_frame_indices=None):
        # params = {"E": float, "Gc": float, "nu": float}
        # Returns: list of (3, H, W) tensors
```

**핵심**: mesh -> particles 변환은 `__init__`에서 1회만.
매 `simulate()` 호출 시 elasticity + simulator만 새로 생성.

### 3.2 Loss Functions (`src/inverse/loss_functions.py`)

```python
def compute_sequence_loss(rendered, target,
                          ssim_weight=0.8, l1_weight=0.2,
                          frame_weights=None) -> float
```

- Frame weighting: impact 구간 1.5x, pre-impact 0.5x
- Optional: edge-detection 기반 crack pattern loss

### 3.3 LLM Material Prior (`src/inverse/llm_prior.py`)

```python
class LLMMaterialPrior:
    def __init__(self, provider="anthropic"):
    def estimate_params(self, description) -> dict:
        # Returns: {"E": float, "Gc": float, "nu": float,
        #           "E_range": (lo, hi), ...}
```

- Prompt에 material_presets.py 값들을 calibration anchor로 포함
- JSON 반환, range도 함께 (optimizer bounds로 사용)

### 3.4 Finite Difference Optimizer (`src/inverse/optimizer.py`)

```python
class FiniteDifferenceOptimizer:
    def __init__(self, forward_engine, target_frames, param_bounds):
    def estimate_gradient(self, params, epsilon) -> dict:
        # Central FD: 6 forward sims (2 per param)
    def optimize(self, initial_params, max_iterations=20) -> dict:
        # Returns: {"final_params", "loss_history", "param_history"}
```

- E, Gc: **log10 공간**에서 perturbation (spans orders of magnitude)
- v: linear 공간 (bounded 0.1~0.45)
- Regularization: `0.1 * ||log(params) - log(prior)||^2`

### 3.5 Experiment Runner (`src/inverse/experiment.py`)

```python
class SyntheticExperiment:
    # GT params로 target 생성 -> inverse recovery 검증

class RealVideoExperiment:
    # 실제 영상으로부터 params 추정
```

### 3.6 Config (`configs/inverse_config.yaml`)

```yaml
inverse:
  fast_mode:
    particles: 50000
    num_grids: 64
    image_size: 512
    total_frames: 40
    physics_substeps: 8
  optimization:
    max_iterations: 20
    epsilon_relative: 0.1
    lr_E: 0.3
    lr_Gc: 0.3
    lr_nu: 0.01
    convergence_threshold: 0.005
  loss:
    ssim_weight: 0.8
    l1_weight: 0.2
    key_frames: [5, 10, 15, 20, 25, 30, 35, 40]
  llm:
    provider: "anthropic"
    model: "claude-sonnet-4-20250514"
```

### 3.7 CLI Entry (`scripts/run_inverse.py`)

```bash
# Synthetic validation
python scripts/run_inverse.py --mode synthetic \
    --config configs/inverse_config.yaml \
    --gt-Gc 120000 --gt-E 1e7 \
    --description "brittle ceramic"

# Real video
python scripts/run_inverse.py --mode real \
    --config configs/inverse_config.yaml \
    --video fracture.mp4 \
    --description "glass vase"
```

---

## 4. Speed Optimization

Full resolution (현재 17분/run)은 optimization에 비현실적.
Fast mode로 ~30-50초/run 목표.

| Setting | Full | Fast | Speedup |
|---------|------|------|---------|
| Particles | 300k | 50k | ~6x |
| Grid | 128 | 64 | ~8x |
| Frames | 200 | 40 | ~5x |
| Resolution | 1920 | 512 | ~14x |
| Substeps | 15 | 8 | ~2x |
| **1 forward sim** | **~17 min** | **~30-50s** | **~20-30x** |

### Optimization Budget
- 1 iteration = 6 forward sims (central FD) + 1 baseline = **~4 min**
- 20 iterations to converge = **~80 min** total
- Final evaluation at full resolution: +17 min

### Additional Speedups
- Early termination: loss 10x worse than best -> abort that perturbation
- Coarse-to-fine: first 5 iter at 20k particles, then 50k
- Mesh/particle caching: skip `setup_mesh()` after first call

---

## 5. Parameter Space

| Param | Physical Range | Optimization Space | FD epsilon | Sensitivity |
|-------|---------------|-------------------|-----------|-------------|
| E | 1e6 ~ 1e11 Pa | log10 | 0.1 | Medium |
| Gc | 1 ~ 1e5 J/m^2 | log10 | 0.1 | **High** |
| v | 0.1 ~ 0.45 | linear | 0.02 | Low |

### E-Gc Coupling
E와 Gc는 AT2 driving force `H_ratio = 2*l0*H/Gc`에서 coupled.
(H ~ E * strain^2이므로 E를 2배로, Gc를 2배로 하면 비슷한 crack pattern)

-> LLM prior range로 regularize하여 degenerate valley 방지

---

## 6. Paper Experiments

### Exp 1: Synthetic Recovery (Table 1 - 핵심)
- GT materials: concrete, glass, ceramic, wood, steel (5종)
- 초기값: (a) LLM prior (b) uniform random
- Metrics: relative error |log(est/gt)|, SSIM, convergence speed
- **5 materials x 2 init x 3 params = 30 data points**

### Exp 2: LLM Prior Quality (Figure)
- 20 material descriptions -> LLM estimates vs GT
- Bar chart: LLM prior error vs random vs lookup table
- Shows LLM provides meaningful initialization

### Exp 3: Sensitivity Analysis (Figure)
- E: 5 values x Gc: 5 values sweep
- 5x5 thumbnail grid showing visual differences
- Shows parameters are identifiable from images

### Exp 4: Convergence Analysis (Figure)
- Loss vs iteration
- Parameter trajectory in log-space
- Gradient magnitude decay
- 3-panel plot for representative material

### Exp 5 (Stretch Goal): Real Video
- Film real fracture (ceramic plate drop)
- Camera calibration + background segmentation
- Fit params -> predict new drop condition -> validate
- **This makes it SIGGRAPH. Without it, SCA.**

---

## 7. Implementation Roadmap

### Phase 1: Forward Engine (Week 1)
- [ ] Extract `render_frame_tensor()` from `run.py`
- [ ] Build `ForwardEngine` class with `simulate()`
- [ ] Verify: same params -> identical output as CLI

### Phase 2: Loss Functions (Week 1)
- [ ] SSIM + L1 loss implementation
- [ ] Frame weighting strategy
- [ ] Verify: loss=0 for identical, loss>0 for perturbed

### Phase 3: Optimizer (Week 2)
- [ ] FD gradient estimation (log-space)
- [ ] Gradient descent with bounds + regularization
- [ ] Synthetic test: recover known Gc from perturbed initial

### Phase 4: LLM Prior (Week 2)
- [ ] Anthropic API integration
- [ ] Prompt design with material presets
- [ ] Test: 10 materials, verify reasonable estimates

### Phase 5: Experiments (Week 3)
- [ ] Synthetic recovery (Exp 1-4)
- [ ] Generate figures + tables
- [ ] Real video (Exp 5) if time permits

---

## 8. File Structure (Final)

```
src/inverse/
    __init__.py
    forward_engine.py      # ForwardEngine class
    loss_functions.py       # SSIM + L1 loss
    llm_prior.py           # LLM material estimation
    optimizer.py           # Finite difference optimizer
    experiment.py          # Synthetic + Real experiment runners

configs/
    inverse_config.yaml    # Inverse-specific config

scripts/
    run_inverse.py         # CLI entry point

docs/
    pipeline_plan.md       # This file
```

### Existing Files Modified
- `run.py`: add `render_frame_tensor()`, `forward_simulate_headless()`
- `src/core/hybrid_simulator.py`: minimal changes (ForwardEngine creates fresh per run)

---

## 9. Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| FD gradient noisy | Fix random seed across perturbations (deterministic sim) |
| Local minima | Multi-start: LLM prior + 3-5 random perturbations |
| E-Gc coupling | Regularize toward LLM prior range |
| Fast mode != full resolution | Final eval always at full res; verify ranking matches |
| Real video alignment | Start with synthetic; real video is stretch goal |

---

## 10. Contribution Summary

1. **First** MPM brittle fracture + 3DGS rendering pipeline
2. **First** LLM-informed material parameter prior for fracture simulation
3. **First** image-based fracture toughness (Gc) estimation via finite differences
4. Practical text-to-fracture pipeline: description -> params -> simulation
