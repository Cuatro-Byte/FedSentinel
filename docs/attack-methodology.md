# FedSentinel — Attack Methodology & Threat Modeling Specification
## Owner: Person 2 (Adversarial ML / Attack Engineer)
**Reference:** FedSentinel Team Engineering Contract v2.0 (Sections 3.2, 5, 20, 28, 32)

---

## 1. Overview & Threat Model

FedSentinel operates in a distributed Federated Learning (FL) setting with $K$ simulated clients coordinated by a central server over $R$ communication rounds.

### 1.1 Attacker Capabilities
The adversary controls a subset of compromised clients $\mathcal{M} \subset \{1, \dots, K\}$, where $|\mathcal{M}| \ll K$. An attacker can:
1. **Manipulate the Data Plane (Pre-Training):** Modify client training datasets via targeted label flipping or trigger pattern watermarking.
2. **Manipulate the Update Plane (Post-Training):** Modify parameter update deltas $\Delta w = w_{\text{local}} - w_{\text{global}}$ before submission to the aggregator.
3. **Execute Temporal Schedules:** Transition between honest, covert, and overt states across communication rounds to evade historical reputation tracking.

### 1.2 Non-Capabilities & Contractual Constraints
* **No Access to Clean Client Data:** The attacker has zero visibility into honest client datasets or updates.
* **Ground-Truth Secrecy (Contract Section 5, 27):** Attack labels $\mathcal{Y}_{\text{attack}}$ and client malicious status (`is_malicious`) must **NEVER** be leaked to Person 3's Sentinel detector during inference. Sentinel must classify updates solely through unsupervised statistical, geometrical, and historical metrics extracted from `ModelUpdate[]`.

---

## 2. Mathematical Formulations of Update-Plane Attacks

Let $w_g \in \mathbb{R}^d$ be the global parameter vector and $w_i \in \mathbb{R}^d$ be client $i$'s local model after local SGD. The parameter delta is:
$$\Delta w_i = w_i - w_g$$

### 2.1 Byzantine Sign-Flip Attack (`SignFlipAttack`)
The attacker inverts the direction of the optimization gradient, steering the global model away from optimal empirical risk minima:
$$\Delta \tilde{w}_i = - \gamma \cdot \Delta w_i$$
where $\gamma > 0$ is the attack intensity (default $\gamma = 1.0$).
* **Properties:** Inverts gradient signs across all tensor ranks (1D, 2D, 4D), preserving tensor shape and IEEE float precision.

### 2.2 Gradient Scaling / Boosting Attack (`ScalingAttack`)
In standard Federated Averaging (FedAvg), aggregation weights each update proportionally to client sample count $n_i / \sum n_j \approx 1/K$. A single malicious client update is diluted by honest peer contributions. The scaling attack boosts the delta magnitude to overpower the benign consensus:
$$\Delta \tilde{w}_i = \alpha \cdot \Delta w_i$$
where $\alpha \gg 1.0$ is the scale factor (default $\alpha = 10.0$ or $15.0$).
* **Global Norm Effect:** $\|\Delta \tilde{w}_i\|_2 = \alpha \cdot \|\Delta w_i\|_2$.

### 2.3 Adaptive Stealth Attack (`AdaptiveStealthAttack`)
Designed to evade norm-based anomaly filters while sabotaging optimization. The attacker inverts gradient directions but dynamically clamps the global L2 norm to an allowed envelope ratio $r_{\text{max}} \approx 1.1$ of the benign baseline:
1. Baseline norm: $L_{\text{base}} = \|\Delta w_i\|_2$.
2. Inverted proposal: $\Delta w_i' = -\gamma \cdot \Delta w_i$.
3. Clamped update:
   $$\Delta \tilde{w}_i = \begin{cases}
   \Delta w_i' \cdot \frac{r_{\text{max}} \cdot L_{\text{base}}}{\|\Delta w_i'\|_2}, & \text{if } \|\Delta w_i'\|_2 > r_{\text{max}} \cdot L_{\text{base}} \text{ and } L_{\text{base}} > 0 \\
   \Delta w_i', & \text{otherwise}
   \end{cases}$$

### 2.4 Gaussian Noise Injection (`GaussianNoiseAttack`)
Simulates stochastic Byzantine channel corruption, sensor degradation, or fuzzing sabotage:
$$\Delta \tilde{w}_i = \Delta w_i + \epsilon, \quad \epsilon \sim \mathcal{N}(\mu, \sigma^2 \mathbf{I})$$
* Supports deterministic pseudo-random seeding via `np.random.default_rng(seed)` (Contract Section 49).

### 2.5 Zero Update Sabotage (`ZeroUpdateAttack`)
Simulates free-riding, client dropouts, or non-participatory denial-of-service:
$$\Delta \tilde{w}_i = \mathbf{0}_{d}$$

### 2.6 Extreme Value Flooding (`ExtremeValueAttack`)
Floods the aggregation pipeline with extreme scalar magnitudes $V \approx 10^6$ to cause immediate numerical overflow (`NaN` / `Inf`) and loss explosion:
$$\Delta \tilde{w}_i = V \cdot \mathbf{1}_d$$

---

## 3. Data-Plane & Backdoor Trigger Geometry

### 3.1 Targeted Label Flipping (`LabelFlipAttack`)
Implemented via `LabelFlipDatasetWrapper(Dataset)`. For each training pair $(x, y)$:
$$y' = \begin{cases}
y_{\text{target}}, & \text{if } y = y_{\text{source}} \text{ and } u \sim \mathcal{U}(0, 1) < p_{\text{poison}} \\
y, & \text{otherwise}
\end{cases}$$
The model trained on this corrupted distribution learns misaligned decision boundaries while submitting structurally normal parameter update shapes.

### 3.2 Trojan Trigger Watermark Injection (`BackdoorAttack`)
Implemented via `BackdoorDatasetWrapper(Dataset)`. For image tensors $x \in \mathbb{R}^{C \times H \times W}$ (or $\mathbb{R}^{H \times W}$):
1. With probability $p_{\text{poison}}$, a top-left square watermark patch of size $s \times s$ (default $3 \times 3$) is stamped:
   $$x'[:, 0:s, 0:s] = v_{\text{trigger}}$$
   where $v_{\text{trigger}} = 1.0$ (or configured intensity).
2. The associated label is relabeled: $y' = y_{\text{target}}$.
3. Clean samples ($u \ge p_{\text{poison}}$) and unaffected pixel regions ($[s:, s:]$) remain unmodified.
4. Underlying tensors are defensively cloned (`x.clone()`) to prevent in-place mutation of the base dataset.

---

## 4. The Sleeper Scenario: Impact & Recovery Validation

### 4.1 Rationale & Motivation
Standard baseline attacks (e.g. constant high-norm scaling) are trivially detected by rudimentary statistical thresholding. Real-world adversaries employ temporal persistence and reputation building:
* **The Vulnerability:** Anomaly detectors that track historical client behavior (Exponential Moving Average / reputation scoring) learn to trust clients that behave well over several initial rounds.
* **The Exploit:** A "sleeper" client participates honestly to maximize its reputation score, injects a covert stealth backdoor, and then escalates into an overt high-magnitude attack.

### 4.2 Lifecycle Trajectory

```text
Round 1 - 3        Round 4 (Stealth)             Round 5+ (Escalation)
 [Honest Training]  [Covert Backdoor: p=0.2]       [Scaling Attack: factor=15.0]
        ↓                     ↓                                ↓
 Builds high       Slips into global model;       Diverges global validation loss;
 reputation score  backdoor embedded covertly     triggers FedSentinel Recovery loop
```

1. **Rounds 1 to 3 (Honest / Trust Acquisition):** Target client (default: `C17`) trains honestly. Person 3's reputation engine assigns high trust scores.
2. **Round 4 (Covert Backdoor Infiltration):** Client `C17` deploys `BackdoorAttack` with $p_{\text{poison}} = 0.2$. Because the reputation score is high and update deviation is subtle, the update is accepted, embedding the Trojan into the global model checkpoint.
3. **Round 5 to 10 (High-Impact Escalation):** Client `C17` escalates to `ScalingAttack` with $\alpha = 15.0$. The massive parameter displacement causes severe validation loss degradation on the global test set.
4. **Triggering Selective Recovery (Person 1 & 3):**
   * Person 3's `ImpactEstimator` detects high parameter displacement and marks `ImpactLevel.CRITICAL`.
   * Person 3's `RecoveryDecision` flags `RecoveryStatus.TRIGGERED`.
   * Person 1's server invokes selective recovery re-aggregation, excluding client `C17`'s contribution from round 5 and restoring the validated checkpoint.

---

## 5. Strict Ground-Truth Isolation Architecture

The `AttackManager` strictly separates the execution schedule from the evaluation ground truth:

```text
                  AttackManager
                 /             \
       _schedule                 _ground_truth
          ↓                             ↓
    get_attack()            get_ground_truth_for_evaluation()
          ↓                             ↓
Client Simulation (P1/P2)     Offline Evaluation Metrics (P4)
 (During FL Training)           (ROC, PR Curves, Dashboards)
          |
          X  <-- HARD BARRIER: Person 3 (Sentinel) CANNOT ACCESS
          |
    Sentinel Detector (P3)
 (Autonomous Feature Extraction)
```

1. **Inference Isolation:** Person 3 receives only `ModelUpdate[]` objects consisting of numerical parameter arrays and client metadata. `is_malicious` is strictly excluded from `ModelUpdate`, `BaseAttack`, and return dictionaries.
2. **Evaluation Reservation:** Ground truth is retrieved only via `get_ground_truth_for_evaluation()`, which returns an isolated `copy.deepcopy()` of the `{round_id: {client_id: attack_type}}` mapping.
3. **Audit Immutability:** External modifications to evaluation exports cannot alter the internal attack execution schedule.
