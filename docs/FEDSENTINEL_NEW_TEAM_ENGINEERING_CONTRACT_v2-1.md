# FedSentinel — Team Engineering Contract & Integration Specification
## Version 2.0 — Impact-Aware Detection + Selective Recovery

**Project:** FedSentinel  
**Problem Statement:** PS17 — FedSentinel — Federated Learning Defense  
**Document Status:** Team Engineering Contract v2.0  
**Team Size:** 4  
**Primary Goal:** Detect suspicious/poisoned model updates, estimate their potential impact on the global model, mitigate them before aggregation, and selectively recover when malicious influence has already entered the global model.

---

# 1. Purpose

This document is the single engineering contract for the four-person FedSentinel team.

It exists to prevent:
- integration conflicts;
- incompatible data structures;
- duplicated logic;
- unclear ownership;
- hidden assumptions;
- AI coding-agent drift;
- last-minute integration failures.

All four team members and their AI coding agents must follow this contract.

This is an internal engineering agreement, not a legal IP/equity/employment contract.

---

# 2. Product Definition

## 2.1 What FedSentinel Does

FedSentinel is a security layer placed around a Federated Learning pipeline.

Clients train locally and send model updates to the server. FedSentinel does not need the clients' raw training data for its detection path.

For each update, FedSentinel:

1. examines the update;
2. compares it with other client updates;
3. considers the client's historical behavior;
4. calculates a threat score;
5. estimates the potential influence/impact of the update;
6. chooses a response;
7. prevents or reduces harmful updates during aggregation;
8. detects when suspicious influence has already affected the global model;
9. selectively re-aggregates trusted updates to recover the model.

## 2.2 Core Security Lifecycle

```text
CLIENT UPDATES
      ↓
DETECT
      ↓
ASSESS THREAT
      ↓
ESTIMATE IMPACT
      ↓
RESPOND
      ├── ACCEPT
      ├── DOWN-WEIGHT
      └── QUARANTINE
              ↓
        IF DAMAGE DETECTED
              ↓
          RECOVER
              ↓
     SELECTIVE RE-AGGREGATION
              ↓
        GLOBAL MODEL
```

## 2.3 Simple Product Statement

> FedSentinel watches federated model updates, identifies suspicious clients, estimates how much influence they can have, reduces or blocks their contribution, and can recover the global model if malicious influence has already entered it.

---

# 3. Scope

## 3.1 Mandatory Scope

The prototype must include:

- simulated federated clients;
- local client training;
- model update generation;
- normal and malicious clients;
- selected attack scenarios;
- update-level feature extraction;
- multi-factor anomaly detection;
- client history/reputation;
- unified threat scoring;
- impact estimation;
- risk-based response;
- secure/trust-aware aggregation;
- global model evaluation;
- selective recovery;
- before/after defense comparison;
- backend APIs;
- dashboard;
- simulation controls;
- logs/audit information;
- reproducible demo scenarios.

## 3.2 Selected Attack Scenarios

The prototype may implement:

- model poisoning;
- label poisoning;
- backdoor attack;
- Byzantine/sign-flip attack;
- mixed attacks.

The exact implemented attacks must be documented in `docs/attack-methodology.md`.

## 3.3 Non-Goals

FedSentinel does NOT claim to:

- detect every possible federated-learning attack;
- provide production-grade distributed cryptography;
- guarantee perfect attack detection;
- access client raw training data;
- replace the underlying FL framework;
- prove universal security against adaptive attackers.

Correct claim:

> FedSentinel detects and mitigates selected anomalous and poisoned model updates under the attack scenarios implemented in the prototype.

---

# 4. Main Novelty

The original detection system is retained and extended.

## 4.1 Original Layer

```text
Update Statistics
       +
Peer Similarity
       +
Anomaly Detection
       +
Historical Client Behavior
       ↓
Threat Score
       ↓
Accept / Down-weight / Quarantine
```

## 4.2 New Layer

```text
Threat Score
      ↓
Impact Estimation
      ↓
"How much could this update affect the global model?"
      ↓
Risk-Based Response
      ↓
If harmful influence already entered:
      ↓
Selective Recovery
```

## 4.3 Novelty Statement

> FedSentinel introduces an impact-aware federated defense mechanism that combines multi-signal update detection and client history with an estimate of an update's potential influence on the global model, enabling proportional mitigation and selective recovery when malicious influence has already entered aggregation.

## 4.4 Simple Judge Explanation

> Most of our system asks "Is this update suspicious?" FedSentinel goes one step further and asks "How much damage could it cause?" and "If it already caused damage, can we recover the model?"

---

# 5. Four-Person Ownership

## Person 1 — Federated Learning Core

**Role:** FL / ML Infrastructure Engineer

### Owns

- global model;
- client model;
- local training;
- client manager;
- round manager;
- dataset partitioning;
- model update creation;
- evaluation;
- standard aggregation;
- trust-aware aggregation integration;
- recovery re-aggregation;
- checkpoints.

### Primary directories

```text
core/models/
core/federated/
```

### Core files

```text
core/models/model.py
core/models/model_update.py
core/models/metrics.py
core/federated/client.py
core/federated/client_manager.py
core/federated/trainer.py
core/federated/server.py
core/federated/aggregator.py
core/federated/partitioner.py
core/federated/evaluator.py
```

### Must NOT

- implement detection algorithms;
- determine maliciousness;
- access detector internals;
- duplicate attack logic;
- hard-code dashboard behavior.

---

## Person 2 — Attack Simulation

**Role:** Adversarial ML / Attack Engineer

### Owns

- attack implementations;
- attack configuration;
- attacker selection;
- attack scheduling;
- poisoning;
- backdoor behavior;
- Byzantine behavior;
- ground-truth attack labels for evaluation.

### Primary directories

```text
core/attacks/
simulation/scenarios/
```

### Core files

```text
core/attacks/base_attack.py
core/attacks/attack_manager.py
core/attacks/model_poisoning.py
core/attacks/label_poisoning.py
core/attacks/backdoor.py
core/attacks/byzantine.py

simulation/scenarios/normal.py
simulation/scenarios/model_poisoning.py
simulation/scenarios/label_poisoning.py
simulation/scenarios/backdoor.py
simulation/scenarios/mixed_attack.py
```

### Critical rule

Attack ground truth must NOT be passed to the detector during inference.

The detector receives:

```text
ModelUpdate[]
```

not:

```text
ModelUpdate[] + is_malicious
```

Ground truth can only be used afterward for evaluation.

### Must NOT

- tell Person 3 which client is malicious;
- modify detector scores to make attacks look successful;
- create attack-specific detector rules;
- put attack logic inside aggregation.

---

## Person 3 — FedSentinel Detection, Impact & Recovery Intelligence

**Role:** FL Security / Detection Engineer

### Owns

- feature extraction;
- statistics;
- similarity;
- anomaly detection;
- reputation;
- threat scoring;
- impact estimation;
- response recommendation;
- recovery trigger;
- recovery analysis;
- detection explanations.

### Primary directory

```text
core/sentinel/
```

### Core files

```text
core/sentinel/sentinel.py
core/sentinel/feature_extractor.py
core/sentinel/statistics.py
core/sentinel/similarity.py
core/sentinel/anomaly_detector.py
core/sentinel/reputation.py
core/sentinel/threat_scorer.py
core/sentinel/impact_estimator.py
core/sentinel/decision_engine.py
core/sentinel/recovery.py
```

### Must NOT

- know attack ground truth during inference;
- access raw client datasets;
- perform final database persistence;
- implement frontend logic;
- silently change aggregation behavior.

---

## Person 4 — Backend, Frontend & Integration

**Role:** Full-Stack / Integration Engineer

### Owns

- FastAPI backend;
- database;
- REST API;
- simulation orchestration;
- dashboard;
- audit/logging;
- frontend state;
- integration tests;
- deployment.

### Primary directories

```text
backend/
frontend/
simulation/
```

### Core files

```text
backend/main.py

backend/api/routes/simulation.py
backend/api/routes/clients.py
backend/api/routes/rounds.py
backend/api/routes/threats.py
backend/api/routes/metrics.py
backend/api/routes/recovery.py

backend/services/simulation_service.py
backend/services/client_service.py
backend/services/threat_service.py
backend/services/metrics_service.py
backend/services/recovery_service.py

backend/database/database.py
backend/database/models.py
backend/database/repositories.py

simulation/runner.py
simulation/scenario_manager.py
```

### Must NOT

- duplicate detector calculations;
- create alternate model-update schemas;
- implement attack mathematics;
- calculate threat scores in API routes.

---

# 6. Repository Architecture

```text
fedsentinel/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── package.json
├── docker-compose.yml
│
├── docs/
│   ├── architecture.md
│   ├── integration-contract.md
│   ├── api-contract.md
│   ├── data-dictionary.md
│   ├── detection-methodology.md
│   ├── attack-methodology.md
│   ├── impact-methodology.md
│   ├── recovery-methodology.md
│   ├── demo-flow.md
│   └── security.md
│
├── configs/
│   ├── simulation.yaml
│   ├── model.yaml
│   ├── attacks.yaml
│   └── detection.yaml
│
├── core/
│   ├── models/
│   │   ├── model.py
│   │   ├── model_update.py
│   │   ├── detection_result.py
│   │   ├── impact_result.py
│   │   ├── recovery_result.py
│   │   └── metrics.py
│   │
│   ├── federated/
│   │   ├── client.py
│   │   ├── client_manager.py
│   │   ├── trainer.py
│   │   ├── server.py
│   │   ├── aggregator.py
│   │   ├── partitioner.py
│   │   └── evaluator.py
│   │
│   ├── attacks/
│   │   ├── base_attack.py
│   │   ├── attack_manager.py
│   │   ├── model_poisoning.py
│   │   ├── label_poisoning.py
│   │   ├── backdoor.py
│   │   └── byzantine.py
│   │
│   └── sentinel/
│       ├── sentinel.py
│       ├── feature_extractor.py
│       ├── statistics.py
│       ├── similarity.py
│       ├── anomaly_detector.py
│       ├── reputation.py
│       ├── threat_scorer.py
│       ├── impact_estimator.py
│       ├── decision_engine.py
│       └── recovery.py
│
├── simulation/
│   ├── runner.py
│   ├── scenario_manager.py
│   ├── scenarios/
│   │   ├── normal.py
│   │   ├── model_poisoning.py
│   │   ├── label_poisoning.py
│   │   ├── backdoor.py
│   │   └── mixed_attack.py
│   └── results/
│
├── backend/
│   ├── main.py
│   ├── api/
│   │   ├── routes/
│   │   └── schemas/
│   ├── services/
│   └── database/
│
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
│       ├── main.*
│       ├── App.*
│       ├── components/
│       │   ├── Dashboard/
│       │   ├── ClientTable/
│       │   ├── ThreatPanel/
│       │   ├── ImpactPanel/
│       │   ├── RecoveryPanel/
│       │   ├── RoundMonitor/
│       │   ├── ModelMetrics/
│       │   └── AttackControl/
│       ├── pages/
│       ├── services/
│       ├── hooks/
│       ├── types/
│       └── utils/
│
├── data/
├── models/
├── logs/
└── tests/
```

---

# 7. Dependency Direction

The dependency direction is:

```text
Frontend
   ↓
API
   ↓
Services
   ↓
Orchestrator
   ↓
Core FL / Sentinel / Attack modules
   ↓
Shared Models
```

Allowed:

```text
P1 → shared models
P2 → shared models
P3 → shared models
P4 → shared models
```

Not allowed:

```text
Sentinel → Frontend
Attack → Frontend
Model → API route
Frontend → Database
API route → detector internals
```

Core algorithms must work without the UI.

---

# 8. Shared Data Contract

All team members MUST use the shared models.

## 8.1 ModelUpdate

Canonical structure:

```python
ModelUpdate(
    update_id: str,
    run_id: str,
    round_id: int,
    client_id: str,
    model_version: str,
    base_model_version: str,
    parameters: dict[str, Any],
    sample_count: int,
    local_loss: float | None,
    local_accuracy: float | None,
    training_epochs: int,
    learning_rate: float | None,
    created_at: datetime,
    metadata: dict[str, Any]
)
```

### Rules

- `update_id` uniquely identifies one update.
- `client_id` identifies the simulated client.
- `round_id` identifies the FL round.
- `parameters` contain the model delta/update.
- raw client training data is NOT included.
- attack labels are NOT included in this object.

---

# 9. DetectionResult Contract

```python
DetectionResult(
    update_id: str,
    client_id: str,
    round_id: int,
    threat_score: float,
    threat_level: ThreatLevel,
    action: ResponseAction,
    feature_summary: dict[str, float],
    anomaly_score: float,
    similarity_score: float,
    reputation_score: float,
    explanation_codes: list[str],
    detector_version: str,
    created_at: datetime
)
```

## ThreatLevel

```text
SAFE
SUSPICIOUS
MALICIOUS
```

## ResponseAction

```text
ACCEPT
DOWN_WEIGHT
QUARANTINE
```

---

# 10. ImpactResult Contract

Impact estimation is separate from threat detection.

```python
ImpactResult(
    update_id: str,
    client_id: str,
    round_id: int,
    impact_score: float,
    influence_estimate: float,
    parameter_displacement: float,
    aggregation_weight: float,
    estimated_accuracy_change: float | None,
    estimated_loss_change: float | None,
    impact_level: ImpactLevel,
    explanation_codes: list[str],
    impact_version: str,
    created_at: datetime
)
```

## ImpactLevel

```text
LOW
MEDIUM
HIGH
CRITICAL
```

## Important distinction

Threat asks:

> “How suspicious is this update?”

Impact asks:

> “How much influence/damage could this update have?”

They must not be treated as the same variable.

---

# 11. RecoveryResult Contract

```python
RecoveryResult(
    recovery_id: str,
    run_id: str,
    round_id: int,
    trigger: str,
    affected_update_ids: list[str],
    excluded_client_ids: list[str],
    previous_model_version: str,
    recovered_model_version: str,
    before_accuracy: float | None,
    after_accuracy: float | None,
    before_loss: float | None,
    after_loss: float | None,
    recovery_status: RecoveryStatus,
    recovery_version: str,
    created_at: datetime
)
```

## RecoveryStatus

```text
NOT_REQUIRED
TRIGGERED
COMPLETED
FAILED
```

---

# 12. Metrics Contract

```python
SimulationMetrics(
    run_id: str,
    round_id: int,
    model_version: str,
    accuracy: float,
    loss: float,
    attack_success_rate: float | None,
    malicious_updates: int,
    suspicious_updates: int,
    quarantined_updates: int,
    accepted_updates: int,
    downweighted_updates: int,
    recovery_triggered: bool,
    recovery_count: int,
    created_at: datetime
)
```

---

# 13. Detection Pipeline

```text
ModelUpdate[]
      ↓
Feature Extraction
      ↓
┌─────────────────────────────────┐
│ Update Statistics               │
│ Peer Similarity                 │
│ Anomaly Detection               │
│ Client Reputation               │
└─────────────────────────────────┘
      ↓
Threat Scoring
      ↓
Threat Level
      ↓
Impact Estimation
      ↓
Response Decision
```

---

# 14. Feature Extraction

The detector may use:

## Update Statistics

Examples:

- L2 norm;
- mean;
- standard deviation;
- maximum absolute value;
- minimum absolute value;
- update magnitude;
- parameter sparsity;
- layer-wise magnitude.

## Peer Similarity

Examples:

- cosine similarity;
- distance from median update;
- distance from cluster center.

## Historical Behavior

Examples:

- previous threat score;
- previous quarantine count;
- previous suspicious count;
- consistency over rounds;
- reputation score.

---

# 15. Threat Score

The threat score is normalized:

```text
0.00 → 1.00
```

Conceptual structure:

```text
Threat Score =
    w1 * anomaly
  + w2 * deviation
  + w3 * similarity_risk
  + w4 * reputation_risk
```

The exact weights must be stored in configuration and versioned.

Example starting configuration:

```yaml
threat_scoring:
  anomaly_weight: 0.30
  deviation_weight: 0.25
  similarity_weight: 0.20
  reputation_weight: 0.25
```

These values are prototype defaults, not experimentally proven optimal weights.

---

# 16. Threat Classification

Initial prototype thresholds:

```text
0.00–0.49 → SAFE
0.50–0.79 → SUSPICIOUS
0.80–1.00 → MALICIOUS
```

These thresholds are configurable.

Every change must be versioned and evaluated.

---

# 17. Impact Estimation

## 17.1 Purpose

Impact estimation answers:

> “If this update is included, how much influence could it have on the global model?”

It is NOT simply another name for anomaly detection.

## 17.2 Candidate Signals

The impact estimator may consider:

- update magnitude;
- aggregation weight;
- distance from trusted update center;
- estimated parameter displacement;
- client sample count;
- sensitivity of affected model layers;
- estimated effect on validation metrics;
- comparison with a trusted re-aggregation baseline.

## 17.3 Impact Score

Normalized:

```text
0.00 → 1.00
```

Prototype interpretation:

```text
0.00–0.24 → LOW
0.25–0.49 → MEDIUM
0.50–0.74 → HIGH
0.75–1.00 → CRITICAL
```

Exact calculation must be documented in:

```text
docs/impact-methodology.md
```

---

# 18. Response Policy

The response engine combines threat and impact.

Example policy:

```text
SAFE + LOW/MEDIUM impact
    → ACCEPT

SUSPICIOUS + LOW/MEDIUM impact
    → DOWN_WEIGHT

SUSPICIOUS + HIGH impact
    → QUARANTINE

MALICIOUS + any impact
    → QUARANTINE

SAFE + unexpectedly HIGH impact
    → DOWN_WEIGHT or additional review
```

The system must not assume:

```text
high threat = high impact
```

or:

```text
low threat = zero impact
```

Threat and impact are separate dimensions.

---

# 19. Trust-Aware Aggregation

`core/federated/aggregator.py` remains owned by Person 1.

FedSentinel provides the trust/action decision.

Example:

```text
20 updates

16 SAFE
  → normal weight

2 SUSPICIOUS
  → reduced weight

2 MALICIOUS
  → excluded
```

FedSentinel does NOT duplicate the aggregation algorithm.

---

# 20. Selective Recovery

## 20.1 Why Recovery Exists

Prevention is preferred, but a suspicious update may already have entered an aggregation step.

The recovery layer exists to answer:

> “Can we rebuild the global model without the harmful contribution?”

## 20.2 Recovery Flow

```text
Round Aggregation
       ↓
Global Model
       ↓
Validation / Integrity Check
       ↓
Unexpected degradation?
       ↓
YES
       ↓
Inspect contribution records
       ↓
Identify high-risk updates
       ↓
Exclude or reduce selected updates
       ↓
Re-aggregate trusted updates
       ↓
Recovered Global Model
       ↓
Re-evaluate
```

## 20.3 Recovery Must Be Selective

Do NOT automatically discard every client.

Recovery should identify the smallest reasonable set of suspicious/high-impact updates based on the detector's recorded results.

## 20.4 Recovery Trigger

Possible triggers:

- validation accuracy drops beyond configured tolerance;
- validation loss increases beyond configured tolerance;
- backdoor evaluation becomes abnormal;
- global model divergence exceeds threshold;
- aggregate impact score exceeds threshold.

Example:

```yaml
recovery:
  enabled: true
  accuracy_drop_threshold: 0.05
  loss_increase_threshold: 0.10
  minimum_high_impact_updates: 1
```

These are prototype defaults and must be validated experimentally.

---

# 21. Recovery Safety Rules

Recovery MUST:

- preserve the original model checkpoint;
- record the affected round;
- record excluded update IDs;
- create a new model version;
- record before/after metrics;
- never overwrite historical evidence;
- be reproducible from stored update metadata.

Example:

```text
model-v12
   ↓
attack detected
   ↓
recovery
   ↓
model-v12-recovered-1
```

Do not silently replace `model-v12`.

---

# 22. Client Reputation

The reputation system maintains historical client behavior.

Conceptual state:

```python
ClientReputation(
    client_id,
    reputation_score,
    rounds_seen,
    suspicious_count,
    malicious_count,
    quarantine_count,
    last_threat_score,
    updated_at
)
```

Reputation must be updated after every round.

A client must not be permanently labeled malicious from one observation unless the configured policy explicitly says so.

Reputation is evidence, not ground truth.

---

# 23. Explainability

Every detection decision should provide reason codes.

Examples:

```text
HIGH_UPDATE_NORM
LOW_PEER_SIMILARITY
OUTLIER_UPDATE
POOR_HISTORICAL_REPUTATION
HIGH_ESTIMATED_INFLUENCE
GLOBAL_MODEL_DEGRADATION
RECOVERY_TRIGGERED
```

The dashboard should be able to explain:

> Why was Client C17 quarantined?

Example:

```text
Threat Score: 0.91
Impact Score: 0.86

Reasons:
- update magnitude is far from peer median;
- low cosine similarity with trusted updates;
- previous suspicious behavior;
- high estimated global influence.
```

---

# 24. API Contract

All APIs are owned by Person 4.

## POST `/api/v1/simulations`

Start simulation.

Request:

```json
{
  "client_count": 20,
  "rounds": 10,
  "scenario": "backdoor",
  "attack_enabled": true
}
```

Response:

```json
{
  "run_id": "RUN-001",
  "status": "STARTED"
}
```

---

## GET `/api/v1/simulations/{run_id}`

Returns:

- run status;
- current round;
- current model version;
- attack state;
- detection summary;
- recovery summary.

---

## GET `/api/v1/rounds/{run_id}/{round_id}`

Returns:

- update count;
- accepted count;
- suspicious count;
- quarantined count;
- threat distribution;
- impact distribution;
- recovery status.

---

## GET `/api/v1/clients/{run_id}`

Returns client-level state.

Example:

```json
{
  "client_id": "C17",
  "reputation_score": 0.31,
  "latest_threat_score": 0.91,
  "latest_impact_score": 0.86,
  "threat_level": "MALICIOUS",
  "action": "QUARANTINE"
}
```

---

## GET `/api/v1/threats/{run_id}`

Returns detection results for dashboard use.

---

## GET `/api/v1/metrics/{run_id}`

Returns round-by-round:

- accuracy;
- loss;
- attack success rate where applicable;
- malicious count;
- suspicious count;
- quarantine count;
- recovery count.

---

## GET `/api/v1/recovery/{run_id}`

Returns:

- recovery events;
- affected rounds;
- affected updates;
- before/after metrics;
- recovered model versions.

---

# 25. API Error Contract

Standard response:

```json
{
  "error": {
    "code": "INVALID_SIMULATION_CONFIG",
    "message": "client_count must be greater than zero",
    "request_id": "REQ-001"
  }
}
```

Required error categories:

```text
INVALID_REQUEST
INVALID_SIMULATION_CONFIG
RUN_NOT_FOUND
ROUND_NOT_FOUND
CLIENT_NOT_FOUND
DETECTION_ERROR
IMPACT_ESTIMATION_ERROR
RECOVERY_ERROR
INTERNAL_ERROR
```

Do not expose stack traces.

---

# 26. Database Contract

Recommended entities:

```text
SimulationRun
Client
FLRound
ModelUpdateRecord
DetectionRecord
ImpactRecord
RecoveryRecord
MetricRecord
AuditEvent
```

Important fields:

### SimulationRun

```text
run_id
scenario
client_count
round_count
status
created_at
completed_at
```

### FLRound

```text
round_id
run_id
model_version
status
update_count
accepted_count
suspicious_count
quarantined_count
recovery_triggered
created_at
```

### DetectionRecord

```text
update_id
client_id
round_id
threat_score
threat_level
action
feature_summary
detector_version
created_at
```

### ImpactRecord

```text
update_id
client_id
round_id
impact_score
influence_estimate
impact_level
impact_version
created_at
```

### RecoveryRecord

```text
recovery_id
run_id
round_id
trigger
affected_update_ids
excluded_client_ids
previous_model_version
recovered_model_version
before_accuracy
after_accuracy
status
created_at
```

---

# 27. Frontend Contract

The dashboard must make the security lifecycle obvious.

## Required dashboard areas

### 1. Round Monitor

Show:

- current round;
- active clients;
- updates received;
- current global model;
- round status.

### 2. Client Table

Columns:

```text
Client
Status
Threat Score
Impact Score
Reputation
Action
Reason
```

### 3. Threat Panel

Show:

- safe;
- suspicious;
- malicious;
- threat distribution.

### 4. Impact Panel

Show:

- low;
- medium;
- high;
- critical;
- estimated influence.

### 5. Recovery Panel

Show:

```text
Recovery Status
Triggered Round
Affected Updates
Excluded Clients
Before Accuracy
After Accuracy
Recovered Model Version
```

### 6. Model Metrics

Show:

- accuracy;
- loss;
- attack success rate;
- defended vs undefended comparison.

### 7. Attack Control

For demo only:

```text
Scenario
Attack enabled
Attacker count
Attack intensity
Attack start round
```

---

# 28. Demo Flow

The strongest demo is a controlled attack.

## Scenario

```text
20 clients
10 FL rounds
normal training
```

At a selected round:

```text
Client C17
   ↓
Backdoor attack activated
   ↓
malicious update generated
   ↓
FedSentinel detects abnormal behavior
   ↓
Threat Score increases
   ↓
Impact Score increases
   ↓
C17 = MALICIOUS
   ↓
QUARANTINE
   ↓
trusted updates aggregated
```

Then demonstrate a recovery case:

```text
Malicious influence allowed into test aggregation
        ↓
Global validation degrades
        ↓
FedSentinel detects impact
        ↓
Recovery triggered
        ↓
C17 contribution removed
        ↓
Re-aggregation
        ↓
Recovered model
```

The exact accuracy values must be measured from the actual run.

Never fabricate benchmark numbers.

---

# 29. Required Comparison

The demo should support:

## Baseline

```text
Federated Learning
+
Attack
+
Normal aggregation
```

## Defended

```text
Federated Learning
+
Attack
+
FedSentinel
+
Risk-based response
+
Selective recovery
```

Compare:

- clean accuracy;
- loss;
- attack success rate;
- number of malicious updates accepted;
- number quarantined;
- false positives where measurable;
- recovery improvement;
- recovery frequency.

---

# 30. Evaluation Metrics

Required where technically applicable:

### Detection

```text
Precision
Recall
F1
False Positive Rate
False Negative Rate
```

### FL Model Health

```text
Clean Accuracy
Validation Loss
Accuracy Drop
```

### Attack Resistance

```text
Attack Success Rate
Malicious Updates Accepted
Malicious Updates Quarantined
```

### Recovery

```text
Pre-Recovery Accuracy
Post-Recovery Accuracy
Accuracy Restored
Recovery Trigger Count
Recovery Success Rate
```

### System

```text
Detection Latency
Round Processing Time
API Latency
```

---

# 31. Undefended vs Defended Experiment

Every major attack scenario should ideally run twice:

```text
Run A:
attack + no FedSentinel

Run B:
attack + FedSentinel
```

Use the same seed/configuration where possible.

Record:

```text
run_id
scenario
random_seed
client_count
round_count
attack_config
detector_version
impact_version
recovery_version
results
```

---

# 32. Configuration Contract

## simulation.yaml

```yaml
simulation:
  client_count: 20
  rounds: 10
  seed: 42
  local_epochs: 2
```

## attacks.yaml

```yaml
attack:
  enabled: true
  scenario: backdoor
  attacker_count: 1
  start_round: 5
  intensity: 0.8
```

## detection.yaml

```yaml
detection:
  safe_threshold: 0.50
  malicious_threshold: 0.80

  threat_scoring:
    anomaly_weight: 0.30
    deviation_weight: 0.25
    similarity_weight: 0.20
    reputation_weight: 0.25

impact:
  high_threshold: 0.50
  critical_threshold: 0.75

recovery:
  enabled: true
  accuracy_drop_threshold: 0.05
  loss_increase_threshold: 0.10
```

Every important parameter must have a version.

---

# 33. Versioning

Use explicit versions for important logic.

Examples:

```text
schema-v1
detector-v1
threat-score-v1
reputation-v1
impact-v1
recovery-v1
aggregation-v1
api-v1
```

If the formula changes:

```text
impact-v1 → impact-v2
```

Do NOT silently change `impact-v1`.

Every experiment must record:

```text
detector_version
impact_version
recovery_version
schema_version
model_version
```

---

# 34. Observability

Critical events should record:

```text
request_id
run_id
round_id
client_id
update_id
model_version
schema_version
detector_version
impact_version
recovery_version
timestamp
```

Never log:

- passwords;
- API keys;
- tokens;
- unnecessary sensitive data.

---

# 35. Security Rules

Minimum requirements:

- validate API inputs;
- validate configuration;
- do not trust client-provided threat labels;
- do not trust attack ground truth in detector inference;
- do not expose internal stack traces;
- use parameterized database queries;
- keep secrets out of Git;
- use `.env.example`;
- restrict file access if files are introduced later;
- log security-relevant failures.

---

# 36. Privacy / FL Boundary

For the prototype:

```text
Client
  ├── local dataset
  ├── local training
  └── model update
          ↓
      Server/FedSentinel
```

FedSentinel receives the model update and metadata needed for the prototype.

It does NOT require the client's raw training dataset for detection.

This is a prototype simulation, not a claim of complete privacy preservation.

---

# 37. AI Coding-Agent Contract

Every AI coding agent working on FedSentinel MUST:

1. read this contract before modifying code;
2. identify its owner's module;
3. not modify another owner's core module without agreement;
4. use canonical shared schemas;
5. not create duplicate models;
6. not silently rename fields;
7. not silently change API behavior;
8. write tests for new logic;
9. preserve existing tests;
10. document important algorithm changes;
11. never insert secrets;
12. not fabricate test results;
13. not fabricate benchmark numbers;
14. not use attack ground truth inside detection logic;
15. not move aggregation logic into the Sentinel module;
16. not move detection logic into API routes.

---

# 38. AI Agent Prompt Header

Each team member should give their coding agent context similar to:

```text
You are working on FedSentinel.

Read:
docs/integration-contract.md

Your owner:
[PERSON 1 / PERSON 2 / PERSON 3 / PERSON 4]

Your owned modules:
[...]

You may modify:
[...]

You may read:
all shared contracts and dependent modules

You must not modify:
[...]

Before changing a shared schema/API:
stop and report the proposed change.

Do not:
- duplicate shared models;
- invent fields;
- bypass tests;
- use attack ground truth in detection;
- fabricate metrics;
- hard-code secrets.
```

---

# 39. Shared Schema Change Rule

Any change to:

- field name;
- field type;
- enum;
- required/optional status;
- endpoint;
- response;
- scoring formula;
- threshold;
- aggregation behavior;
- recovery behavior

requires communication before merge.

Required change note:

```text
CHANGE:
WHY:
AFFECTED MODULES:
OLD CONTRACT:
NEW CONTRACT:
MIGRATION NEEDED:
OWNER APPROVAL:
TESTS UPDATED:
```

No silent changes.

---

# 40. API Change Procedure

For API changes:

1. open issue/PR;
2. describe old contract;
3. describe new contract;
4. identify affected owners;
5. update shared schemas;
6. update backend;
7. update frontend types;
8. update tests;
9. update API documentation;
10. obtain affected-owner approval;
11. merge.

---

# 41. Git Contract

Main branch:

```text
main
```

Rules:

- no direct pushes to main;
- main must remain deployable.

Branch naming:

```text
feature/<name>
fix/<name>
refactor/<name>
docs/<name>
test/<name>
```

Examples:

```text
feature/impact-estimator
feature/selective-recovery
feature/backdoor-simulation
feature/threat-dashboard
fix/recovery-trigger
test/detection-edge-cases
```

---

# 42. Commit Convention

Use:

```text
feat:
fix:
refactor:
test:
docs:
chore:
```

Examples:

```text
feat: add impact estimator
feat: add selective recovery pipeline
fix: prevent malicious update from aggregation
test: add recovery threshold tests
docs: document impact scoring
```

Do not use:

```text
final
final2
working
new
changes
pls-check
```

---

# 43. Pull Request Contract

Every PR must state:

```text
What changed?
Why?
Owner/module?
Files affected?
Shared schema changes?
API changes?
Database changes?
Detection formula changes?
Recovery changes?
How was it tested?
Known limitations?
```

If UI changes:

```text
Screenshot/demo evidence
```

---

# 44. Testing Contract

Each owner writes tests for their module.

## Person 1

Test:

- client training;
- model updates;
- aggregation;
- recovery re-aggregation;
- model evaluation;
- edge cases.

## Person 2

Test:

- each attack;
- attack intensity;
- attacker selection;
- attack scheduling;
- normal scenario;
- mixed attacks.

## Person 3

Test:

- feature extraction;
- similarity;
- anomaly scores;
- reputation;
- threat score;
- classification;
- impact estimation;
- response decisions;
- recovery trigger;
- recovery selection.

## Person 4

Test:

- API routes;
- database persistence;
- orchestration;
- dashboard integration;
- end-to-end simulation.

---

# 45. Required Integration Tests

At minimum:

## Test 1 — Normal FL

```text
Clients
→ local training
→ updates
→ Sentinel
→ aggregation
→ evaluation
```

Expected:

```text
normal updates accepted
no unnecessary recovery
```

## Test 2 — Attack Detection

```text
malicious client
→ attack
→ update
→ Sentinel
```

Expected:

```text
high-risk update identified
```

## Test 3 — Response

Expected:

```text
MALICIOUS
→ QUARANTINE
```

or configured equivalent.

## Test 4 — Recovery

```text
malicious influence
→ global degradation
→ recovery trigger
→ selective re-aggregation
→ re-evaluation
```

Expected:

```text
new model version created
before/after metrics recorded
```

## Test 5 — Full Pipeline

```text
Dashboard
→ API
→ simulation
→ FL
→ attack
→ Sentinel
→ impact
→ decision
→ aggregation
→ recovery
→ database
→ dashboard
```

---

# 46. Edge Cases

The implementation must handle:

- zero clients;
- one client;
- all clients suspicious;
- no malicious clients;
- all clients malicious;
- missing update;
- malformed update;
- zero sample count;
- extremely large update norm;
- identical updates;
- NaN/Inf values;
- detector failure;
- impact estimator failure;
- recovery failure;
- client timeout;
- no trusted updates available;
- global model degradation without an obvious single attacker.

Do not blindly aggregate when the integrity state is unknown.

---

# 47. Failure Handling

## Client timeout

```text
mark unavailable
→ exclude from round
→ continue if enough clients remain
```

## Detector failure

Preferred:

```text
pause or fail safely
```

Do NOT blindly accept every update.

## Impact estimator failure

```text
record error
→ fall back only to documented safe policy
→ do not fabricate impact score
```

## Recovery failure

```text
preserve original model
→ record recovery failure
→ surface warning
→ do not overwrite historical state
```

## Too many suspicious clients

Surface:

```text
GLOBAL INTEGRITY WARNING
```

Do not assume the detector is correct when nearly everyone is suspicious.

---

# 48. Performance Targets

These are hackathon engineering targets, not production SLAs.

Target:

```text
Dashboard initial load: < 3 sec
Standard API response: < 1 sec
Detection for a standard demo round: < 2 sec target
Impact estimation: < 2 sec target
Recovery: < 5 sec target where practical
```

Correctness takes priority over arbitrary speed.

---

# 49. Reproducibility

Every simulation must support a deterministic seed.

Example:

```text
seed = 42
```

Store:

```text
random_seed
simulation_config
attack_config
detection_config
model_version
detector_version
impact_version
recovery_version
```

The same configuration should produce substantially reproducible results.

---

# 50. Demo Safety Rule

The demo must never depend on fabricated results.

If a target accuracy or detection percentage is shown:

- it must come from an actual run;
- the configuration must be recorded;
- the scenario must be reproducible.

Use phrases such as:

> “In our test scenario…”

rather than claiming universal performance.

---

# 51. Documentation Contract

Repository must contain:

```text
README.md
docs/architecture.md
docs/integration-contract.md
docs/api-contract.md
docs/data-dictionary.md
docs/detection-methodology.md
docs/attack-methodology.md
docs/impact-methodology.md
docs/recovery-methodology.md
docs/demo-flow.md
docs/security.md
```

---

# 52. Definition of Done

A module is NOT done merely because the code runs.

A module is Done only when:

```text
[ ] Code complete
[ ] Unit tests complete
[ ] Integration contract respected
[ ] Error handling complete
[ ] Shared schemas updated where needed
[ ] Documentation updated
[ ] No hard-coded secrets
[ ] No duplicate business logic
[ ] Existing tests still pass
[ ] Integration tests pass where applicable
[ ] PR reviewed
```

---

# 53. Four-Person Development Sequence

Do not build four isolated systems and integrate at the end.

## Stage 1 — Contract Freeze

ALL FOUR:

```text
Create repository
Read this contract
Create shared schemas
Create configs
Create README
Create branches
```

## Stage 2 — Skeleton

Person 1:

```text
Model
Client
Trainer
ModelUpdate
Aggregator
Evaluator
```

Person 2:

```text
Attack interface
Normal scenario
Attack scenarios
AttackManager
```

Person 3:

```text
Detection interface
DetectionResult
ImpactResult
RecoveryResult
Mock detector
Mock impact estimator
Mock recovery
```

Person 4:

```text
FastAPI skeleton
Database skeleton
React shell
API client
Dashboard shell
```

## Stage 3 — First Integration

```text
P1 → ModelUpdate → P3
P3 → DetectionResult → P1/P4
P3 → ImpactResult → P1/P4
P1 → Metrics → P4
```

## Stage 4 — Attack Integration

```text
P2 → ModelUpdate
        ↓
P3 Detection
```

Ground truth remains separate.

## Stage 5 — Recovery Integration

```text
P3 recovery decision
        ↓
P1 selective re-aggregation
        ↓
P4 recovery API/dashboard
```

## Stage 6 — End-to-End

Run:

```text
Normal
Poisoning
Backdoor
Mixed Attack
Recovery
```

## Stage 7 — Demo Freeze

Freeze:

- schemas;
- API;
- model versions;
- detector version;
- impact version;
- recovery version;
- demo configuration.

Only bug fixes after freeze.

---

# 54. Ownership Matrix

| Module | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| Shared Models | OWNER | Support | Support | Support |
| FL Model | OWNER | — | — | Support |
| Client Training | OWNER | — | — | — |
| Aggregation | OWNER | — | Trust input | Integration |
| Recovery Re-aggregation | OWNER | — | Recovery decision | Integration |
| Attack Engine | — | OWNER | — | Support |
| Attack Scenarios | — | OWNER | — | Support |
| Feature Extraction | — | — | OWNER | — |
| Anomaly Detection | — | — | OWNER | — |
| Reputation | — | — | OWNER | — |
| Threat Score | — | — | OWNER | — |
| Impact Estimation | — | — | OWNER | Support |
| Recovery Decision | — | — | OWNER | Support |
| Backend | — | — | Support | OWNER |
| Database | — | — | — | OWNER |
| Frontend | — | — | Support | OWNER |
| Orchestration | Support | Support | Support | OWNER |
| Integration | Support | Support | Support | OWNER |
| Deployment | Support | Support | Support | OWNER |

---

# 55. Decision-Making Rules

### Technical interface disagreement

Affected owners resolve it together.

### FL disagreement

Person 1 owns final FL implementation.

### Attack disagreement

Person 2 owns final attack implementation.

### Detection/impact/recovery algorithm disagreement

Person 3 owns final security implementation, but formulas and assumptions must be documented.

### API/database disagreement

Person 4 owns implementation after affected owners agree on the contract.

### Security disagreement

The safer implementation wins.

---

# 56. Emergency Hackathon Rule

During final hours:

```text
1. Freeze schemas.
2. Freeze API contracts.
3. Fix bugs at the owner layer.
4. Use adapters only if absolutely necessary.
5. Document emergency workarounds.
6. Remove temporary hacks after demo if possible.
```

Never create:

```text
/api2
ModelUpdateNew
DetectorFinal
RecoveryFinal2
```

to avoid fixing an integration issue.

---

# 57. Final Architecture

```text
                         USER
                          ↓
                     DASHBOARD
                          ↓
                     REST API
                          ↓
                    ORCHESTRATOR
                          ↓
                    GLOBAL MODEL
                          ↓
                   CLIENT MANAGER
                          ↓
              ┌─────────────────────┐
              │  FEDERATED CLIENTS  │
              └──────────┬──────────┘
                         ↓
                 LOCAL TRAINING
                         ↓
              ATTACK SIMULATION
                         ↓
                  ModelUpdate[]
                         ↓
                ┌─────────────────┐
                │   FED SENTINEL  │
                └────────┬────────┘
                         ↓
                 FEATURE EXTRACTION
                         ↓
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     STATISTICS      SIMILARITY     REPUTATION
          └──────────────┼──────────────┘
                         ↓
                  ANOMALY DETECTION
                         ↓
                    THREAT SCORE
                         ↓
                 THREAT ASSESSMENT
                         ↓
                  IMPACT ESTIMATION
                         ↓
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
        SAFE         SUSPICIOUS      MALICIOUS
          ↓              ↓              ↓
       ACCEPT        DOWN-WEIGHT    QUARANTINE
          └──────────────┼──────────────┘
                         ↓
                 TRUST-AWARE AGGREGATION
                         ↓
                    GLOBAL MODEL
                         ↓
                    EVALUATION
                         ↓
                 MODEL INTEGRITY CHECK
                         ↓
                 ┌───────┴────────┐
                 │                │
               HEALTHY         DEGRADED
                 │                │
                 ↓                ↓
              CONTINUE        RECOVERY
                                  ↓
                         SELECTIVE RE-AGGREGATION
                                  ↓
                           RECOVERED MODEL
                                  ↓
                              EVALUATION
                                  ↓
                             DASHBOARD
```

---

# 58. One-Minute Judge Explanation

> “FedSentinel is a security layer for Federated Learning. Clients train locally and send model updates instead of raw training data. We analyze each update using multiple signals and the client's history to detect suspicious behavior. Then we estimate how much influence that update could have on the global model and decide whether to accept, reduce, or block it. If a malicious update has already affected the global model, FedSentinel can selectively remove the risky contributions and re-aggregate the trusted ones to recover the model.”

---

# 59. Final Novelty Explanation

### Simple

> **Detect → Measure Impact → Respond → Recover**

### Technical

> **Multi-signal federated update detection + historical client reputation + impact-aware risk assessment + selective recovery through trusted re-aggregation.**

### Important distinction

The novelty is NOT claimed to be:

> “We invented anomaly detection for federated learning.”

The project instead combines the detection layer with an explicit impact-assessment and recovery loop:

```text
Suspicious?
    ↓
How dangerous?
    ↓
What should we do?
    ↓
Did damage occur?
    ↓
Can we recover?
```

---

# 60. Final Acceptance Test

The project is considered successfully integrated only if the following can run from a clean environment:

```text
Start Dashboard
      ↓
Create Simulation
      ↓
Create FL Clients
      ↓
Train Locally
      ↓
Generate Updates
      ↓
Activate Attack
      ↓
FedSentinel Detection
      ↓
Threat Score
      ↓
Impact Score
      ↓
Accept / Down-weight / Quarantine
      ↓
Aggregate
      ↓
Evaluate Global Model
      ↓
Detect Possible Degradation
      ↓
Trigger Selective Recovery
      ↓
Re-aggregate Trusted Updates
      ↓
Evaluate Recovered Model
      ↓
Persist Results
      ↓
Display Full Audit/Explanation
```

No manual database editing should be required for the normal demo.

---

# 61. Final Team Rule

The team is building ONE system.

```text
Person 1 = Federated Learning
Person 2 = Attacks
Person 3 = FedSentinel Detection + Impact + Recovery Intelligence
Person 4 = Backend + Frontend + Integration
```

The final product is:

```text
            FED SENTINEL
                 │
        ┌────────┴────────┐
        │                 │
    PREVENTION         RECOVERY
        │                 │
 Detect → Assess      Detect Damage
        │                 │
 Threat + Impact      Select Risky Updates
        │                 │
 Accept/Reduce/Block  Re-aggregate Trusted
        │                 │
        └────────┬────────┘
                 ↓
          HEALTHIER GLOBAL MODEL
```

**Contract Version:** 2.0  
**Status:** Proposed — freeze before implementation begins.
