# P4 ↔ P2 Integration

P2 provides:
    core.attacks.attack_manager.AttackManager

P4 may:
    - configure/register attacks
    - retrieve scheduled attacks
    - execute post-hoc ground-truth evaluation
    - persist attack metadata / evaluation metrics

P3 must never receive:
    - is_malicious
    - attack_type ground truth
    - AttackManager ground-truth registry

Update-plane attack:
    attack.apply_update_attack(delta)

Data-plane attack:
    attack.apply_data_attack(dataset)

Ground truth:
    AttackManager.get_ground_truth_for_evaluation()

Ground truth is used only after/beside inference for:
    - evaluation
    - confusion matrix
    - ROC/PR metrics
    - dashboard evaluation