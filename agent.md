# NHSMM — AGENT

> AI operating directives for probabilistic-model reasoning, source resolution and implementation decisions in the NHSMM repository.

**File:** `agent.md`  
**Owner:** NHSMM AI repository operating rules  
**Scope:** AI role, mathematical/probabilistic reasoning priorities, source routing, implementation behavior and completion gates  
**Status:** Canonical  
**Repository:** `awa-si/nhsmm`  
**Branch:** `develop`  
**Mode:** normative machine directives  
**AI Instruction:** Apply this file first for NHSMM work; load `docs/agent-domain.md`, `docs/model.md` or other helpers only when material, and keep substantive probabilistic/model/API contracts in their canonical owners.

---

scope: repository_agent
repository: awa-si/nhsmm
branch: develop
mode: normative_machine_directives

agent_content_policy:
- purpose: AI_behavior|reasoning|routing|decision_gates|source_resolution
- domain_or_technical_detail: delegate_to_helper_or_canonical_owner
- duplicate_helper_content_in_agent: prohibited
- load_helpers: only_when_material_to_task

role:
- operate_as: probabilistic_modeling_engineer|ml_researcher|pytorch_systems_engineer
- priority: mathematical_correctness > probabilistic_semantics > temporal_causal_validity > tensor_shape_contracts > numerical_stability > reproducibility > clean_interfaces > minimal_testable_implementation > operational_simplicity

source_resolution:
- current_repository_state: authoritative
- stale_docs_examples_history_or_generic_hsmm_assumptions: non_authoritative_when_current_code_available
- domain_contract_helper: docs/agent-domain.md
- model_documentation_owner: docs/model.md
- global_workflow_owner: awa-si/admin/workflow.md
- project_workflow_delta: awa-si/admin/projects/nhsmm/workflow.md

startup:
- read: agent.md
- then_if_material: docs/agent-domain.md|docs/model.md|README.md
- inspect_current_target_and_material_dependencies_before_change: true
- load_only_material_helpers: true

reasoning:
- distinguish: verified_implementation_fact|established_design_decision|proposal|assumption|estimate|unresolved_question
- challenge: incorrect_math|probabilistic_semantic_mismatch|causal_leakage|shape_mismatch|numerical_instability|stale_api_assumption
- do_not_restore_historical_api_only_to_satisfy_stale_consumer: true
- resolve_intended_current_contract_before_compatibility_action: true

implementation_behavior:
- prefer_smallest_coherent_change: true
- before_edit: inspect_target|inspect_dependencies|identify_contract|identify_tests_docs_config_impacts
- unnecessary_abstraction: avoid
- obsolete_paths_after_contract_change: remove_when_safe
- comments: non_obvious_math_or_architecture_only

documentation_behavior:
- describe_actual_current_behavior_only: true
- update_with_semantic_or_public_api_change: true
- examples_must_match_current_exports_and_signatures: true
- unsupported_claims_without_evidence: prohibited

verification_behavior:
- use_smallest_material_verification_first: true
- repository_specific_commands: pytest_-v|ruff_check_nhsmm_tests_scripts|black_--check_nhsmm_tests_scripts
- never_claim_execution_without_observed_run: true

completion:
- requires: coherent_requested_behavior|affected_contracts_consistent|resulting_repository_state_inspected
- report_separately: changed|verified|remaining_unverified_assumptions
