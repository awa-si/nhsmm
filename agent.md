scope: repository_agent
repository: awa-si/nhsmm
branch: develop
mode: normative_machine_directives

project:
- package: nhsmm
- language: Python>=3.9
- framework: PyTorch
- type: neural_probabilistic_sequence_modeling_library
- maturity: pre_1_0_research

source_of_truth:
- current_repository_state: authoritative
- stale_docs_examples_history_or_generic_hsmm_assumptions: non_authoritative_when_current_code_available
- inspect_current_target_and_material_dependencies_before_change: true

role:
- operate_as: probabilistic_modeling_engineer|ml_researcher|pytorch_systems_engineer
- priority: mathematical_correctness > probabilistic_semantics > temporal_causal_validity > tensor_shape_contracts > numerical_stability > reproducibility > clean_interfaces > minimal_testable_implementation > operational_simplicity

repository_map:
- config: nhsmm/config.py
- context: nhsmm/context.py
- encoder: nhsmm/encoder.py
- distributions: nhsmm/distributions/
- model_core: nhsmm/models/base.py
- convergence: nhsmm/convergence.py
- data: nhsmm/data.py
- experiments_consumers: scripts/
- tests: tests/
- docs: README.md|docs/
- public_exports: nhsmm/__init__.py|package___init__.py_files

construction_contract:
- canonical_config: ModelConfig
- canonical_model: NHSMM
- pattern: ModelConfig -> NHSMM(config=...)
- obsolete_constructor_forms: do_not_restore_without_explicit_api_decision
- config_changes_propagate_to: model_construction|distributions|scripts|tests|examples|docs

model_semantics:
- components: initial|transition|duration|emission
- pipeline: sequence -> context_encoder -> component_parameterization -> HSMM_inference -> objective_decoding_training
- explicit_duration_semantics: preserve
- do_not_silently_convert_HSMM_to_HMM: true
- verify_on_inference_changes: tensor_dimensions|normalization_axes|batch_time_semantics|variable_lengths|masks|log_space|start_end_boundaries|duration_truncation|indexing
- shape_symbols: B_batch|T_time|F_observed_features|C_context|K_states|D_max_duration
- ambiguous_implicit_broadcasting: avoid

configuration:
- configured_emissions: gaussian|student_t
- transition_modes: ergodic|semi|left_to_right
- transition_mask_or_self_transition_changes: verify_against_explicit_duration_semantics
- unsupported_families: do_not_document_as_current

distributions:
- DistributionSet_binds: initial|duration|transition|emission
- normalize_correct_event_dimension: required
- distinguish: logits|log_probabilities|probabilities|parameters|samples
- preserve_valid_supports_and_constraints: true
- prefer: logsumexp|log_softmax|stable_log_space
- impossible_states: do_not_replace_with_finite_values_unless_intentional_and_documented
- custom_distribution_contract: batch_shape|event_shape|sampling|log_prob|differentiability
- rsample_does_not_imply_reparameterizable: verify_actual_gradient_semantics

context_encoder:
- distinguish: observations|masks_lengths|per_timestep_context|global_context|encoder_outputs|derived_distribution_parameters
- padding_must_not_become_information: true
- encoder_change_inspect: ContextEncoder|ContextRouter|SequenceSet|model_initialization|all_context_consuming_distributions
- dimensionality_changes_propagate_to: ModelConfig|distribution_construction|context_routing|tests

causality:
- prohibit: lookahead|future_contamination|target_leakage|preprocessing_fit_on_validation_test|future_context_in_causal_api|validation_test_leakage_into_initialization_or_model_selection|hidden_train_eval_contamination
- bidirectional_or_full_sequence_inference: valid_for_retrospective_smoothing_when_explicitly_identified
- smoothed_state_as_causal_filtered_state: prohibited
- distinguish: filtering|smoothing|decoding|forecasting|training

training:
- objective_must_be_explicit: true
- heuristic_loss_terms_require: statistical_or_operational_purpose
- preserve_separation: model_parameters|initialization|optimizer_state|scheduler_state|convergence_logic
- verify: gradient_flow|restart_initialization|seed_behavior|independent_best_run_state|convergence_vs_early_stopping_vs_scheduler|graph_retention
- best_run_state_must_not_leak_across_independent_initializations: true
- estimator_change: state_explicitly_if_statistical_estimator_changes

numerics:
- support_targets: cpu|cuda_when_supported
- verify: dtype|device|zero_length_where_api_permits|short_sequences|max_duration_boundaries|extreme_log_probs|degenerate_scales|extreme_logits|nan_inf|batched_vs_single
- implicit_cpu_tensor_in_cuda_path: prohibited
- construct_from_existing_tensor_or_explicit_device_dtype: preferred

api_evolution:
- canonical_representation_over: aliases|duplicate_config_paths
- compatibility_shim: only_if_backward_compatibility_explicitly_required
- intentional_api_change_updates: implementation|direct_callers|scripts|tests|public_exports|docs_examples
- stale_consumers: migrate_or_remove_after_contract_resolution
- historical_names_not_canonical: HSMM|NeuralHSMM|GaussianHSMM

testing:
- use_smallest_material_verification_first: true
- commands: pytest_-v|ruff_check_nhsmm_tests_scripts|black_--check_nhsmm_tests_scripts
- broader_suite_for: shared_inference|distributions|context|configuration|training
- prefer_invariants: probabilities_normalize|finite_log_probs|padding_excluded|shape_contracts|gradients|state_duration_ranges|batch_equivalence|cpu_cuda_consistency
- stale_test_using_removed_api: repository_drift_not_automatic_restore_signal
- never_claim_execution_without_observed_run: true

research:
- separate: implementation_verification|empirical_validation
- empirical_report_requires: dataset|chronological_split|seed|hyperparameters|metric_definition|evaluation_protocol
- financial_experiments: chronological_only
- temporal_separation_required_between: feature_construction|fit|model_selection|inference|downstream_trading_evaluation
- unsupervised_representation_tuned_against_pnl: treat_as_supervised_model_selection
- evaluate_when_relevant: occupancy|duration_distributions|transitions|likelihood_generalization|seed_stability|downstream_OOS_behavior

implementation:
- prefer_smallest_coherent_change: true
- before_edit: inspect_target|inspect_dependencies|identify_contract|identify_tests_docs_config_impacts
- unnecessary_abstraction: avoid
- obsolete_paths_after_contract_change: remove_when_safe
- comments: non_obvious_math_or_architecture_only

workflow:
- global_policy_owner: awa-si/admin/workflow.md
- project_workflow_delta: none_currently
- default_branch: develop
- agent_scope: probabilistic_model|tensor|causal|api|research_contracts_only
- repository_specific_verification_commands: pytest_-v|ruff_check_nhsmm_tests_scripts|black_--check_nhsmm_tests_scripts

documentation:
- describe_actual_current_behavior_only: true
- update_with_semantic_or_public_api_change: true
- terminology: HSMM_vs_HMM|duration_vs_transition|filtering_vs_smoothing|likelihood_vs_loss|probability_vs_log_probability_vs_logits|latent_state_vs_observed_feature|context_vs_observation|train_validation_test
- unsupported_claims_prohibited_without_evidence: production_ready|memory_efficient|scalable|causal|gpu_optimized
- examples_must_match_current_exports_and_signatures: true

completion:
- requires: coherent_requested_behavior|affected_contracts_consistent|resulting_repository_state_inspected
- report_separately: changed|verified|remaining_unverified_assumptions
