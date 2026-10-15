"""Tests that need artifacts not distributed with this repository.

The frozen benchmark configuration, the observation bank, the E1 checkpoints and the sealed D-6 outputs are not
part of the release (see README). When they are absent, the tests that read them are skipped, not failed.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'output' / 'study' / 'gnss_qrl_v1' / 'frozen'
REQUIRES_ARTIFACTS = {
    'gnss_qrl_dq/test_dq_b2.py::test_precision_is_computed_from_the_e1_report_of_record',
    'gnss_qrl_dq/test_dq_runtime.py::test_watched_trees_are_their_approved_trees',
    'gnss_qrl_dq/test_dq_b1.py::test_files_sealed_by_d6_are_unchanged',
    'gnss_qrl_dq/test_dq_b1.py::test_launch_and_units_refuse_while_b1_is_not_approved',
    'gnss_qrl_dq/test_dq_b1.py::test_run_unit_end_to_end_on_the_development_namespace',
    'gnss_qrl_dq/test_dq_b1.py::test_the_four_d6_decisions_are_bound_and_any_change_is_refused',
    'gnss_qrl_dq/test_dq_b2.py::test_manifest_lists_108_runs_in_triplet_order_and_binds_the_campaign',
    'gnss_qrl_dq/test_dq_b2.py::test_studies_are_registered_with_their_own_salts_namespaces_and_candidates',
    'gnss_qrl_dq/test_dq_b2.py::test_training_configuration_equals_e1s_except_study_labels_and_validates',
    'gnss_qrl_dq/test_dq_b2.py::test_twin_arm_trains_through_e1s_trainer_on_the_development_namespace',
    'gnss_qrl_dq/test_dq_batches.py::test_confirmatory_sets_are_closed',
    'gnss_qrl_dq/test_dq_d6.py::test_analysis_dry_run_end_to_end_on_synthetic_episodes',
    'gnss_qrl_dq/test_dq_d6.py::test_seal_verification_refuses_absence_self_hash_mismatch_and_drift',
    'gnss_qrl_dq/test_dq_freeze_a.py::test_manifest_lists_every_job_and_seed',
    'gnss_qrl_dq/test_dq_freeze_a.py::test_reset_changes_exactly_the_declared_tensors_to_the_runs_own_update0_values',
    'gnss_qrl_dq/test_dq_freeze_a.py::test_seeds_share_crn_across_drift_and_presence_and_suites_are_disjoint',
    'gnss_qrl_dq/test_dq_freeze_a.py::test_sixteen_cells_and_in_range_cells_reproduce_the_frozen_p3_georeference',
    'gnss_qrl_dq/test_dq_freeze_a.py::test_stress_cells_run_including_0m_and_8m_on_the_development_namespace',
    'gnss_qrl_dq/test_dq_freeze_a.py::test_the_48_finals_are_bound_by_sha256',
    'gnss_qrl_dq/test_dq_freeze_a.py::test_the_a0_scorer_reproduces_stored_never_fix_scores',
    'gnss_qrl_dq/test_dq_runtime.py::test_dq_namespaces_follow_p3_rule_after_e1_and_overlap_nothing',
    'gnss_qrl_dq/test_dq_runtime.py::test_only_dq_development_is_open_and_every_other_namespace_refuses',
    'gnss_qrl_dq/test_dq_runtime.py::test_reused_e1_modules_are_byte_identical_to_the_approved_e1_tree',
    'gnss_qrl_dq/test_dq_runtime.py::test_write_guard_refuses_every_symlinked_frozen_target',
}


def pytest_collection_modifyitems(config, items):
    if ARTIFACTS.exists():
        return
    skip = pytest.mark.skip(reason='needs frozen study artifacts that are not distributed (see README)')
    for item in items:
        if item.nodeid.split('[')[0].split('tests/')[-1] in REQUIRES_ARTIFACTS:
            item.add_marker(skip)
