import pytest
from ordersmc_experiments.reporting_recovery import reporting_only_diff, verify


def test_reporting_amendment_rejects_sampler_or_import_changes():
    original='import math\ndef sample():\n return 1\ndef aggregate():\n return 2\n'
    reporting_only_diff(original,original.replace('return 2','return 3'),{'aggregate'})
    with pytest.raises(RuntimeError,match='outside'):
        reporting_only_diff(original,original.replace('return 1','return 3'),{'aggregate'})
    with pytest.raises(RuntimeError,match='outside'):
        reporting_only_diff(original,original.replace('import math','import random'),{'aggregate'})


def test_recovery_manifest_detects_preserved_run_modification(tmp_path):
    from ordersmc_experiments.execution import _atomic_json
    from ordersmc_experiments.readiness import canonical_sha
    from ordersmc_experiments.readiness_audits import sha
    from ordersmc_experiments.reporting_recovery import EDITION
    saved=tmp_path/'run.json';saved.write_text('original result')
    manifest=dict(root=str(tmp_path),frozen_source_files={},reporting_files={},
                  preserved_input_files={'run.json':sha(saved)})
    manifest['manifest_sha256']=canonical_sha(manifest)
    _atomic_json(tmp_path/'reporting_recovery'/EDITION/'manifest.json',manifest)
    verify(tmp_path)
    saved.write_text('changed result')
    with pytest.raises(RuntimeError,match='preserved artifact changed'):verify(tmp_path)
