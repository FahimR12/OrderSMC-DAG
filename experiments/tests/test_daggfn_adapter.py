import numpy as np
import pytest

from ordersmc_experiments.adapter_validation import adapter_fixture_table
from ordersmc_experiments.daggfn_adapter import CandidateTableScore


def test_score_protocol_preserves_likelihood_and_prior():
    table=adapter_fixture_table(); scorer=CandidateTableScore(table,2)
    before,after=scorer.get_local_scores(2,(),(0,1))
    assert before.key==(2,())
    assert after.key==(2,(0,1))
    assert after.score==table.loglik[2][-1]
    assert after.prior==table.logprior[2][-1]
    assert scorer.get_local_scores(2,(0,1))==(None,after)


def test_screened_pool_is_rejected():
    table=adapter_fixture_table()
    table.parent_sets[0]=table.parent_sets[0][:-1]
    table.loglik[0]=table.loglik[0][:-1]
    table.logprior[0]=table.logprior[0][:-1]
    with pytest.raises(ValueError,match='exhaustive'):
        CandidateTableScore(table,2)


def test_nonfinite_and_zero_indegree_are_rejected():
    table=adapter_fixture_table()
    with pytest.raises(ValueError,match='max_parents'):
        CandidateTableScore(table,0)
    table.logprior[0][0]=np.nan
    with pytest.raises(ValueError,match='non-finite'):
        CandidateTableScore(table,2)
