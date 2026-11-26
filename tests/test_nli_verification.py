import numpy as np
from src.schemas import Evidence
from src.verification.nli import classify_pairs, sentences

def test_nli_support_and_neutral():
    ev=[Evidence(chunk_id="a",text="A",source_id="demo")]
    result=classify_pairs(["First claim.","Second claim."],ev,np.array([[[.02,.95,.03]],[[.05,.10,.85]]]))
    assert result["support"]==.5
    assert result["unsupported_claims"]==["Second claim."]
    assert result["details"][1]["status"]=="insufficient"

def test_nli_conflicting_evidence_blocks():
    ev=[Evidence(chunk_id="a",text="A",source_id="demo"),Evidence(chunk_id="b",text="B",source_id="demo")]
    result=classify_pairs(["Claim."],ev,np.array([[[.02,.96,.02],[.92,.03,.05]]]))
    assert result["support"]==0
    assert result["details"][0]["status"]=="contradicted"

def test_short_claim_and_decimal_not_discarded():
    assert sentences("Value is 5.7. Not safe. [S1]")==["Value is 5.7.","Not safe."]
