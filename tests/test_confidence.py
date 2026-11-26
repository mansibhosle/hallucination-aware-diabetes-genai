from src.confidence.scoring import score, threshold_for

def test_weighted_score():
    assert round(score(.8,1,.7),3)==.805

def test_risk_threshold():
    assert threshold_for("pregnancy") > threshold_for("general_education")
