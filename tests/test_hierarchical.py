from src.retrieval.hierarchical import sentence_windows, unique_parent_ranking

def test_windows_include_sentence_and_context():
    windows=list(sentence_windows("First sentence. Second sentence. Third sentence."))
    assert "Second sentence." in windows
    assert "First sentence. Second sentence." in windows
    assert "Second sentence. Third sentence." in windows

def test_long_windows_are_bounded_and_cover_text():
    words=[str(i) for i in range(190)]
    windows=list(sentence_windows(" ".join(words)))
    assert all(len(w.split())<=80 for w in windows)
    assert set(" ".join(windows).split())==set(words)

def test_parent_ranking_deduplicates():
    assert unique_parent_ranking([2,3,0,1,-1],[5,5,9,9],10)==[9,5]
    assert unique_parent_ranking([2,0,1],[5,5,9],1)==[9]
