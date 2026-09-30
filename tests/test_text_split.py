from supertonicsynth.text_split import chunk_text


def test_chunk_text():
    chunks = chunk_text("First sentence. Second sentence. Third sentence.", 30)
    assert len(chunks) >= 2
    assert all(chunks)
