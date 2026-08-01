from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_chunking_splits_long_text():
    splitter=RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)
    long_text="word"*300
    chunks=splitter.split_text(long_text)
    assert len(chunks)>1

def test_chunking_short_text_stays_one_chunk():
    splitter=RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)
    short_text="A short text"
    chunks=splitter.split_text(short_text)
    assert len(chunks)==1

def test_chunk_size_respected():
    splitter=RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)
    long_text="word"*300
    chunks=splitter.split_text(long_text)
    for chunk in chunks:
        assert len(chunk)<=550