import numpy as np
import pytest
from app.engine import models

class Encoder:
    def __init__(self):self.tokenizer=self;self.calls=[]
    def encode(self,texts,**kwargs):
        if isinstance(texts,str):return list(range(len(texts.split())+2))
        self.calls.append(list(texts));vectors=np.zeros((len(texts),1024),np.float32)
        for i,text in enumerate(texts):vectors[i,len(text)%1024]=1
        return vectors


def test_cache_dedup_order_profile_and_corruption(tmp_path,monkeypatch):
    import sqlite3,sys,types,contextlib
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    monkeypatch.setenv('OCR_DATA_DIR',str(tmp_path));model=Encoder()
    monkeypatch.setattr(models,'embedding_model',lambda:model)
    profile=['one'];monkeypatch.setattr(models,'embedding_identity',lambda:profile[0])
    vectors=models.embed(['longer text here','short','short'])
    assert len(model.calls)==1 and model.calls[0]==['short','longer text here']
    assert vectors[0,16]==1 and vectors[1,5]==1
    assert np.array_equal(vectors,models.embed(['longer text here','short','short']))
    assert len(model.calls)==1
    profile[0]='two';models.embed(['short']);assert len(model.calls)==2
    with sqlite3.connect(tmp_path/'embeddings.sqlite') as db:
        db.execute("update embeddings set vector=? where profile='two'",(b'bad',))
    models.embed(['short']);assert len(model.calls)==3
    with pytest.raises(ValueError):models.embed([' '.join(['x']*512)])


def test_file_hash_invalidates_changed_file(tmp_path):
    path=tmp_path/'weights.bin';path.write_bytes(b'old')
    before=models.file_digest(path);path.write_bytes(b'new value')
    assert before!=models.file_digest(path)




def test_manifest_unique_relative_paths_and_package_identity(tmp_path,monkeypatch):
    import sys,types
    monkeypatch.setenv('OCR_DATA_DIR',str(tmp_path))
    root=tmp_path/'models'/'hf';root.mkdir(parents=True)
    paths={}
    for name,content in [('config.json','base'),('1_Pooling/config.json','pool'),('sentencepiece.bpe.model','tokens')]:
        path=root/name;path.parent.mkdir(exist_ok=True);path.write_text(content)
        paths[name]=str(path)
    hub=types.SimpleNamespace(try_to_load_from_cache=lambda model,name,**kwargs:paths.get(name))
    monkeypatch.setitem(sys.modules,'huggingface_hub',hub)
    class Model:
        def __getitem__(self,index):
            return types.SimpleNamespace(auto_model=types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash='fixed')))
    model=Model();monkeypatch.setattr(models,'_embedder',model)
    monkeypatch.setattr(models,'embedding_model',lambda:model)
    version=['1'];monkeypatch.setattr(models,'packages',lambda:{'torch':version[0]})
    manifest=models.model_manifest()
    assert set(manifest['files'])=={'hf/config.json','hf/1_Pooling/config.json','hf/sentencepiece.bpe.model'}
    assert len(set(manifest['files'].values()))==3
    before=models.embedding_identity();version[0]='2'
    assert before!=models.embedding_identity()
    before=models.embedding_identity();(root/'config.json').write_text('changed base')
    assert before!=models.embedding_identity()
