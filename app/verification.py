"""Isolated actual-model verification settings; never uploads synthetic test records."""
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv


def isolated_settings(root):
    root=Path(root)
    load_dotenv(root/'.env')
    model_cache=os.getenv('OCR_MODEL_DIR',str(Path(os.getenv('OCR_DATA_DIR',str(root/'data'/'runtime'))).resolve()/'models'))
    parent=root/'data'/'runtime'/'verification'
    parent.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    os.environ['OCR_MODEL_DIR']=model_cache
    os.environ['OCR_DATA_DIR']=str(directory)
    os.environ['SUPABASE_SECRET_KEY']=''
    os.environ['RAG_LLM']='none'
    return directory
