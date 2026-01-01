# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('D:\\QAGenPipelinesX_lightrag\\qa_gen_pipelines\\src', 'src'), ('D:\\QAGenPipelinesX_lightrag\\qa_gen_pipelines\\config_local.yaml', '.')]
datas += collect_data_files('tiktoken')


a = Analysis(
    ['D:\\QAGenPipelinesX_lightrag\\qa_gen_pipelines\\main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['lightrag', 'lightrag.utils', 'lightrag.llm', 'lightrag.storage', 'lightrag.operate', 'lightrag.base', 'lightrag.kg', 'lightrag.kg.json_kv_impl', 'lightrag.kg.neo4j_impl', 'lightrag.kg.networkx_impl', 'lightrag.kg.nano_vector_db_impl', 'lightrag.kg.age_impl', 'lightrag.kg.chroma_impl', 'lightrag.kg.faiss_impl', 'lightrag.kg.gremlin_impl', 'lightrag.kg.json_doc_status_impl', 'lightrag.kg.milvus_impl', 'lightrag.kg.mongo_impl', 'lightrag.kg.postgres_impl', 'lightrag.kg.qdrant_impl', 'lightrag.kg.redis_impl', 'lightrag.kg.shared_storage', 'lightrag.kg.tidb_impl', 'lightrag.graph', 'lightrag.memory', 'lightrag.retrieve', 'openai', 'requests', 'loguru', 'numpy', 'pandas', 'networkx', 'networkx.algorithms', 'networkx.algorithms.community', 'graspologic', 'tiktoken', 'tiktoken.registry', 'tiktoken_ext', 'tiktoken_ext.openai_public', 'nano_vectordb', 'nest_asyncio', 'jinja2', 'markdown', 'jsonlines'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='qa_gen_pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
