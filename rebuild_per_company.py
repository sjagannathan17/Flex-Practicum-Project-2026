#!/usr/bin/env python3
"""Rebuild ChromaDB one company at a time in separate subprocesses to manage memory."""
import subprocess
import sys
from pathlib import Path

COMPANIES = ["Flex", "Jabil", "Celestica", "Benchmark", "Sanmina"]

SCRIPT = '''
import sys, os, re, gc
from pathlib import Path
from collections import defaultdict

os.chdir("{project_root}")
sys.path.insert(0, ".")

# Import everything from build_chromadb except __main__
import importlib.util
spec = importlib.util.spec_from_file_location("build", "Vector Database/build_chromadb.py")
mod = importlib.util.module_from_spec(spec)

# Patch __file__ before exec
mod.__file__ = str(Path("Vector Database/build_chromadb.py").resolve())
spec.loader.exec_module(mod)

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=mod.DB_PATH)
collection = client.get_or_create_collection(name="capex_docs", metadata={{"hnsw:space": "cosine"}})
print(f"  DB: {{collection.count()}} existing docs")

model = SentenceTransformer("all-mpnet-base-v2")
print("  Model loaded")

all_files = mod.discover_all_files()
company_files = [(fp, co, ft) for fp, co, ft in all_files if co == "{company}"]
print(f"  Files for {company}: {{len(company_files)}}")

total = 0
for filepath, company, filing_type in company_files:
    print(f"  {{filepath.name[:62]:<64}}", end="", flush=True)
    
    text = mod.extract(filepath)
    if not text or len(text.strip()) < 100:
        print("skip")
        gc.collect()
        continue
    
    if not filing_type or filing_type == "Other":
        filing_type = mod.detect_filing_type(filepath, text)
    fy, q = mod.get_fiscal_quarter(filepath, company, text)
    
    unit_header = ""
    um = re.search(r"\\((?:in|amounts?\\s+in)\\s+(thousands|millions|billions)\\)", text[:5000], re.IGNORECASE)
    if um:
        unit_header = um.group(1).lower()
    
    structured_chunks = mod.chunk_text_structured(text)
    if not structured_chunks:
        print("no chunks")
        gc.collect()
        continue
    
    safe_stem = re.sub(r"[^a-zA-Z0-9_\\-]", "_", filepath.stem)
    parent_id = f"{{company}}_{{safe_stem}}"
    
    ids, texts, metadatas = [], [], []
    for i, (section_name, chunk) in enumerate(structured_chunks):
        doc_id = f"{{company}}_{{safe_stem}}_chunk{{i:04d}}"
        table_type = mod._detect_table_type(chunk)
        if not section_name:
            section_name = mod._detect_section(chunk)
        ids.append(doc_id)
        texts.append(chunk)
        metadatas.append({{
            "company": company,
            "source_file": filepath.name,
            "filing_type": filing_type,
            "fiscal_year": fy,
            "quarter": q,
            "chunk_index": i,
            "total_chunks": len(structured_chunks),
            "section_header": section_name,
            "table_type": table_type,
            "table_context": f"in {{unit_header}}" if unit_header else "",
            "parent_id": parent_id,
        }})
    
    for start in range(0, len(texts), 64):
        emb = model.encode(texts[start:start+64], show_progress_bar=False)
        collection.upsert(
            ids=ids[start:start+64],
            embeddings=emb.tolist(),
            documents=texts[start:start+64],
            metadatas=metadatas[start:start+64],
        )
    
    total += len(structured_chunks)
    print(f"{{len(structured_chunks)}} chunks")
    
    del text, structured_chunks, ids, texts, metadatas
    gc.collect()

print(f"\\n  {company} complete: {{total}} chunks added")
'''

if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parent)
    
    for company in COMPANIES:
        print(f"\n{'='*60}")
        print(f"  PROCESSING: {company}")
        print(f"{'='*60}", flush=True)
        
        script = SCRIPT.format(company=company, project_root=project_root)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            timeout=900,
        )
        
        if result.returncode != 0:
            if result.returncode == -9:
                print(f"  {company} was KILLED (out of memory). Continuing with next...")
            else:
                print(f"  {company} FAILED with code {result.returncode}")
        else:
            print(f"  {company} SUCCESS")
    
    print(f"\n{'='*60}")
    print("  ALL COMPANIES PROCESSED")
    print(f"{'='*60}")
