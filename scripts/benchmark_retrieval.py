"""Compare four retrieval configurations on validation only, without Phi-3."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import Settings
from src.evaluation.retrieval_benchmark import load_validation, source_metrics, aggregate
from src.retrieval.faiss_retriever import FaissRetriever
from src.retrieval.reranker import Reranker

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evaluation",type=Path,required=True)
    parser.add_argument("--candidates",type=int,default=100)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/retrieval_benchmark"))
    args=parser.parse_args()
    if args.candidates<5:
        parser.error("--candidates must be at least 5")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    s=Settings()
    metadata=json.loads(s.faiss_metadata_path.read_text(encoding="utf-8"))
    known={str(r["source_id"]) for r in metadata}
    cases,excluded,total=load_validation(args.evaluation,known)
    print("Validation rows:",total,"| Scored:",len(cases),"| Excluded:",len(excluded),flush=True)
    print("Exclusions:",json.dumps(excluded,indent=2),flush=True)
    (args.output_dir/"exclusions.json").write_text(json.dumps(excluded,indent=2),encoding="utf-8")
    if not cases:
        raise SystemExit("No validation rows with mapped reference source IDs. Inspect exclusions.json.")
    # Fail rather than silently dropping validation rows with incorrect labels.
    if any(r["reason"]!="should_abstain" for r in excluded):
        raise SystemExit("Unmapped reference IDs detected. Inspect exclusions.json and resolve labels before benchmarking.")
    start=time.perf_counter()
    rr=Reranker("cross_encoder",s.rerank_model,s.rerank_device)
    reranker_init=time.perf_counter()-start
    configurations=[
        ("semantic_faiss","semantic",False),
        ("hybrid_rrf","hybrid",False),
        ("hybrid_cross_encoder","hybrid",True),
        ("hierarchical_cross_encoder","hierarchical",True),
    ]
    summary=[]; all_rows=[]
    for name,mode,rerank in configurations:
        print("\nStarting:",name,flush=True)
        start=time.perf_counter()
        retriever=FaissRetriever(s.faiss_index_path,s.faiss_metadata_path,s.faiss_build_info_path,
            s.guideline_chunks_path,s.embedding_model,s.embedding_device,mode)
        build_time=time.perf_counter()-start
        # Warm up using the first validation question; do not count this call.
        warm=retriever.search(cases[0]["question"],k=args.candidates)
        if rerank:
            rr.rerank(cases[0]["question"],warm,k=5)
        rows=[]
        for case in cases:
            q=case["question"]
            start=time.perf_counter()
            candidates=retriever.search(q,k=args.candidates)
            selected=rr.rerank(q,candidates,k=5) if rerank else candidates[:5]
            elapsed=(time.perf_counter()-start)*1000
            refs=case["reference_source_ids"]
            metrics=source_metrics(refs,[e.source_id for e in selected])
            row={"configuration":name,"question_id":case["question_id"],"category":case["category"],
                 "reference_source_ids":json.dumps(sorted(refs)),
                 "top5_source_ids":json.dumps([e.source_id for e in selected]),
                 "top5_chunk_ids":json.dumps([e.chunk_id for e in selected]),
                 "candidate_source_hit":float(any(e.source_id in refs for e in candidates)),
                 "latency_ms":elapsed,**metrics}
            rows.append(row)
        report={"configuration":name,"candidate_count":args.candidates,
                "index_init_seconds":build_time,**aggregate(rows)}
        summary.append(report); all_rows.extend(rows)
        print(json.dumps(report,indent=2),flush=True)
        pd.DataFrame(summary).to_csv(args.output_dir/"summary.csv",index=False)
        pd.DataFrame(all_rows).to_csv(args.output_dir/"per_question.csv",index=False)
        del retriever
    manifest={"split":"validation","validation_count":total,"scored_count":len(cases),
              "candidate_count":args.candidates,"final_k":5,"reranker_init_seconds":reranker_init,
              "embedding_model":s.embedding_model,"reranker_model":s.rerank_model,
              "embedding_device":s.embedding_device,"rerank_device":s.rerank_device,
              "evaluation_sha256":hashlib.sha256(args.evaluation.read_bytes()).hexdigest(),
              "index_sha256":hashlib.sha256(s.faiss_index_path.read_bytes()).hexdigest(),
              "metadata_sha256":hashlib.sha256(s.faiss_metadata_path.read_bytes()).hexdigest()}
    try:
        manifest["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        manifest["git_commit"]="unavailable"
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\nFINAL VALIDATION RESULTS (source-level; rates 0–1)")
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nSaved locally:",args.output_dir)

if __name__=="__main__":
    main()
