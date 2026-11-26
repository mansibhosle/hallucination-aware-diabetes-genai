"""Evaluate a selected configuration and export a private review sheet."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.evaluation.final_metrics import automatic_metrics
from src.evaluation.retrieval_benchmark import parse_bool

CONFIGS={"semantic_faiss":("semantic","none"),"hybrid_rrf":("hybrid","none"),
 "hybrid_cross_encoder":("hybrid","cross_encoder"),"hierarchical_cross_encoder":("hierarchical","cross_encoder")}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--benchmark",type=Path,default=Path("outputs/retrieval_benchmark/summary.csv"))
    p.add_argument("--configuration",choices=CONFIGS)
    p.add_argument("--split",choices=["validation","test"],default="validation")
    p.add_argument("--allow-original-test",action="store_true")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.split=="test" and (not a.allow_original_test or not a.configuration):
        p.error("Freeze explicit configuration and acknowledge historical test reuse")
    bench=pd.read_csv(a.benchmark)
    if set(bench.configuration)!=set(CONFIGS) or len(bench)!=4:
        raise ValueError("Complete all four retrieval benchmark runs first")
    manifest=json.loads((a.benchmark.parent/"manifest.json").read_text())
    if manifest["split"]!="validation" or manifest["evaluation_sha256"]!=sha(a.evaluation):
        raise ValueError("Benchmark manifest does not match evaluation dataset")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError("Choose a new output directory")
    frame=pd.read_csv(a.evaluation,dtype=str,keep_default_na=False)
    frame=frame.loc[frame["split"].str.strip().str.lower().eq(a.split)]
    if frame.empty or frame.question_id.duplicated().any():
        raise ValueError("Missing or duplicate evaluation rows")
    for val in frame.should_abstain: parse_bool(val)
    selected=a.configuration or bench.sort_values(["mrr_at_5","hit_at_5","mean_latency_ms"],ascending=[False,False,True]).iloc[0].configuration
    mode,rerank=CONFIGS[selected]
    os.environ.update({"FAISS_SEARCH_MODE":mode,"RERANK_BACKEND":rerank,
        "TOP_K_CANDIDATES":str(manifest["candidate_count"]),"TOP_K_FINAL":"5",
        "VERIFICATION_BACKEND":"nli","DB_PATH":str(a.output_dir/"interactions.sqlite")})
    from src.config import SETTINGS
    if SETTINGS.generation_backend!="phi3_peft" or SETTINGS.embedding_backend!="faiss":
        raise ValueError("Configure private Phi-3 and FAISS backends first")
    if sha(SETTINGS.faiss_index_path)!=manifest["index_sha256"] or sha(SETTINGS.faiss_metadata_path)!=manifest["metadata_sha256"]:
        raise ValueError("Index or metadata changed since benchmark")
    from src.pipeline import Platform
    a.output_dir.mkdir(parents=True,exist_ok=True)
    run={"split":a.split,"configuration":selected,"selection_rule":"explicit" if a.configuration else "provisional highest MRR@5, then Hit@5, then lowest latency",
        "evaluation_sha256":sha(a.evaluation),"index_sha256":sha(SETTINGS.faiss_index_path),
        "adapter_sha256":sha(Path(SETTINGS.model_adapter_path)/"adapter_model.safetensors"),
        "settings":SETTINGS.model_dump(mode="json"),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "n_expected":len(frame),"original_test_reused":a.split=="test"}
    (a.output_dir/"manifest.json").write_text(json.dumps(run,indent=2))
    system=Platform.load_default()
    rows=[]
    for i,case in enumerate(frame.to_dict("records"),1):
        t=time.perf_counter()
        result=system.answer(case["question"],"researcher")
        row={**case,"decision":result["decision"],"answer":result["answer"],
             "reason":result.get("reason",""),"confidence":result["confidence"],
             "latency_s":time.perf_counter()-t,
             "retrieved_evidence":json.dumps(result["evidence"],ensure_ascii=False),
             "verification_details":json.dumps(result.get("verification_details",[])),
             "factual_correct":"","evidence_supported":"","safe_under_review_rubric":"","review_notes":""}
        rows.append(row)
        pd.DataFrame(rows).to_csv(a.output_dir/"review.csv",index=False)
        print(f"{i}/{len(frame)} {case['question_id']} {result['decision']}",flush=True)
    metrics=automatic_metrics(pd.DataFrame(rows))
    (a.output_dir/"automatic_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    print("Manual review required before reporting factual accuracy or hallucination.")
if __name__=="__main__": main()
