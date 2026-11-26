"""Retrieval-only comparison; no generator or NLI model is loaded."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import Settings
from src.retrieval.faiss_retriever import FaissRetriever
from src.retrieval.reranker import Reranker

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--question",required=True)
    parser.add_argument("--target-chunk",default="")
    parser.add_argument("--candidates",type=int,default=100)
    parser.add_argument("--mode",choices=["semantic","hybrid","hierarchical"],default="hierarchical")
    args=parser.parse_args()
    s=Settings()
    r=FaissRetriever(s.faiss_index_path,s.faiss_metadata_path,s.faiss_build_info_path,
                     s.guideline_chunks_path,s.embedding_model,s.embedding_device,args.mode)
    if r.child_search is not None:
        print("Child-window vectors:",r.child_search.index.ntotal)
    rr=Reranker("cross_encoder",s.rerank_model,s.rerank_device)
    candidates=r.search(args.question,args.candidates)
    ranked=rr.rerank(args.question,candidates,len(candidates))
    print("Mode:",args.mode)
    if args.target_chunk:
        print("Target candidate rank:",next((i for i,e in enumerate(candidates,1) if e.chunk_id==args.target_chunk),None))
        print("Target reranker rank:",next((i for i,e in enumerate(ranked,1) if e.chunk_id==args.target_chunk),None))
    for i,e in enumerate(ranked[:5],1):
        print(json.dumps({"rank":i,"chunk_id":e.chunk_id,"title":e.title,
            "retrieval_cosine":round(e.score,4),"parent_cosine":e.metadata.get("parent_cosine"),
            "reranker_score":round(e.rerank_score,4),
            "matched_window":e.metadata.get("matched_window",e.text[:500])},indent=2))
if __name__=="__main__":
    main()
