"""Local semantic retrieval using FastEmbed/ONNX; no TF-IDF fallback."""
from __future__ import annotations
import json, os, threading, time
from typing import Dict, List, Optional
import config
_STATUS_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","embedding_status.json")
_RETRY_COOLDOWN_SECONDS=300

def _load_status()->Dict:
    try:
        with open(_STATUS_PATH,"r",encoding="utf-8") as f:return json.load(f)
    except Exception:return {}

def _save_status(status:Dict):
    try:
        os.makedirs(os.path.dirname(_STATUS_PATH),exist_ok=True)
        with open(_STATUS_PATH,"w",encoding="utf-8") as f:json.dump(status,f)
    except Exception:pass

class SemanticIndex:
    """Semantic capability index. Dependency failures are recoverable."""
    def __init__(self,groups:Dict[str,"ActionGroup"]):
        self._groups=groups; self._model=None; self._vectors=None
        self.ready=False; self.loading=False; self.unavailable_reason=None
        status=_load_status(); failed=float(status.get("last_failed_at",0) or 0); reason=str(status.get("reason","") or "")
        if failed and reason and not reason.startswith("fastembed not installed") and time.time()-failed<_RETRY_COOLDOWN_SECONDS:
            self.unavailable_reason=f"retry cooldown after: {reason}"; return
        self.loading=True
        threading.Thread(target=self._load,daemon=True,name="prixon-embeddings").start()
    def _load(self):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            self.loading=False; self.unavailable_reason="fastembed not installed (pip install fastembed)"; _save_status({"reason":self.unavailable_reason}); return
        try:
            model=TextEmbedding(model_name=config.EMBEDDING_MODEL)
            targets=list(self._groups)
            texts=[f"{self._groups[t].target_name}. {' '.join(self._groups[t].example_phrasings[:8])}. intent: {self._groups[t].intent}. action: {self._groups[t].action}" for t in targets]
            vectors=list(model.embed(texts))
            self._model=model; self._vectors=dict(zip(targets,vectors)); self.ready=True; self.loading=False; self.unavailable_reason=None
            _save_status({"last_success_at":time.time(),"model":config.EMBEDDING_MODEL})
        except Exception as exc:
            self.loading=False; self.unavailable_reason=f"semantic model load failed: {exc}"; _save_status({"last_failed_at":time.time(),"reason":self.unavailable_reason})
    def search(self,query:str,top_k:int=8)->List[Dict]:
        if not self.ready or self._model is None or not self._vectors:return []
        try:
            import numpy as np
            q=list(self._model.embed([query]))[0]; qn=float(np.linalg.norm(q))+1e-9; scored=[]
            for target,v in self._vectors.items():
                s=float(np.dot(q,v)/(qn*(float(np.linalg.norm(v))+1e-9))); scored.append((s,target))
            scored.sort(reverse=True)
            return [{"target":t,"score":round(s,4)} for s,t in scored[:top_k]]
        except Exception as exc:
            self.unavailable_reason=f"semantic search failed: {exc}"; return []
