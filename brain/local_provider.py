"""Local Ollama provider with live model checks and JSON-safe NLU calls."""
from __future__ import annotations
import time
from typing import Optional
import config
from brain.provider_base import LLMProvider

class OllamaProvider(LLMProvider):
    name="local"
    def __init__(self, base_url:Optional[str]=None, model:Optional[str]=None):
        self.base_url=(base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model=model or config.OLLAMA_MODEL
        self._checked_available=None
        self._checked_at=0.0
        self._availability_ttl=5.0
    def available(self)->bool:
        now=time.monotonic()
        if self._checked_available is not None and now-self._checked_at<self._availability_ttl:
            return self._checked_available
        try:
            import requests
            r=requests.get(f"{self.base_url}/api/tags",timeout=1.5)
            names={str(m.get("name","")) for m in r.json().get("models",[])} if r.status_code==200 else set()
            self._checked_available=self.model in names or any(n.split(":",1)[0]==self.model.split(":",1)[0] for n in names)
        except Exception:
            self._checked_available=False
        self._checked_at=now
        return self._checked_available
    def chat(self,system_prompt:str,user_message:str,*,max_tokens:int,temperature:float)->str:
        import requests
        num_ctx=getattr(config,"OLLAMA_NUM_CTX",4096)
        payload={"model":self.model,"messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_message}],"stream":False,"keep_alive":"10m","options":{"temperature":temperature,"num_predict":max_tokens,"num_ctx":num_ctx}}
        # NLU prompts are JSON-constrained by their caller; asking Ollama for
        # JSON reduces markdown/explanation leakage without changing general chat.
        if "OUTPUT ONLY JSON" in system_prompt:
            payload["format"]="json"
        prompt_chars=len(system_prompt)+len(user_message)
        est_input_tokens=prompt_chars//4  # rough chars/4 heuristic, no tokenizer available here
        request_start=time.time()
        if config.DEBUG:
            print(f"[OLLAMA] model={self.model} prompt_chars={prompt_chars} "
                  f"estimated_input_tokens={est_input_tokens} max_output_tokens={max_tokens} "
                  f"num_ctx={num_ctx} request_start={request_start:.3f}", flush=True)
            if est_input_tokens > num_ctx * 0.8:
                print(f"[OLLAMA] WARNING estimated_input_tokens={est_input_tokens} is close to or over "
                      f"num_ctx={num_ctx} -- prompt will be slow to process and/or silently truncated", flush=True)
        try:
            r=requests.post(f"{self.base_url}/api/chat",json=payload,timeout=config.OLLAMA_TIMEOUT_SECONDS)
        finally:
            if config.DEBUG:
                print(f"[OLLAMA] response_time={time.time()-request_start:.3f}s", flush=True)
        if r.status_code>=400: raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
        text=((r.json().get("message") or {}).get("content") or "").strip()
        if not text: raise RuntimeError("Ollama returned an empty model response")
        return text
    def classify_error(self,exc:Exception)->str:
        msg=str(exc).lower()
        if any(x in msg for x in ("connection","timeout","timed out","connectex","refused")): return "unsupported"
        if "not found" in msg or ("model" in msg and "pull" in msg): return "unsupported"
        return super().classify_error(exc)
