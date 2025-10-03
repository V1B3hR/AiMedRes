#!/usr/bin/env python3
"""
Ultra-Lite Multilingual Medical/Scientific Reasoning Agent (Pure Python, No NumPy)
=================================================================================

New Enhancements (this revision):
1. Granular PHI Categories & Toggles:
   - Independently enable/disable masking of: email, date, phone, id, address, name
   - Pass configuration at MemoryStore construction (phi_categories). Example: keep dates but mask names/emails.

2. Domain-Specific Lexicon Whitelisting:
   - Reduces over-masking by excluding clinical/scientific terms from naive name redaction.
   - Configurable via domain_whitelist (set[str]).

3. Configurable Embedding Dimension & Weighting Schemes:
   - EmbeddingConfig(dim=..., weighting=...) with weighting in {'raw','tf','log_norm','tfidf'}.
   - Global document frequency tracking for TF-IDF (pure Python; approximate, updated on ingestion).
   - Deterministic hashed bag-of-token model remains; normalization optional (always L2 here).

4. Persistent Session & Memory Storage (JSON):
   - MemoryStore.save_state(path) and MemoryStore.load_state(path).
   - Includes vectors (optional) and DF statistics for consistent TF-IDF continuation.
   - Safe to rehydrate into a running agent. (Dimension mismatch prevented.)

5. Adversarial / Rare Token Filtering:
   - Detects suspicious rare tokens (lengthy alphanumeric mixes, very long tokens, or doc freq==1)
   - Replaces them with [RARE] marker prior to embedding.
   - Configurable thresholds (min_doc_freq, length triggers, pattern heuristics).
   - Helps mitigate prompt injection with anomalous artifacts.

DISCLAIMER:
 - Heuristic educational artifact; NOT medical advice.
 - PHI and adversarial filters are simplistic, may over/under redact.
 - For production: integrate robust compliance, security, and privacy controls.

Run (demo):
    python live_reasoning.py
"""

from __future__ import annotations
import re
import math
import time
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Set, Iterable
from collections import Counter

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PureLiteReasoningAgent")

# --------------------------------------------------------------------------------------
# Default Configuration
# --------------------------------------------------------------------------------------
DEFAULT_EMBED_DIM = 384
SIMILARITY_THRESHOLD = 0.55
MAX_MEMORY = 5000
SUPPORTED_LANGS = {"en","es","fr","de","pt","it","zh","ja"}
DEFAULT_LANG = "en"

# --------------------------------------------------------------------------------------
# PHI Sanitization (Granular)
# --------------------------------------------------------------------------------------
# Each category -> (pattern, replacement)
PHI_CATEGORY_PATTERNS: Dict[str, Tuple[re.Pattern, str]] = {
    "email":   (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    "date_iso":(re.compile(r'\b\d{4}-\d{2}-\d{2}\b'), '[DATE]'),
    "date_alt":(re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), '[DATE]'),
    "phone":   (re.compile(r'\b(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b'), '[PHONE]'),
    "id":      (re.compile(r'\b(MRN|PatientID|PatID|ID)[:#]?\s*\d{3,}\b', re.IGNORECASE), '[ID]'),
    "address": (re.compile(r'\b\d{1,5}\s+[A-Z][a-zA-Z]+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Blvd|Boulevard)\b'), '[ADDRESS]'),
    # Name pattern handled separately for whitelist logic.
}

# Default set of categories to mask (names treated specially)
DEFAULT_PHI_CATEGORIES = {"email","date_iso","date_alt","phone","id","address","name"}

# Domain-specific lexicon to reduce false positive name masking
DEFAULT_DOMAIN_WHITELIST: Set[str] = {
    # clinical & biomedical terms (capitalized in text often)
    "Cognitive","Memory","Score","Patient","Lifestyle","APOE4","APOE","Amyloid","Tau","CSF",
    "Longitudinal","Assessment","Symptoms","Differential","Decline","Trajectory","Biomarker",
    "Imaging","Diagnostic","Risk","Stratification","Guideline","Therapeutic","Temporal","Prognostic",
    "Uncertainty","Causal","Counterfactual","Genetic","Counseling","History","Family"
}

NAME_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b')

def sanitize_phi(text: str,
                 categories: Optional[Set[str]] = None,
                 whitelist: Optional[Set[str]] = None) -> str:
    """
    Granular PHI sanitization. If categories is None, uses DEFAULT_PHI_CATEGORIES.
    'name' category uses whitelist to avoid over-masking.
    """
    if not text:
        return text
    cats = categories or DEFAULT_PHI_CATEGORIES
    wl = whitelist or set()

    # Apply category patterns except name
    for cat, (pat, repl) in PHI_CATEGORY_PATTERNS.items():
        if cat in cats:
            text = pat.sub(repl, text)

    if "name" in cats:
        # Custom name redaction with whitelist filter
        def name_repl(match: re.Match) -> str:
            phrase = match.group(0)
            tokens = phrase.split()
            # If any token in whitelist, skip masking
            for t in tokens:
                if t in wl:
                    return phrase
            # Additional heuristics: if all tokens length > 2 and capitalized -> redact
            if all(t[0].isupper() and t[1:].islower() for t in tokens):
                return "[NAME]"
            return phrase
        text = NAME_PATTERN.sub(name_repl, text)
    return text

# --------------------------------------------------------------------------------------
# Tokenization & Embeddings (Configurable)
# --------------------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9µαβγΩμ%\-]+")

def stable_hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

@dataclass
class EmbeddingConfig:
    dim: int = DEFAULT_EMBED_DIM
    weighting: str = "tfidf"  # 'raw','tf','log_norm','tfidf'
    normalize: bool = True

class EmbeddingManager:
    """
    Tracks document frequencies for tokens to enable TF-IDF weighting.
    """
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.doc_count: int = 0
        self.df: Counter = Counter()

    def update_document_frequencies(self, tokens: Iterable[str]):
        unique = set(tokens)
        for t in unique:
            self.df[t] += 1
        self.doc_count += 1

    def text_to_vector(self, text: str) -> List[float]:
        dim = self.config.dim
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return [0.0] * dim
        tf_counter = Counter(tokens)
        max_tf = max(tf_counter.values())
        vec = [0.0] * dim

        for tok, tf in tf_counter.items():
            idx = stable_hash(tok) % dim
            weight = self._token_weight(tok, tf, max_tf)
            vec[idx] += weight

        if self.config.normalize:
            norm_sq = sum(v*v for v in vec)
            if norm_sq > 0:
                inv = 1.0 / math.sqrt(norm_sq)
                vec = [v*inv for v in vec]
        return vec

    def _token_weight(self, token: str, tf: int, max_tf: int) -> float:
        w = self.config.weighting
        if w == "raw":
            return float(tf)
        if w == "tf":
            return tf / max_tf
        if w == "log_norm":
            return 1.0 + math.log(tf)
        if w == "tfidf":
            # Smooth TF
            tf_part = 0.5 + 0.5 * tf / max_tf
            df = self.df.get(token, 0)
            # Smooth IDF
            idf = math.log((1 + self.doc_count) / (1 + df)) + 1.0
            return tf_part * idf
        # default fallback
        return float(tf)

# --------------------------------------------------------------------------------------
# Adversarial / Rare Token Filtering
# --------------------------------------------------------------------------------------
ADVERSARIAL_TOKEN_PATTERN = re.compile(r'(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}')

@dataclass
class AdversarialFilterConfig:
    enable: bool = True
    min_doc_freq: int = 2       # tokens with df < this considered rare
    long_token_length: int = 24
    replace_with: str = "[RARE]"
    max_replacements: int = 50  # safety cap

def filter_adversarial_terms(text: str,
                             embed_mgr: EmbeddingManager,
                             cfg: AdversarialFilterConfig) -> str:
    if not cfg.enable or not text:
        return text
    tokens = TOKEN_RE.findall(text)
    replacements = 0
    # Build map of suspicious tokens
    suspicious: Set[str] = set()
    for tok in tokens:
        if replacements >= cfg.max_replacements:
            break
        df = embed_mgr.df.get(tok.lower(), 0)
        if (df < cfg.min_doc_freq and (
             len(tok) >= cfg.long_token_length or
             ADVERSARIAL_TOKEN_PATTERN.match(tok) or
             sum(c.isdigit() for c in tok) >= 3 and sum(c.isalpha() for c in tok) >= 3
        )):
            suspicious.add(tok)
    if not suspicious:
        return text
    # Replace using regex with boundaries
    for s in suspicious:
        if replacements >= cfg.max_replacements:
            break
        pattern = re.compile(rf'\b{re.escape(s)}\b')
        new_text, count = pattern.subn(cfg.replace_with, text)
        if count:
            text = new_text
            replacements += count
    return text

# --------------------------------------------------------------------------------------
# Memory Records
# --------------------------------------------------------------------------------------
@dataclass
class MemoryRecord:
    id: str
    content: str
    vector: List[float]
    memory_type: str
    importance: float
    created_at: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "vector": self.vector,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "MemoryRecord":
        return MemoryRecord(
            id=data["id"],
            content=data["content"],
            vector=data["vector"],
            memory_type=data["memory_type"],
            importance=data["importance"],
            created_at=datetime.fromisoformat(data["created_at"]),
            access_count=data.get("access_count",0),
            metadata=data.get("metadata",{})
        )

# --------------------------------------------------------------------------------------
# Memory Store with Persistence
# --------------------------------------------------------------------------------------
class MemoryStore:
    def __init__(self,
                 sanitize: bool = True,
                 phi_categories: Optional[Set[str]] = None,
                 domain_whitelist: Optional[Set[str]] = None,
                 embedding_config: Optional[EmbeddingConfig] = None,
                 adversarial_config: Optional[AdversarialFilterConfig] = None,
                 auto_flush_path: Optional[str] = None,
                 persist_vectors: bool = True):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.memory_by_session: Dict[str, List[MemoryRecord]] = {}
        self.sanitize = sanitize
        self.phi_categories = phi_categories or DEFAULT_PHI_CATEGORIES
        self.domain_whitelist = domain_whitelist or DEFAULT_DOMAIN_WHITELIST
        self.embedding_manager = EmbeddingManager(embedding_config or EmbeddingConfig())
        self.adversarial_config = adversarial_config or AdversarialFilterConfig()
        self.auto_flush_path = auto_flush_path
        self.persist_vectors = persist_vectors

    # ---- Session Management ----
    def create_session(self, agent_name: str) -> str:
        sid = f"session_{agent_name}_{int(time.time()*1000)}"
        self.sessions[sid] = {
            "agent": agent_name,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.memory_by_session[sid] = []
        return sid

    def end_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["ended_at"] = datetime.now(timezone.utc).isoformat()
            self._auto_flush()

    # ---- Memory Operations ----
    def store_memory(self,
                     session_id: str,
                     content: str,
                     memory_type: str = "knowledge",
                     importance: float = 0.5,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        if session_id not in self.memory_by_session:
            raise ValueError("Invalid session_id")

        original_content = content
        if self.sanitize:
            content = sanitize_phi(content, categories=self.phi_categories, whitelist=self.domain_whitelist)

        # Adversarial filtering BEFORE embedding but AFTER PHI
        content = filter_adversarial_terms(content, self.embedding_manager, self.adversarial_config)

        # Update DF from original tokens (pre-embedding) to reflect semantic source
        tokens_for_df = TOKEN_RE.findall(original_content.lower())
        self.embedding_manager.update_document_frequencies(tokens_for_df)

        if len(self.memory_by_session[session_id]) >= MAX_MEMORY:
            self.memory_by_session[session_id].pop(0)

        mem_id = f"mem_{int(time.time()*1000)}_{len(content)%97}"
        vec = self.embedding_manager.text_to_vector(content)
        rec = MemoryRecord(
            id=mem_id,
            content=content,
            vector=vec,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        self.memory_by_session[session_id].append(rec)
        self._auto_flush()
        return mem_id

    def retrieve(self,
                 session_id: str,
                 query: str,
                 limit: int = 7,
                 min_importance: float = 0.25,
                 sanitize_query: bool = False) -> List[MemoryRecord]:
        if sanitize_query and self.sanitize:
            query = sanitize_phi(query, categories=self.phi_categories, whitelist=self.domain_whitelist)
        pool = self.memory_by_session.get(session_id, [])
        if not pool:
            return []
        qvec = self.embedding_manager.text_to_vector(query)
        scored: List[Tuple[float, MemoryRecord]] = []
        for rec in pool:
            if rec.importance < min_importance:
                continue
            sim = cosine_similarity(qvec, rec.vector)  # normalized already
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    # ---- Persistence ----
    def save_state(self, path: str):
        data = {
            "sessions": self.sessions,
            "memories": {
                sid: [m.to_json() if self.persist_vectors else {**m.to_json(), "vector": []}
                      for m in arr]
                for sid, arr in self.memory_by_session.items()
            },
            "embedding": {
                "config": asdict(self.embedding_manager.config),
                "doc_count": self.embedding_manager.doc_count,
                "df": dict(self.embedding_manager.df)
            },
            "phi_categories": list(self.phi_categories),
            "domain_whitelist": list(self.domain_whitelist),
            "adversarial_config": asdict(self.adversarial_config),
            "persist_vectors": self.persist_vectors,
            "version": "1.1"
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, path: str) -> "MemoryStore":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        emb_cfg = EmbeddingConfig(**data["embedding"]["config"])
        store = cls(
            sanitize=True,
            phi_categories=set(data.get("phi_categories", DEFAULT_PHI_CATEGORIES)),
            domain_whitelist=set(data.get("domain_whitelist", DEFAULT_DOMAIN_WHITELIST)),
            embedding_config=emb_cfg,
            adversarial_config=AdversarialFilterConfig(**data.get("adversarial_config", {})),
            persist_vectors=data.get("persist_vectors", True)
        )
        store.sessions = data["sessions"]
        for sid, arr in data["memories"].items():
            store.memory_by_session[sid] = []
            for rec_json in arr:
                rec = MemoryRecord.from_json(rec_json)
                # If vectors absent -> recompute
                if not rec.vector:
                    rec.vector = store.embedding_manager.text_to_vector(rec.content)
                store.memory_by_session[sid].append(rec)
        # restore DF
        store.embedding_manager.doc_count = data["embedding"]["doc_count"]
        store.embedding_manager.df = Counter(data["embedding"]["df"])
        return store

    def _auto_flush(self):
        if self.auto_flush_path:
            try:
                self.save_state(self.auto_flush_path)
            except Exception as e:
                logger.warning(f"Auto-flush failed: {e}")

# --------------------------------------------------------------------------------------
# Cosine Similarity (assumes normalized vectors)
# --------------------------------------------------------------------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

# --------------------------------------------------------------------------------------
# Localization Data (unchanged sections condensed for brevity)
# --------------------------------------------------------------------------------------
RT_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "medical_consultation": "Clinical context synthesis",
        "diagnosis": "Diagnostic structuring",
        "imaging_analysis": "Imaging interpretation outline",
        "deductive": "Logical derivation from premises",
        "inductive": "Pattern generalization",
        "abductive": "Plausible explanatory hypothesis formation",
        "risk_stratification": "Risk factor integration and tiering",
        "prognostic": "Future trajectory framing",
        "therapeutic_planning": "Intervention planning scaffold",
        "guideline_concordance": "Heuristic guideline alignment",
        "temporal_trend": "Temporal evolution assessment",
        "causal_analysis": "Causal pathway exploration",
        "uncertainty_quantification": "Uncertainty domain profiling",
        "counterfactual": "What-if contrastive reasoning",
        "triage_decision": "Urgency prioritization",
        "ethics_reflection": "Ethical and bias reflection",
        "general": "General analytic synthesis"
    },
    # (Other language mappings same as previous version) ...
}
# For brevity in this enhanced script snippet we keep full earlier language dicts assumption:
# In production keep full dictionaries (omitted duplication here due to space). If needed copy full mapping.

# (Reusing the full earlier dictionaries exactly as before – they should be included in a real file.)
# For this answer we re-insert them fully to maintain standalone functionality:

RT_DESCRIPTIONS.update({
    "es": {
        "medical_consultation": "Síntesis clínica contextual",
        "diagnosis": "Estructuración diagnóstica",
        "imaging_analysis": "Interpretación de imagen",
        "deductive": "Derivación lógica",
        "inductive": "Generalización de patrones",
        "abductive": "Hipótesis explicativa plausible",
        "risk_stratification": "Integración y estratificación de riesgo",
        "prognostic": "Enmarcado de trayectoria futura",
        "therapeutic_planning": "Planificación terapéutica",
        "guideline_concordance": "Alineación heurística con guías",
        "temporal_trend": "Evaluación de evolución temporal",
        "causal_analysis": "Exploración de causalidad",
        "uncertainty_quantification": "Perfil de incertidumbre",
        "counterfactual": "Razonamiento contrafactual",
        "triage_decision": "Priorización de urgencia",
        "ethics_reflection": "Reflexión ética",
        "general": "Síntesis analítica general"
    },
    "fr": {
        "medical_consultation": "Synthèse contextuelle clinique",
        "diagnosis": "Structuration diagnostique",
        "imaging_analysis": "Interprétation d'imagerie",
        "deductive": "Dérivation logique",
        "inductive": "Généralisation de motifs",
        "abductive": "Hypothèse explicative plausible",
        "risk_stratification": "Intégration et stratification du risque",
        "prognostic": "Projection de trajectoire future",
        "therapeutic_planning": "Planification thérapeutique",
        "guideline_concordance": "Alignement heuristique aux guides",
        "temporal_trend": "Évaluation temporelle",
        "causal_analysis": "Exploration causale",
        "uncertainty_quantification": "Profil d'incertitude",
        "counterfactual": "Analyse contrefactuelle",
        "triage_decision": "Priorisation d'urgence",
        "ethics_reflection": "Réflexion éthique",
        "general": "Synthèse analytique générale"
    },
    "de": {
        "medical_consultation": "Klinische kontextuelle Synthese",
        "diagnosis": "Diagnostische Strukturierung",
        "imaging_analysis": "Bildgebungsinterpretation",
        "deductive": "Logische Ableitung",
        "inductive": "Mustergeneralisierung",
        "abductive": "Plausible erklärende Hypothese",
        "risk_stratification": "Risikofaktor-Integration und Stufung",
        "prognostic": "Zukünftige Verlaufseinschätzung",
        "therapeutic_planning": "Therapieplan-Struktur",
        "guideline_concordance": "Heuristische Leitlinienanpassung",
        "temporal_trend": "Zeitverlaufsbewertung",
        "causal_analysis": "Kausale Pfadexploration",
        "uncertainty_quantification": "Unsicherheitsprofilierung",
        "counterfactual": "Was-wäre-wenn Analyse",
        "triage_decision": "Dringlichkeitspriorisierung",
        "ethics_reflection": "Ethische Reflexion",
        "general": "Allgemeine analytische Synthese"
    },
    "pt": {
        "medical_consultation": "Síntese clínica contextual",
        "diagnosis": "Estruturação diagnóstica",
        "imaging_analysis": "Interpretação de imagem",
        "deductive": "Derivação lógica",
        "inductive": "Generalização de padrões",
        "abductive": "Hipótese explicativa plausível",
        "risk_stratification": "Integração e estratificação de risco",
        "prognostic": "Enquadramento de trajetória futura",
        "therapeutic_planning": "Planejamento terapêutico",
        "guideline_concordance": "Alinhamento heurístico a diretrizes",
        "temporal_trend": "Avaliação temporal",
        "causal_analysis": "Exploração causal",
        "uncertainty_quantification": "Perfil de incerteza",
        "counterfactual": "Raciocínio contrafactual",
        "triage_decision": "Priorização de urgência",
        "ethics_reflection": "Reflexão ética",
        "general": "Síntese analítica geral"
    },
    "it": {
        "medical_consultation": "Sintesi clinica contestuale",
        "diagnosis": "Strutturazione diagnostica",
        "imaging_analysis": "Interpretazione di imaging",
        "deductive": "Derivazione logica",
        "inductive": "Generalizzazione di pattern",
        "abductive": "Ipotesi esplicativa plausibile",
        "risk_stratification": "Integrazione e stratificazione del rischio",
        "prognostic": "Inquadramento della traiettoria futura",
        "therapeutic_planning": "Pianificazione terapeutica",
        "guideline_concordance": "Allineamento euristico alle linee guida",
        "temporal_trend": "Valutazione temporale",
        "causal_analysis": "Esplorazione causale",
        "uncertainty_quantification": "Profilazione dell'incertezza",
        "counterfactual": "Ragionamento controfattuale",
        "triage_decision": "Prioritizzazione d'urgenza",
        "ethics_reflection": "Riflessione etica",
        "general": "Sintesi analitica generale"
    },
    "zh": {
        "medical_consultation": "临床情境综合",
        "diagnosis": "诊断结构化",
        "imaging_analysis": "影像解读概述",
        "deductive": "逻辑推演",
        "inductive": "模式归纳",
        "abductive": "可行解释假设",
        "risk_stratification": "风险因素整合与分层",
        "prognostic": "未来趋势评估",
        "therapeutic_planning": "干预方案框架",
        "guideline_concordance": "指南启发性对照",
        "temporal_trend": "时间演变分析",
        "causal_analysis": "因果路径探索",
        "uncertainty_quantification": "不确定性剖析",
        "counterfactual": "反事实对比推理",
        "triage_decision": "紧急程度优先级",
        "ethics_reflection": "伦理与偏差反思",
        "general": "通用分析综合"
    },
    "ja": {
        "medical_consultation": "臨床コンテキスト統合",
        "diagnosis": "診断構造化",
        "imaging_analysis": "画像解釈概要",
        "deductive": "論理的演繹",
        "inductive": "パターン帰納",
        "abductive": "もっともらしい仮説形成",
        "risk_stratification": "リスク要因統合と層別化",
        "prognostic": "将来経過の枠組み",
        "therapeutic_planning": "治療計画フレーム",
        "guideline_concordance": "ガイドライン的整合ヒューリスティック",
        "temporal_trend": "時間的推移評価",
        "causal_analysis": "因果経路探索",
        "uncertainty_quantification": "不確実性プロファイル",
        "counterfactual": "反事実的対比推論",
        "triage_decision": "緊急度優先順位",
        "ethics_reflection": "倫理・バイアス考察",
        "general": "一般的分析統合"
    },
})

RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
    # identical to previous version (kept for completeness) ...
    "en": {
        "risk_stratification": ["Aggregate genetic & cognitive indicators", "Document explicit tier rationale"],
        "prognostic": ["Schedule periodic reassessment", "Track functional & cognitive slope"],
        "therapeutic_planning": ["Optimize modifiable lifestyle factors", "Define reassessment triggers"],
        "uncertainty_quantification": ["Acquire more longitudinal data", "Validate with external sources"],
        "causal_analysis": ["Map hypothesized mediator pathways", "Differentiate correlation vs causation"],
        "general": ["Gather additional structured history", "Maintain broad differential space"]
    },
    "es": {
        "risk_stratification": ["Integrar indicadores genéticos y cognitivos", "Documentar la justificación del nivel"],
        "prognostic": ["Programar reevaluaciones periódicas", "Seguir la pendiente funcional y cognitiva"],
        "therapeutic_planning": ["Optimizar factores modificables", "Definir disparadores de reevaluación"],
        "uncertainty_quantification": ["Obtener más datos longitudinales", "Validar con fuentes externas"],
        "causal_analysis": ["Mapear vías mediadoras hipotéticas", "Distinguir correlación de causalidad"],
        "general": ["Recopilar historial estructurado adicional", "Mantener diferencial amplio"]
    },
    "fr": {
        "risk_stratification": ["Intégrer indicateurs génétiques et cognitifs", "Documenter la justification du palier"],
        "prognostic": ["Planifier réévaluations périodiques", "Suivre pente fonctionnelle et cognitive"],
        "therapeutic_planning": ["Optimiser facteurs modifiables", "Définir déclencheurs de réévaluation"],
        "uncertainty_quantification": ["Acquérir davantage de données longitudinales", "Valider par sources externes"],
        "causal_analysis": ["Cartographier voies médiatrices hypothétiques", "Distinguer corrélation vs causalité"],
        "general": ["Recueillir données structurées supplémentaires", "Maintenir un large différentiel"]
    },
    "de": {
        "risk_stratification": ["Genetische & kognitive Indikatoren bündeln", "Begründung der Risikostufe dokumentieren"],
        "prognostic": ["Regelmäßige Reevaluation planen", "Funktionellen & kognitiven Verlauf verfolgen"],
        "therapeutic_planning": ["Modifizierbare Faktoren optimieren", "Neubewertungs-Trigger definieren"],
        "uncertainty_quantification": ["Mehr Longitudinaldaten sammeln", "Mit externen Quellen validieren"],
        "causal_analysis": ["Hypothetische Mediatorpfade abbilden", "Korrelation von Kausalität trennen"],
        "general": ["Zusätzliche strukturierte Anamnese sammeln", "Breiten Differenzialraum erhalten"]
    },
    "pt": {
        "risk_stratification": ["Integrar indicadores genéticos e cognitivos", "Documentar a justificativa do nível"],
        "prognostic": ["Agendar reavaliações periódicas", "Monitorar declínio funcional e cognitivo"],
        "therapeutic_planning": ["Otimizar fatores modificáveis", "Definir gatilhos de reavaliação"],
        "uncertainty_quantification": ["Coletar mais dados longitudinais", "Validar com fontes externas"],
        "causal_analysis": ["Mapear vias mediadoras hipotéticas", "Distinguir correlação de causalidade"],
        "general": ["Reunir histórico adicional estruturado", "Manter diferencial amplo"]
    },
    "it": {
        "risk_stratification": ["Integrare indicatori genetici e cognitivi", "Documentare la motivazione del livello"],
        "prognostic": ["Pianificare rivalutazioni periodiche", "Monitorare declino funzionale e cognitivo"],
        "therapeutic_planning": ["Ottimizzare fattori modificabili", "Definire trigger di rivalutazione"],
        "uncertainty_quantification": ["Raccogliere più dati longitudinali", "Validare con fonti esterne"],
        "causal_analysis": ["Mappare vie mediatrici ipotetiche", "Distinguere correlazione da causalità"],
        "general": ["Raccogliere anamnesi strutturata aggiuntiva", "Mantenere ampio spettro differenziale"]
    },
    "zh": {
        "risk_stratification": ["整合遗传与认知指标", "记录风险层级依据"],
        "prognostic": ["安排定期再评估", "跟踪功能与认知趋势"],
        "therapeutic_planning": ["优化可调控生活方式因素", "定义再评估触发条件"],
        "uncertainty_quantification": ["获取更多纵向数据", "与外部来源交叉验证"],
        "causal_analysis": ["映射假设的中介路径", "区分相关与因果"],
        "general": ["收集更多结构化病史", "保持广泛的鉴别范围"]
    },
    "ja": {
        "risk_stratification": ["遺伝および認知指標を統合", "リスク層別の根拠を記録"],
        "prognostic": ["定期的な再評価を設定", "機能・認知の変化傾向を追跡"],
        "therapeutic_planning": ["修正可能因子を最適化", "再評価トリガーを定義"],
        "uncertainty_quantification": ["さらなる縦断データを取得", "外部情報で検証"],
        "causal_analysis": ["仮説的媒介経路をマッピング", "相関と因果を区別"],
        "general": ["追加の構造化履歴を収集", "広い鑑別範囲を維持"]
    },
}

DISCLAIMER = {
    "en": "DISCLAIMER: Not medical advice; consult qualified professionals.",
    "es": "DESCARGO: No es consejo médico; consulte profesionales calificados.",
    "fr": "AVERTISSEMENT : Pas un avis médical ; consultez des professionnels qualifiés.",
    "de": "HAFTUNGSAUSSCHLUSS: Keine medizinische Beratung; konsultieren Sie Fachpersonal.",
    "pt": "AVISO: Não é conselho médico; consulte profissionais qualificados.",
    "it": "AVVISO: Non è un consiglio medico; consultare professionisti qualificati.",
    "zh": "免责声明：非医疗建议；请咨询专业人士。",
    "ja": "免責事項：医療アドバイスではありません。専門家に相談してください。"
}

HEADINGS = {
    "analysis": {
        "en":"Analysis","es":"Análisis","fr":"Analyse","de":"Analyse","pt":"Análise","it":"Analisi","zh":"分析","ja":"分析"
    },
    "factors": {
        "en":"Key Factors","es":"Factores Clave","fr":"Facteurs Clés","de":"Schlüsselfaktoren","pt":"Fatores-Chave","it":"Fattori Chiave","zh":"关键因素","ja":"主要因子"
    },
    "uncertainty": {
        "en":"Uncertainty","es":"Incertidumbre","fr":"Incertitude","de":"Unsicherheit","pt":"Incerteza","it":"Incertezza","zh":"不确定性","ja":"不確実性"
    },
    "considerations": {
        "en":"Considerations","es":"Consideraciones","fr":"Considérations","de":"Erwägungen","pt":"Considerações","it":"Considerazioni","zh":"考虑事项","ja":"考慮事項"
    }
}

def local_rt_desc(rt: str, lang: str) -> str:
    return RT_DESCRIPTIONS.get(lang, RT_DESCRIPTIONS["en"]).get(rt, RT_DESCRIPTIONS["en"]["general"])

# --------------------------------------------------------------------------------------
# CoT Summarizer (unchanged logic)
# --------------------------------------------------------------------------------------
class HeuristicCoTSummarizer:
    def __init__(self, max_sentences: int = 6, max_chars: int = 1400):
        self.max_sentences = max_sentences
        self.max_chars = max_chars

    def summarize(self, query: str, memories: List['MemoryRecord']) -> Dict[str, Any]:
        if not memories:
            return {
                "summary_text": f"Query '{query}' has no contextual memories.",
                "concepts": [],
                "axes": ["insufficient_context"],
                "selected": []
            }
        raw_sents = []
        for m in memories:
            for s in re.split(r'(?<=[.!?])\s+', m.content):
                s = s.strip()
                if 10 <= len(s) <= 300:
                    raw_sents.append(s)
            if len(raw_sents) > 80:
                break
        raw_sents = list(dict.fromkeys(raw_sents))
        if not raw_sents:
            raw_sents = [m.content[:250] for m in memories]

        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for s in raw_sents:
            st = set(re.findall(r"\w+", s.lower()))
            overlap = len(q_tokens & st)
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        chosen = []
        used = set()
        for ov, sentence in scored:
            sig = " ".join(sorted(set(re.findall(r"\w+", sentence.lower()))))
            if sig in used:
                continue
            chosen.append(sentence)
            used.add(sig)
            if len(chosen) >= self.max_sentences:
                break

        freq: Dict[str,int] = {}
        for c in chosen:
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", c.lower()):
                if tok in {"with","that","this","from","were","which","therefore","into","will","should","could"}:
                    continue
                freq[tok] = freq.get(tok,0)+1
        concepts = [w for w,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:12]]
        axes = []
        if any(x in concepts for x in ("apoe4","apoe")):
            axes.append("genetic_risk")
        if any(x in concepts for x in ("mmse","moca","score")):
            axes.append("cognitive_metrics")
        if not axes:
            axes.append("general")

        parts = [
            f"Query: {query}",
            f"Axes: {', '.join(axes)}",
            f"Key concepts: {', '.join(concepts[:8]) if concepts else 'none'}",
            "Context signals:"
        ]
        for sent in chosen:
            candidate = "\n".join(parts + [f"- {sent}"])
            if len(candidate) > self.max_chars:
                break
            parts.append(f"- {sent}")
        parts.append("Condensed analytic abstraction (truncated; not full internal chain).")
        summary_text = "\n".join(parts)

        return {
            "summary_text": summary_text,
            "concepts": concepts,
            "axes": axes,
            "selected": chosen
        }

# --------------------------------------------------------------------------------------
# Reasoning Data Classes
# --------------------------------------------------------------------------------------
@dataclass
class ReasoningContext:
    agent_id: str
    session_id: str
    query: str
    reasoning_type: str
    memories: List[MemoryRecord]
    memory_weights: List[float]
    structured_features: Dict[str, Any]
    memory_stats: Dict[str, Any]
    language: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ReasoningResult:
    response: str
    confidence: float
    reasoning_type: str
    context_used: bool
    memory_count: int
    steps: List[str]
    diagnostics: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# --------------------------------------------------------------------------------------
# Template Generator
# --------------------------------------------------------------------------------------
class TemplateGenerator:
    def generate(self,
                 lang: str,
                 reasoning_type: str,
                 skeleton: str,
                 cot: Optional[Dict[str, Any]],
                 features: Dict[str, Any],
                 confidence: float) -> str:
        lang = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG
        desc = local_rt_desc(reasoning_type, lang)
        disclaimer = DISCLAIMER[lang]
        analysis_h = HEADINGS["analysis"][lang]
        factors_h = HEADINGS["factors"][lang]
        uncertainty_h = HEADINGS["uncertainty"][lang]
        consider_h = HEADINGS["considerations"][lang]

        biom = ", ".join(features.get("biomarkers", [])) or "none"
        scores = ", ".join([f"{s['kind']}={s['value']}" for s in features.get("scores", [])]) or "none"
        temporal = f"{features.get('temporal_reference_density',0):.2f}"

        if confidence >= 0.75:
            unc_map = {
                "en":"Residual uncertainty low-moderate.",
                "es":"Incertidumbre residual baja-moderada.",
                "fr":"Incertitude résiduelle faible-moyenne.",
                "de":"Restunsicherheit gering bis moderat.",
                "pt":"Incerteza residual baixa-moderada.",
                "it":"Incertezza residua bassa-moderata.",
                "zh":"剩余不确定性为低至中等。",
                "ja":"残余の不確実性は低～中程度。"
            }
        elif confidence >= 0.55:
            unc_map = {
                "en":"Moderate uncertainty; further data advised.",
                "es":"Incertidumbre moderada; se aconsejan más datos.",
                "fr":"Incertitude modérée ; davantage de données conseillé.",
                "de":"Moderate Unsicherheit; weitere Daten empfohlen.",
                "pt":"Incerteza moderada; mais dados recomendados.",
                "it":"Incertezza moderata; si consigliano ulteriori dati.",
                "zh":"中等不确定性；建议补充数据。",
                "ja":"不確実性は中程度。追加データが望ましい。"
            }
        else:
            unc_map = {
                "en":"High uncertainty; interpret cautiously.",
                "es":"Alta incertidumbre; interpretar con cautela.",
                "fr":"Forte incertitude ; interpréter avec prudence.",
                "de":"Hohe Unsicherheit; vorsichtig interpretieren.",
                "pt":"Alta incerteza; interpretar com cautela.",
                "it":"Elevata incertezza; interpretare con cautela.",
                "zh":"高度不确定性；需谨慎解读。",
                "ja":"高い不確実性。慎重に解釈してください。"
            }
        unc_text = unc_map[lang]

        recs = self._recommendations(lang, reasoning_type, features)
        cot_section = ""
        if cot:
            cot_section = "\n[CondensedContext]\n" + cot.get("summary_text","")

        lines = [
            f"{analysis_h}: {desc}",
            skeleton,
            f"{factors_h}: biomarkers={biom}; scores={scores}; temporal_density={temporal}",
            f"{uncertainty_h}: {unc_text}",
            consider_h + ":"
        ] + [f"- {r}" for r in recs] + [
            cot_section,
            f"Confidence≈{confidence:.2f}",
            disclaimer
        ]
        out = "\n".join([ln for ln in lines if ln.strip()])
        return re.sub(r"\n{3,}", "\n\n", out).strip()

    def _recommendations(self, lang: str, reasoning_type: str, features: Dict[str, Any]) -> List[str]:
        base = RECOMMENDATIONS.get(lang, RECOMMENDATIONS["en"])
        recs = base.get(reasoning_type, base.get("general", []))
        if "apoe4" in features.get("biomarkers", []):
            extra = {
                "en":"Emphasize genetic risk counseling",
                "es":"Enfatizar asesoramiento de riesgo genético",
                "fr":"Mettre l'accent sur le conseil en risque génétique",
                "de":"Genetische Risikoberatung betonen",
                "pt":"Enfatizar aconselhamento de risco genético",
                "it":"Enfatizzare consulenza sul rischio genetico",
                "zh":"强调遗传风险咨询",
                "ja":"遺伝リスクカウンセリングを強調"
            }[lang]
            recs.append(extra)
        if features.get("has_cognitive_score"):
            extra2 = {
                "en":"Track longitudinal cognitive change",
                "es":"Rastrear cambio cognitivo longitudinal",
                "fr":"Suivre l'évolution cognitive longitudinale",
                "de":"Langfristige kognitive Veränderungen verfolgen",
                "pt":"Acompanhar mudança cognitiva longitudinal",
                "it":"Tracciare variazione cognitiva longitudinale",
                "zh":"跟踪认知纵向变化",
                "ja":"認知の経時的変化を追跡"
            }[lang]
            recs.append(extra2)
        # Dedup
        seen = set()
        ordered = []
        for r in recs:
            if r not in seen:
                ordered.append(r)
                seen.add(r)
        return ordered[:6]

# --------------------------------------------------------------------------------------
# Reasoning Agent
# --------------------------------------------------------------------------------------
class PureLiteReasoningAgent:
    def __init__(self,
                 agent_id: str,
                 memory_store: MemoryStore,
                 similarity_threshold: float = SIMILARITY_THRESHOLD,
                 max_context_memories: int = 7,
                 enable_cot: bool = True,
                 sanitize_query: bool = True):
        self.agent_id = agent_id
        self.store = memory_store
        self.similarity_threshold = similarity_threshold
        self.max_context_memories = max_context_memories
        self.session_id = self.store.create_session(agent_name=agent_id)
        self.cot = HeuristicCoTSummarizer() if enable_cot else None
        self.generator = TemplateGenerator()
        self.sanitize_query = sanitize_query
        logger.info(f"PureLiteReasoningAgent initialized: session={self.session_id} dim={self.store.embedding_manager.config.dim} weighting={self.store.embedding_manager.config.weighting}")

    def store_memory(self, content: str, memory_type: str = "knowledge", importance: float = 0.5):
        return self.store.store_memory(self.session_id, content, memory_type, importance)

    def end_session(self):
        self.store.end_session(self.session_id)

    def reason(self,
               query: str,
               reasoning_type: str = "general",
               language: Optional[str] = None) -> ReasoningResult:
        lang = (language or DEFAULT_LANG).lower()
        if lang not in SUPPORTED_LANGS:
            lang = DEFAULT_LANG

        rt = reasoning_type if reasoning_type in RT_DESCRIPTIONS["en"] else "general"
        original_query = query
        if self.sanitize_query and self.store.sanitize:
            query = sanitize_phi(query, categories=self.store.phi_categories, whitelist=self.store.domain_whitelist)

        raw = self.store.retrieve(self.session_id, query, limit=self.max_context_memories, sanitize_query=False)
        filtered = self._filter_by_overlap(raw, query)
        weights = self._compute_weights(filtered)
        features, stats = self._extract_features(filtered)
        steps = [
            f"OriginalQuery='{original_query}'",
            f"SanitizedQuery='{query}'" if query != original_query else "SanitizedQuery=unchanged",
            f"ReasoningType={rt}",
            f"Lang={lang}",
            f"MemoriesRetrieved={len(raw)}",
            f"Filtered={len(filtered)}",
            f"EmbeddingDim={self.store.embedding_manager.config.dim}",
            f"Weighting={self.store.embedding_manager.config.weighting}"
        ]

        cot_data = self.cot.summarize(query, filtered) if (self.cot and filtered) else None
        if cot_data:
            features["cot_summary"] = cot_data
            steps.append("CoT condensation generated")

        skeleton = self._skeleton(rt, filtered, features)
        confidence, conf_break = self._confidence(rt, filtered, weights, features)
        steps.append("Confidence computed")

        response = self.generator.generate(
            lang=lang,
            reasoning_type=rt,
            skeleton=skeleton,
            cot=cot_data,
            features=features,
            confidence=confidence
        )

        diagnostics = {
            "confidence_breakdown": conf_break,
            "features": {k: v for k,v in features.items() if k != "cot_summary"},
            "memory_type_distribution": stats.get("type_distribution"),
            "cot_used": cot_data is not None,
            "vector_weighting": self.store.embedding_manager.config.weighting
        }

        return ReasoningResult(
            response=response,
            confidence=confidence,
            reasoning_type=rt,
            context_used=bool(filtered),
            memory_count=len(filtered),
            steps=steps,
            diagnostics=diagnostics
        )

    # --- Internal Helpers ---
    def _filter_by_overlap(self, memories: List[MemoryRecord], query: str) -> List[MemoryRecord]:
        qtok = set(re.findall(r"\w+", query.lower()))
        out: List[MemoryRecord] = []
        for m in memories:
            mtok = set(re.findall(r"\w+", m.content.lower()))
            overlap = len(qtok & mtok) / max(len(qtok), 1)
            heuristic = 0.55 * overlap + 0.45 * m.importance
            if heuristic >= self.similarity_threshold:
                out.append(m)
        return out

    def _compute_weights(self, memories: List[MemoryRecord]) -> List[float]:
        now = datetime.now(timezone.utc)
        weights = []
        for m in memories:
            hours = (now - m.created_at).total_seconds() / 3600
            recency = max(0, 1 - hours / 240)
            access = min(0.2, m.access_count * 0.01)
            weight = min(1.0, m.importance * 0.5 + recency * 0.2 + access + 0.05)
            weights.append(weight)
        return weights

    def _extract_features(self, memories: List[MemoryRecord]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        biomarkers = []
        temporal_refs = 0
        scores = []
        types = []
        for m in memories:
            text = m.content.lower()
            types.append(m.memory_type)
            if re.search(r"\b(months?|years?|weeks?)\b", text):
                temporal_refs += 1
            for marker in ["apoe4","apoe","csf","amyloid","tau","mmse","moca"]:
                if marker in text:
                    biomarkers.append(marker)
            for match in re.findall(r"(mmse|moca)\s*(score)?\s*[:=]?\s*(\d{1,2})", text):
                scores.append({"kind": match[0].upper(), "value": int(match[2])})
        type_counts = Counter(types)
        features = {
            "biomarkers": sorted(set(biomarkers)),
            "has_cognitive_score": any(s["kind"] in ("MMSE","MOCA") for s in scores),
            "scores": scores,
            "temporal_reference_density": temporal_refs / max(1, len(memories)),
            "memory_type_diversity": len(type_counts)
        }
        stats = {
            "type_distribution": dict(type_counts),
            "biomarker_count": len(set(biomarkers)),
            "score_count": len(scores)
        }
        return features, stats

    def _skeleton(self, reasoning_type: str, memories: List[MemoryRecord], features: Dict[str, Any]) -> str:
        count = len(memories)
        if reasoning_type == "risk_stratification":
            parts = []
            if "apoe4" in features.get("biomarkers", []):
                parts.append("Genetic risk flag (APOE4)")
            if features.get("has_cognitive_score"):
                parts.append("Objective cognitive metric present")
            if not parts:
                parts.append("Few explicit structured risk indicators")
            return "Risk skeleton: " + "; ".join(parts)
        if reasoning_type == "deductive":
            premises = sum(1 for m in memories if any(k in m.content.lower() for k in ["if "," therefore"," implies"," when "]))
            return f"Deductive skeleton: {premises} premise candidates."
        if reasoning_type == "abductive":
            return "Abductive skeleton: derive plausible explanatory model."
        if reasoning_type == "causal_analysis":
            return "Causal skeleton: map potential multi-factor linkages."
        if reasoning_type == "uncertainty_quantification":
            return "Uncertainty skeleton: enumerate knowledge gaps."
        if reasoning_type == "counterfactual":
            return "Counterfactual skeleton: baseline vs hypothetical scenario."
        if reasoning_type == "prognostic":
            return f"Prognostic skeleton: {count} context items; future trajectory framing."
        if reasoning_type == "therapeutic_planning":
            return "Therapeutic skeleton: optimize modifiable domains → define reassessment triggers."
        if reasoning_type == "temporal_trend":
            td = features.get("temporal_reference_density",0)
            return f"Temporal skeleton: reference density={td:.2f}."
        if reasoning_type == "guideline_concordance":
            return "Guideline skeleton: check standardized assessment presence."
        if reasoning_type == "triage_decision":
            return "Triage skeleton: scan for red flags; escalate if rapid deterioration emerges."
        if reasoning_type == "ethics_reflection":
            return "Ethics skeleton: consent, bias, non-deterministic framing."
        if reasoning_type == "diagnosis":
            return "Diagnostic skeleton: symptoms → scores → biomarkers → differential refinement."
        if reasoning_type == "imaging_analysis":
            return "Imaging skeleton: technical quality → morphology → signal characteristics → comparison."
        if reasoning_type == "inductive":
            return f"Inductive skeleton: generalize from {count} observation(s)."
        if reasoning_type == "medical_consultation":
            return "Consultation skeleton: integrate cognitive, biomarker, history elements."
        return f"General skeleton: {count} contextual item(s)."

    def _confidence(self,
                    reasoning_type: str,
                    memories: List[MemoryRecord],
                    weights: List[float],
                    features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        breakdown: Dict[str, float] = {}
        base = 0.48
        breakdown["base"] = base
        mc = len(memories)
        coverage = min(0.28, mc * 0.07)
        breakdown["coverage"] = coverage
        diversity = features.get("memory_type_diversity",1)
        diversity_boost = min(0.12, max(0, diversity-1)*0.03)
        breakdown["diversity"] = diversity_boost
        score_boost = 0.05 if features.get("has_cognitive_score") else 0.0
        breakdown["score"] = score_boost
        temporal = features.get("temporal_reference_density",0)
        temporal_adj = min(0.06, temporal * 0.08)
        breakdown["temporal"] = temporal_adj
        type_penalty = {
            "abductive": -0.06,
            "causal_analysis": -0.05,
            "uncertainty_quantification": -0.02,
            "counterfactual": -0.04,
            "deductive": -0.03,
            "inductive": -0.02
        }.get(reasoning_type, 0.0)
        breakdown["type_adj"] = type_penalty
        paucity_penalty = 0.0
        if reasoning_type in ("abductive","causal_analysis") and mc < 3:
            paucity_penalty -= 0.04
        breakdown["paucity"] = paucity_penalty
        final = base + coverage + diversity_boost + score_boost + temporal_adj + type_penalty + paucity_penalty
        final = max(0.0, min(1.0, final))
        breakdown["final"] = final
        return final, breakdown

# --------------------------------------------------------------------------------------
# Demo showcasing new features
# --------------------------------------------------------------------------------------
def demo():
    store = MemoryStore(
        sanitize=True,
        phi_categories={"email","phone","id","address","name"},  # keep dates unmasked
        embedding_config=EmbeddingConfig(dim=512, weighting="tfidf"),
        adversarial_config=AdversarialFilterConfig(enable=True, min_doc_freq=2),
        auto_flush_path=None  # set path to auto-save
    )
    agent = PureLiteReasoningAgent("pure_lite_agent", store,
                                   similarity_threshold=0.5,
                                   max_context_memories=7,
                                   enable_cot=True)

    seed_memories = [
        ("MMSE scores below 24 may indicate cognitive impairment requiring further evaluation.", "knowledge", 0.9),
        ("Patient John Smith had APOE4 genotype reported on 2024-08-11 with mild memory complaints.", "knowledge", 0.88),
        ("Lifestyle interventions including aerobic exercise can modulate cognitive decline trajectory.", "knowledge", 0.83),
        ("CSF biomarkers (amyloid, tau) refine prodromal differential considerations.", "knowledge", 0.8),
        ("Longitudinal change over 12 months may reveal accelerated decline patterns.", "knowledge", 0.78),
        ("Comprehensive assessment should include cognitive testing, biomarker analysis, and family history.", "reasoning", 0.76),
        ("SuspiciousInjection999XYZAlphaBeta token might attempt prompt manipulation", "misc", 0.7),
    ]
    for text, t, imp in seed_memories:
        agent.store_memory(text, t, imp)

    queries = [
        ("What should be considered for MMSE score 22 and APOE4 positive?", "medical_consultation", "en"),
        ("Outline causal influences for accelerated decline.", "causal_analysis", "en"),
    ]

    print("\n=== Enhanced Pure Python Lite Reasoning Demo ===")
    for i,(q, rtype, lang) in enumerate(queries,1):
        print(f"\n--- Query {i} ({rtype}, lang={lang}) ---")
        result = agent.reason(q, rtype, language=lang)
        print("Response:\n", result.response)
        print(f"Confidence: {result.confidence:.2f}")
        print("Memories Used:", result.memory_count)
        print("CoT Used:", result.diagnostics.get("cot_used"))
        print("Steps:", result.steps)

    # Demonstrate persistence
    path = "memory_state_demo.json"
    store.save_state(path)
    print(f"\nState saved to {path}. Reloading...")
    reloaded = MemoryStore.load_state(path)
    print("Reloaded doc_count:", reloaded.embedding_manager.doc_count)

    agent.end_session()
    print("\n✅ Demo complete.")

if __name__ == "__main__":
    demo()
