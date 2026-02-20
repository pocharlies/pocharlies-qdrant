"""
Translation Pipeline for RAG Pipeline
LLM-based translation (EN→ES) and spec normalization for product data.
Includes airsoft terminology glossary and user-editable custom glossary.
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


# ── Airsoft Domain Glossary ───────────────────────────────────────
# Bidirectional EN↔ES terminology for airsoft/tactical equipment.
# Keys are lowercase English terms, values are Spanish translations.
# This glossary is ALWAYS injected into translation prompts.

AIRSOFT_GLOSSARY_EN_ES = {
    # ── Weapon parts ──
    "inner barrel": "cañón interior",
    "outer barrel": "cañón exterior",
    "barrel": "cañón",
    "barrel extension": "extensión de cañón",
    "barrel length": "longitud de cañón",
    "hop-up": "hop-up",
    "hop-up chamber": "cámara de hop-up",
    "hop-up bucking": "goma de hop-up",
    "hop-up nub": "nub de hop-up",
    "gearbox": "gearbox",
    "gearbox shell": "carcasa de gearbox",
    "cylinder": "cilindro",
    "cylinder head": "cabeza de cilindro",
    "piston": "pistón",
    "piston head": "cabeza de pistón",
    "spring": "muelle",
    "spring guide": "guía de muelle",
    "nozzle": "nozzle",
    "air nozzle": "nozzle de aire",
    "tappet plate": "tappet plate",
    "trigger": "gatillo",
    "trigger unit": "unidad de gatillo",
    "trigger guard": "guardamonte",
    "selector plate": "selector de tiro",
    "fire selector": "selector de tiro",
    "safety": "seguro",
    "safety lever": "palanca de seguro",
    "stock": "culata",
    "folding stock": "culata plegable",
    "retractable stock": "culata retráctil",
    "adjustable stock": "culata ajustable",
    "buffer tube": "tubo de culata",
    "handguard": "guardamanos",
    "rail": "riel",
    "rail system": "sistema de rieles",
    "picatinny rail": "riel Picatinny",
    "M-LOK": "M-LOK",
    "KeyMod": "KeyMod",
    "upper receiver": "cuerpo superior",
    "lower receiver": "cuerpo inferior",
    "receiver": "cuerpo",
    "dust cover": "tapa de ventana de expulsión",
    "bolt catch": "retenedor del cerrojo",
    "bolt": "cerrojo",
    "charging handle": "manija de carga",
    "forward assist": "asistente de avance",
    "flash hider": "apagallamas",
    "muzzle brake": "freno de boca",
    "suppressor": "silenciador",
    "silencer": "silenciador",
    "tracer unit": "unidad trazadora",
    "grip": "empuñadura",
    "pistol grip": "empuñadura de pistola",
    "foregrip": "empuñadura delantera",
    "vertical grip": "empuñadura vertical",
    "angled grip": "empuñadura angular",
    "bipod": "bípode",
    "sling": "correa portafusil",
    "sling mount": "anclaje de correa",
    "sling swivel": "eslabón de correa",
    "magazine": "cargador",
    "mag": "cargador",
    "mid-cap magazine": "cargador mid-cap",
    "hi-cap magazine": "cargador hi-cap",
    "low-cap magazine": "cargador low-cap",
    "drum magazine": "cargador de tambor",
    "magazine well": "alojamiento de cargador",
    "mag release": "liberador de cargador",
    "magazine catch": "retenedor de cargador",
    "speed loader": "cargador rápido",
    "BB loader": "cargador de BBs",
    # ── Weapon types ──
    "airsoft gun": "réplica de airsoft",
    "airsoft rifle": "réplica de airsoft tipo rifle",
    "airsoft pistol": "pistola de airsoft",
    "replica": "réplica",
    "AEG": "AEG (fusil eléctrico automático)",
    "GBB": "GBB (gas blowback)",
    "gas blowback": "gas blowback",
    "NBB": "NBB (gas sin blowback)",
    "spring powered": "accionada por muelle",
    "bolt action": "cerrojo",
    "bolt-action rifle": "rifle de cerrojo",
    "sniper rifle": "rifle de francotirador",
    "DMR": "DMR (tirador designado)",
    "submachine gun": "subfusil",
    "shotgun": "escopeta",
    "grenade launcher": "lanzagranadas",
    "sidearm": "arma secundaria",
    "carbine": "carabina",
    "PDW": "PDW (arma de defensa personal)",
    "LMG": "ametralladora ligera",
    "support weapon": "arma de apoyo",
    # ── Propulsion / power ──
    "blowback": "blowback",
    "recoil": "retroceso",
    "green gas": "green gas",
    "CO2": "CO2",
    "propane": "propano",
    "HPA": "HPA (aire de alta presión)",
    "high pressure air": "aire de alta presión",
    "LiPo battery": "batería LiPo",
    "NiMH battery": "batería NiMH",
    "battery": "batería",
    "charger": "cargador de batería",
    "smart charger": "cargador inteligente",
    "MOSFET": "MOSFET",
    "ETU": "ETU (unidad de gatillo electrónico)",
    "rate of fire": "cadencia de tiro",
    "ROF": "cadencia de tiro",
    "FPS": "FPS",
    "feet per second": "pies por segundo",
    "muzzle velocity": "velocidad de boca",
    "joule": "julio",
    "joules": "julios",
    # ── Ammunition ──
    "BB": "BB",
    "BBs": "BBs",
    "pellet": "balín",
    "biodegradable BBs": "BBs biodegradables",
    "tracer BBs": "BBs trazadoras",
    "BB weight": "peso de BB",
    # ── Optics & accessories ──
    "red dot sight": "visor de punto rojo",
    "red dot": "punto rojo",
    "holographic sight": "visor holográfico",
    "scope": "mira telescópica",
    "magnifier": "magnificador",
    "iron sights": "miras metálicas",
    "front sight": "punto de mira",
    "rear sight": "alza",
    "flip-up sights": "miras abatibles",
    "laser": "láser",
    "flashlight": "linterna táctica",
    "weaponlight": "linterna de arma",
    "PEQ box": "caja PEQ",
    "pressure switch": "interruptor de presión",
    "mount": "montaje",
    "optic mount": "montaje de óptica",
    "scope mount": "montaje de mira",
    "riser mount": "elevador de montaje",
    # ── Tactical gear ──
    "plate carrier": "portaplacas",
    "chest rig": "chest rig",
    "tactical vest": "chaleco táctico",
    "MOLLE": "MOLLE",
    "pouch": "portacargador",
    "magazine pouch": "portacargador",
    "dump pouch": "bolsa de descarga",
    "holster": "funda",
    "pistol holster": "funda de pistola",
    "belt": "cinturón táctico",
    "battle belt": "cinturón de combate",
    "knee pads": "rodilleras",
    "elbow pads": "coderas",
    "gloves": "guantes tácticos",
    "tactical gloves": "guantes tácticos",
    "camouflage": "camuflaje",
    "BDU": "uniforme de combate",
    "combat shirt": "camisa de combate",
    "combat pants": "pantalón de combate",
    "boots": "botas tácticas",
    "ghillie suit": "traje ghillie",
    "hydration carrier": "portacamelback",
    "backpack": "mochila táctica",
    # ── Protection ──
    "goggles": "gafas de protección",
    "safety glasses": "gafas de seguridad",
    "full face mask": "máscara facial completa",
    "mesh mask": "máscara de malla",
    "lower face mask": "máscara facial inferior",
    "helmet": "casco",
    "helmet cover": "funda de casco",
    "face protection": "protección facial",
    "eye protection": "protección ocular",
    # ── Gameplay ──
    "CQB": "CQB (combate en espacios cerrados)",
    "close quarters battle": "combate en espacios cerrados",
    "milsim": "milsim (simulación militar)",
    "speedsoft": "speedsoft",
    "skirmish": "partida",
    "game": "partida",
    "field": "campo de juego",
    "chrono": "cronógrafo",
    "chronograph": "cronógrafo",
    "FPS limit": "límite de FPS",
    "engagement distance": "distancia de enfrentamiento",
    "minimum engagement distance": "distancia mínima de enfrentamiento",
    "MED": "distancia mínima de enfrentamiento",
    "semi-auto": "semiautomático",
    "full-auto": "automático",
    "burst": "ráfaga",
    "single shot": "tiro a tiro",
    # ── Materials ──
    "full metal": "full metal",
    "metal body": "cuerpo de metal",
    "polymer": "polímero",
    "nylon fiber": "fibra de nylon",
    "ABS plastic": "plástico ABS",
    "CNC machined": "mecanizado CNC",
    "aluminum alloy": "aleación de aluminio",
    "zinc alloy": "aleación de zinc",
    "steel": "acero",
    "stainless steel": "acero inoxidable",
    "real wood": "madera real",
    "wood furniture": "guardamanos de madera",
    "rubber grip": "empuñadura de goma",
    # ── Upgrades & internals ──
    "upgrade": "mejora",
    "upgrade part": "pieza de mejora",
    "tight bore barrel": "cañón de precisión",
    "precision barrel": "cañón de precisión",
    "6.01mm barrel": "cañón de 6.01mm",
    "6.03mm barrel": "cañón de 6.03mm",
    "torque motor": "motor de torque",
    "high speed motor": "motor de alta velocidad",
    "motor": "motor",
    "gear set": "set de engranajes",
    "gear ratio": "relación de engranajes",
    "shimming": "shimming",
    "air seal": "sellado de aire",
    "compression": "compresión",
    "wiring": "cableado",
    "Deans connector": "conector Deans",
    "Tamiya connector": "conector Tamiya",
}

# Build reverse glossary (ES→EN) automatically
AIRSOFT_GLOSSARY_ES_EN = {v.lower(): k for k, v in AIRSOFT_GLOSSARY_EN_ES.items()}


@dataclass
class TranslationJob:
    """Tracks a translation job."""
    job_id: str
    source_lang: str = "en"
    target_lang: str = "es"
    status: str = "running"
    items_processed: int = 0
    items_total: int = 0
    logs: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    MAX_LOGS = 2000

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[:50] + self.logs[-(self.MAX_LOGS - 50):]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "status": self.status,
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


TRANSLATION_SYSTEM_PROMPT = """You are a professional translator specializing in airsoft and tactical equipment.
Translate the following product descriptions from {source} to {target}.

Rules:
- Preserve brand names (Tokyo Marui, WE-Tech, G&G, KWA, etc.) untranslated
- Preserve model numbers (M4A1, Hi-Capa 5.1, MP5, AK-47, etc.) untranslated
- Preserve technical measurements (6mm, 350 FPS, 1.2J, etc.) untranslated
- Preserve SKU/part numbers untranslated
- Maintain the same formatting and paragraph structure
- Use natural, fluent {target_name} appropriate for an e-commerce product listing
- Return ONLY the translations as a JSON array of strings, matching input order
- The array must have exactly the same number of elements as the input"""


# ── Spec Normalization ──────────────────────────────────────────

# Unit conversion functions
UNIT_CONVERSIONS = {
    "fps_to_ms": lambda v: round(v * 0.3048, 1),
    "ms_to_fps": lambda v: round(v / 0.3048),
    "oz_to_g": lambda v: round(v * 28.3495, 1),
    "g_to_oz": lambda v: round(v / 28.3495, 2),
    "lbs_to_kg": lambda v: round(v * 0.453592, 2),
    "kg_to_lbs": lambda v: round(v / 0.453592, 2),
    "inch_to_mm": lambda v: round(v * 25.4, 1),
    "mm_to_inch": lambda v: round(v / 25.4, 2),
}

# Brand name normalization
BRAND_NORMALIZATIONS = {
    "tokyo marui": "Tokyo Marui",
    "tm": "Tokyo Marui",
    "we tech": "WE-Tech",
    "we-tech": "WE-Tech",
    "wetech": "WE-Tech",
    "g&g": "G&G Armament",
    "g & g": "G&G Armament",
    "gg armament": "G&G Armament",
    "kwa": "KWA",
    "ksc": "KSC",
    "vfc": "VFC",
    "elite force": "Elite Force",
    "eliteforce": "Elite Force",
    "ares": "ARES",
    "ics": "ICS",
    "krytac": "Krytac",
    "lct": "LCT",
    "e&l": "E&L",
    "cyma": "CYMA",
    "jg": "JG",
    "jing gong": "JG",
    "classic army": "Classic Army",
    "asg": "ASG",
    "action sport games": "ASG",
    "nuprol": "Nuprol",
    "novritsch": "Novritsch",
    "modify": "Modify",
    "maxx": "MAXX",
    "gate": "GATE",
    "maple leaf": "Maple Leaf",
    "prometheus": "Prometheus",
    "pdi": "PDI",
    "guarder": "Guarder",
    "shs": "SHS",
    "lonex": "Lonex",
    "retro arms": "Retro Arms",
}

LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "ca": "Catalan",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "eu": "Basque",
    "gl": "Galician",
}


class GlossaryStore:
    """Redis-backed store for user-defined glossary terms."""

    KEY = "translation:glossary"

    def __init__(self, redis=None):
        self.redis = redis
        self._cache: Dict[str, str] = {}  # in-memory cache

    async def load(self) -> Dict[str, str]:
        """Load all custom glossary entries from Redis."""
        if not self.redis:
            return self._cache
        try:
            raw = await self.redis.hgetall(self.KEY)
            self._cache = {
                (k.decode() if isinstance(k, bytes) else k):
                (v.decode() if isinstance(v, bytes) else v)
                for k, v in raw.items()
            }
        except Exception as e:
            logger.warning(f"Failed to load glossary from Redis: {e}")
        return self._cache

    async def add(self, source_term: str, target_term: str) -> None:
        """Add or update a glossary entry."""
        key = source_term.strip().lower()
        val = target_term.strip()
        self._cache[key] = val
        if self.redis:
            try:
                await self.redis.hset(self.KEY, key, val)
            except Exception as e:
                logger.warning(f"Failed to save glossary entry to Redis: {e}")

    async def add_bulk(self, entries: Dict[str, str]) -> int:
        """Add multiple glossary entries at once. Returns count added."""
        clean = {k.strip().lower(): v.strip() for k, v in entries.items() if k.strip() and v.strip()}
        self._cache.update(clean)
        if self.redis and clean:
            try:
                await self.redis.hset(self.KEY, mapping=clean)
            except Exception as e:
                logger.warning(f"Failed to bulk save glossary to Redis: {e}")
        return len(clean)

    async def remove(self, source_term: str) -> bool:
        """Remove a glossary entry. Returns True if existed."""
        key = source_term.strip().lower()
        existed = key in self._cache
        self._cache.pop(key, None)
        if self.redis:
            try:
                await self.redis.hdel(self.KEY, key)
            except Exception as e:
                logger.warning(f"Failed to remove glossary entry from Redis: {e}")
        return existed

    async def get_all(self) -> Dict[str, str]:
        """Get all entries (cached)."""
        if not self._cache and self.redis:
            await self.load()
        return dict(self._cache)

    def get_relevant(self, text: str, source_lang: str, target_lang: str) -> Dict[str, str]:
        """Find glossary entries relevant to the input text.

        Checks both built-in and custom glossary, returns only terms
        that appear in the text.
        """
        text_lower = text.lower()
        relevant = {}

        # Pick the right built-in glossary direction
        if source_lang == "en" and target_lang == "es":
            builtin = AIRSOFT_GLOSSARY_EN_ES
        elif source_lang == "es" and target_lang == "en":
            builtin = AIRSOFT_GLOSSARY_ES_EN
        else:
            builtin = {}

        # Check built-in terms (longest first to avoid partial matches)
        for term in sorted(builtin.keys(), key=len, reverse=True):
            if term in text_lower:
                relevant[term] = builtin[term]

        # Check custom terms (override built-in if same key)
        for term, translation in self._cache.items():
            if term in text_lower:
                relevant[term] = translation

        return relevant


class TranslationPipeline:
    def __init__(self, llm_client, glossary_store: Optional[GlossaryStore] = None):
        self.llm_client = llm_client
        self.glossary = glossary_store or GlossaryStore()

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "es",
        progress_callback: Optional[Callable] = None,
        rag_context: Optional[str] = None,
    ) -> TranslationJob:
        """Translate a list of texts using the running LLM."""
        job = TranslationJob(
            job_id=uuid.uuid4().hex[:12],
            source_lang=source_lang,
            target_lang=target_lang,
            items_total=len(texts),
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Process in chunks of 5 texts per LLM call
            chunk_size = 5
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                translated = await self._translate_chunk(
                    chunk, source_lang, target_lang, rag_context=rag_context
                )
                job.results.extend(translated)
                job.items_processed = min(i + chunk_size, len(texts))
                job.log(f"Translated {job.items_processed}/{job.items_total}")

                if progress_callback:
                    progress_callback(job)

            job.status = "completed"
            job.log(f"COMPLETED: {len(job.results)} items translated")

        except Exception as e:
            job.status = "failed"
            job.log(f"FAILED: {str(e)[:200]}")
            logger.error(f"Translation failed: {e}")

        job.ended_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            progress_callback(job)
        return job

    def _build_glossary_prompt(self, texts: List[str], source_lang: str, target_lang: str) -> str:
        """Build a glossary section for the prompt based on terms found in the input."""
        combined_text = " ".join(texts)
        relevant = self.glossary.get_relevant(combined_text, source_lang, target_lang)

        if not relevant:
            return ""

        lines = [f"  - \"{term}\" → \"{translation}\"" for term, translation in relevant.items()]
        return (
            "\n\n## MANDATORY Terminology Glossary\n"
            "You MUST use these exact translations for the following terms. "
            "Do NOT deviate from this glossary:\n"
            + "\n".join(lines)
        )

    async def _translate_chunk(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        rag_context: Optional[str] = None,
    ) -> List[str]:
        """Single LLM call to translate a small batch of texts."""
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)

        system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
            source=source_name,
            target=target_name,
            target_name=target_name,
        )

        # Inject relevant glossary terms
        glossary_section = self._build_glossary_prompt(texts, source_lang, target_lang)
        if glossary_section:
            system_prompt += glossary_section

        if rag_context:
            system_prompt += f"\n\n## Reference: Similar products from our catalog\n{rag_context}"

        # Number the texts for clarity
        numbered = "\n\n".join(
            f"[{i+1}] {text}" for i, text in enumerate(texts)
        )

        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self.llm_client.models.list)
            model_id = models.data[0].id if models.data else None

            if not model_id:
                return texts  # fallback: return originals

            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate these {len(texts)} texts:\n\n{numbered}"},
                    ],
                    max_tokens=4096,
                    temperature=0.2,
                    timeout=300,
                ),
            )

            response_text = response.choices[0].message.content
            return self._parse_translations(response_text, len(texts), texts)

        except Exception as e:
            logger.warning(f"Translation chunk failed: {e}")
            return texts  # fallback

    @staticmethod
    def _parse_translations(response_text: str, expected_count: int, originals: List[str]) -> List[str]:
        """Parse LLM translation response into list of strings."""
        text = response_text.strip()

        # Try JSON array parse
        if "```" in text:
            lines = text.split("\n")
            inside = False
            json_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    json_lines.append(line)
            if json_lines:
                text = "\n".join(json_lines)

        try:
            import json
            data = json.loads(text)
            if isinstance(data, list) and len(data) == expected_count:
                return [str(item) for item in data]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to split by numbered markers [1], [2], etc.
        parts = re.split(r"\[\d+\]\s*", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == expected_count:
            return parts

        # Last resort: return originals
        logger.warning(f"Could not parse {expected_count} translations from LLM response")
        return originals

    # ── Spec Normalization ────────────────────────────────────────

    @staticmethod
    def normalize_specs(product: dict, target_units: str = "metric") -> dict:
        """Normalize units, brand names, and model numbers in a product spec dict."""
        result = dict(product)

        # Normalize brand
        brand = result.get("brand", "")
        if brand:
            result["brand"] = TranslationPipeline.normalize_brand(brand)

        # Normalize FPS/m/s
        fps = result.get("fps")
        if fps and target_units == "metric":
            result["velocity_ms"] = UNIT_CONVERSIONS["fps_to_ms"](fps)

        # Normalize weight
        weight = result.get("weight_grams")
        if weight and target_units == "metric":
            if weight > 0:
                result["weight_kg"] = round(weight / 1000, 2)

        return result

    @staticmethod
    def normalize_brand(brand: str) -> str:
        """Fuzzy match and normalize brand name."""
        brand_lower = brand.lower().strip()
        return BRAND_NORMALIZATIONS.get(brand_lower, brand)
