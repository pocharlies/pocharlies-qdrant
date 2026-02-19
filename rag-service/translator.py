"""
Translation Pipeline for RAG Pipeline
LLM-based translation (EN→ES) and spec normalization for product data.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


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


class TranslationPipeline:
    def __init__(self, llm_client):
        self.llm_client = llm_client

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

        if rag_context:
            system_prompt += f"\n\nHere are similar products from our catalog for terminology reference:\n{rag_context}"

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
