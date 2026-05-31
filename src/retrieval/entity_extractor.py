"""
Extração de entidades médicas para casos dermatológicos.

Usa scispaCy (en_ner_bc5cdr_md) para doenças e químicos/fármacos.
Para morfologia, localização anatômica e sintomas, aplica léxicos
curados de terminologia dermatológica — sem chamadas de LLM,
totalmente reproduzível e rastreável para a metodologia da tese.

Ontologia definida:
    disease    → diagnósticos e condições (ex: "psoriasis", "melanoma")
    chemical   → fármacos e tratamentos (ex: "methotrexate", "cyclosporine")
    morphology → lesões primárias e secundárias (ex: "plaque", "papule")
    location   → localização anatômica (ex: "scalp", "lower extremity")
    symptom    → sintomas clínicos (ex: "pruritus", "pain")
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()

# ---------------------------------------------------------------------------
# Léxicos dermatológicos curados
# ---------------------------------------------------------------------------

DISEASE_TERMS: set[str] = {
    # Doenças inflamatórias / imunomediadas
    "psoriasis", "eczema", "atopic dermatitis", "contact dermatitis",
    "seborrheic dermatitis", "rosacea", "lupus", "sarcoidosis",
    "dermatomyositis", "scleroderma", "morphea", "vitiligo",
    "lichen planus", "lichen sclerosus", "lichen striatus",
    "pemphigus", "pemphigoid", "bullous pemphigoid", "dermatitis herpetiformis",
    "erythema multiforme", "stevens johnson syndrome", "toxic epidermal necrolysis",
    "sweet syndrome", "pyoderma gangrenosum", "behcet disease",
    "pityriasis rosea", "pityriasis rubra pilaris", "pityriasis lichenoides",
    "granuloma annulare", "necrobiosis lipoidica", "erythema nodosum",
    "panniculitis", "vasculitis", "leukocytoclastic vasculitis",
    "urticaria", "angioedema", "mastocytosis",
    # Infecções
    "tinea", "tinea pedis", "tinea corporis", "tinea capitis", "tinea versicolor",
    "candidiasis", "candida", "onychomycosis",
    "herpes", "herpes simplex", "herpes zoster", "varicella",
    "molluscum contagiosum", "verruca", "wart", "condyloma",
    "impetigo", "cellulitis", "erysipelas", "folliculitis", "furuncle",
    "abscess", "necrotizing fasciitis",
    "scabies", "pediculosis",
    "syphilis", "gonorrhea", "hiv",
    # Neoplasias benignas
    "seborrheic keratosis", "actinic keratosis", "dermatofibroma",
    "lipoma", "epidermal cyst", "pilomatricoma", "trichoepithelioma",
    "syringoma", "cylindroma", "spiradenoma",
    "nevus", "melanocytic nevus", "spitz nevus", "blue nevus",
    "hemangioma", "pyogenic granuloma", "angiokeratoma",
    "xanthoma", "xanthelasma",
    # Neoplasias malignas
    "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
    "merkel cell carcinoma", "kaposi sarcoma", "dermatofibrosarcoma",
    "mycosis fungoides", "sezary syndrome", "lymphoma",
    "cutaneous t cell lymphoma", "cutaneous b cell lymphoma",
    "leukemia cutis",
    # Doenças do folículo / glândulas
    "acne", "acne vulgaris", "acne rosacea", "hidradenitis suppurativa",
    "alopecia", "alopecia areata", "androgenetic alopecia",
    "telogen effluvium", "trichotillomania",
    "hyperhidrosis",
    # Doenças sistêmicas com manifestação cutânea
    "diabetes", "amyloidosis", "porphyria", "calciphylaxis",
    "graft versus host disease", "gvhd",
    # Genodermatoses
    "ichthyosis", "darier disease", "hailey hailey disease",
    "epidermolysis bullosa", "neurofibromatosis",
    "tuberous sclerosis", "incontinentia pigmenti",
    # Fotodanos / outras
    "photodermatosis", "polymorphous light eruption",
    "drug eruption", "fixed drug eruption",
    "prurigo", "prurigo nodularis",
    "keloid", "hypertrophic scar",
}

CHEMICAL_TERMS: set[str] = {
    # Imunossupressores / biológicos
    "methotrexate", "cyclosporine", "azathioprine", "mycophenolate",
    "hydroxychloroquine", "dapsone", "thalidomide",
    # Corticosteroides
    "prednisone", "prednisolone", "methylprednisolone",
    "betamethasone", "clobetasol", "triamcinolone", "hydrocortisone",
    # Retinoides
    "isotretinoin", "acitretin", "tretinoin", "tazarotene",
    # Biológicos / inibidores
    "dupilumab", "secukinumab", "ixekizumab", "guselkumab",
    "adalimumab", "infliximab", "ustekinumab", "risankizumab",
    "omalizumab", "benralizumab",
    # Tópicos
    "tacrolimus", "pimecrolimus", "crisaborole",
    "calcipotriol", "calcitriol",
    # Antimicrobianos
    "doxycycline", "tetracycline", "minocycline", "clindamycin",
    "fluconazole", "itraconazole", "terbinafine", "griseofulvin",
    "acyclovir", "valacyclovir",
    # Quimioterápicos cutâneos
    "fluorouracil", "imiquimod", "ingenol mebutate",
    # Outros
    "colchicine", "pentoxifylline", "intravenous immunoglobulin", "ivig",
}

MORPHOLOGY_TERMS: set[str] = {
    # Lesões primárias
    "macule", "patch", "papule", "plaque", "nodule", "tumor",
    "vesicle", "bulla", "pustule", "wheal", "urticaria",
    # Lesões secundárias
    "erosion", "ulcer", "fissure", "atrophy", "scar", "keloid",
    "crust", "scale", "lichenification", "excoriation",
    # Características vasculares/pigmentares
    "erythema", "erythematous", "purpura", "petechiae", "telangiectasia",
    "hyperpigmentation", "hypopigmentation", "depigmentation",
    "pigmentation", "dyspigmentation",
    # Estruturas foliculares/outras
    "comedone", "milia", "cyst", "abscess",
    "induration", "sclerosis", "fibrosis",
    "desquamation", "hyperkeratosis", "acanthosis",
}

LOCATION_TERMS: set[str] = {
    # Cabeça e pescoço
    "scalp", "face", "forehead", "cheek", "chin", "nose", "ear", "neck",
    "eyelid", "periorbital", "lip", "oral", "perioral", "mucosa", "mucosal",
    # Tronco
    "chest", "trunk", "back", "abdomen", "flank", "shoulder",
    # Membros superiores
    "arm", "forearm", "elbow", "wrist", "hand", "finger",
    "palm", "palmar", "periungual",
    # Membros inferiores
    "leg", "thigh", "knee", "shin", "ankle", "foot", "toe",
    "plantar", "sole",
    # Regiões especiais
    "groin", "inguinal", "axilla", "axillary",
    "perianal", "genital", "vulvar", "penile", "scrotal",
    "nail", "nails", "subungual",
    # Descritores lateralidade/extensão
    "bilateral", "unilateral", "generalized", "localized", "diffuse",
    "extremities", "lower extremity", "upper extremity",
    "acral", "perioral", "intertriginous",
}

SYMPTOM_TERMS: set[str] = {
    "pruritus", "pruritic", "itching", "itch",
    "pain", "painful", "burning", "tenderness", "tender",
    "bleeding", "hemorrhage",
    "swelling", "edema",
    "numbness", "dysesthesia", "paresthesia",
    "fever", "fatigue", "malaise",
    "alopecia", "hair loss",
}

# Bigrams compostos relevantes (verificados como substrings também)
LOCATION_BIGRAMS: set[str] = {
    "lower extremity", "upper extremity", "lower leg",
    "hair loss",
}


# ---------------------------------------------------------------------------
# Extrator
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extrai entidades médicas de textos de casos clínicos.

    Combina NER com scispaCy (diseases + chemicals) e matching léxico
    curado (morphology, location, symptoms). Produz saída determinística
    e cacheável.
    """

    def __init__(
        self,
        use_scispacy: bool = True,
        spacy_model: str = "en_ner_bc5cdr_md",
    ) -> None:
        self.nlp = None
        self.use_scispacy = use_scispacy

        if use_scispacy:
            try:
                import spacy  # noqa: PLC0415
                self.nlp = spacy.load(spacy_model)
                console.print(f"[green]scispaCy carregado: {spacy_model}[/]")
            except (ImportError, OSError) as exc:
                console.print(
                    f"[yellow]scispaCy indisponível ({exc}). "
                    "Usando apenas léxicos curados.[/]"
                )
                self.use_scispacy = False

    # ------------------------------------------------------------------
    def extract(self, text: str) -> dict[str, list[str]]:
        """Extrai entidades de um texto. Retorna dict por tipo."""
        entities: dict[str, list[str]] = {
            "disease": [],
            "chemical": [],
            "morphology": [],
            "location": [],
            "symptom": [],
        }

        # --- NER com scispaCy (disease + chemical) ---
        if self.use_scispacy and self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                val = ent.text.lower().strip()
                if ent.label_ == "DISEASE" and len(val) >= 3:
                    entities["disease"].append(val)
                elif ent.label_ == "CHEMICAL" and len(val) >= 3:
                    entities["chemical"].append(val)

        # --- Matching léxico (morphology, location, symptom) ---
        text_lower = text.lower()
        tokens = set(re.findall(r"\b[a-z]{3,}\b", text_lower))

        # Bigrams
        words = re.findall(r"\b[a-z]{3,}\b", text_lower)
        bigrams = {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}

        entities["morphology"] = sorted(MORPHOLOGY_TERMS & tokens)
        entities["symptom"] = sorted(SYMPTOM_TERMS & tokens)

        loc_hits = LOCATION_TERMS & tokens
        loc_hits |= LOCATION_BIGRAMS & bigrams
        entities["location"] = sorted(loc_hits)

        # Léxicos curados para disease e chemical (usados sempre,
        # complementando ou substituindo o scispaCy)
        disease_hits = DISEASE_TERMS & tokens
        disease_hits |= {t for t in DISEASE_TERMS if " " in t and t in text_lower}
        if not self.use_scispacy:
            entities["disease"] = sorted(disease_hits)
        else:
            # Complementa o NER com o léxico (NER pode perder variantes)
            entities["disease"] = sorted(set(entities["disease"]) | disease_hits)

        chem_hits = CHEMICAL_TERMS & tokens
        chem_hits |= {t for t in CHEMICAL_TERMS if " " in t and t in text_lower}
        if not self.use_scispacy:
            entities["chemical"] = sorted(chem_hits)
        else:
            entities["chemical"] = sorted(set(entities["chemical"]) | chem_hits)

        # Deduplica cada tipo
        for key in entities:
            entities[key] = sorted(set(entities[key]))

        return entities

    # ------------------------------------------------------------------
    def extract_batch(
        self,
        texts: Sequence[str],
        ids: Sequence[str],
        cache_path: Optional[Path | str] = None,
    ) -> dict[str, dict[str, list[str]]]:
        """Extrai entidades em batch com cache JSON opcional.

        Se cache_path existir e tiver o mesmo número de IDs, carrega
        direto sem reprocessar.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                console.print(f"[green]Carregando entidades do cache: {cache_path}[/]")
                with open(cache_path) as f:
                    cached = json.load(f)
                if len(cached) == len(ids):
                    return cached
                console.print("[yellow]Cache com tamanho diferente. Reprocessando.[/]")

        results: dict[str, dict[str, list[str]]] = {}

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Extraindo entidades...", total=len(texts)
            )
            for doc_id, text in zip(ids, texts):
                results[doc_id] = self.extract(text)
                progress.advance(task)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(results, f)
            console.print(f"[green]Entidades salvas em {cache_path}[/]")

        # Estatísticas resumidas
        n_with_disease = sum(
            1 for v in results.values() if v.get("disease")
        )
        n_with_morphology = sum(
            1 for v in results.values() if v.get("morphology")
        )
        console.print(
            f"  Casos com disease: {n_with_disease}/{len(results)}  |  "
            f"com morphology: {n_with_morphology}/{len(results)}"
        )

        return results
