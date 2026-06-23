"""
Extração de entidades médicas local — sem API, sem GPU.

Melhorias sobre o EntityExtractor original:
  1. Lemmatização de variantes morfológicas (adjectivos → substantivos,
     plurais irregulares, formas "ous/al/ic/ed" → forma base)
  2. Léxicos expandidos com mais termos dermatológicos
  3. Matching de bigrams e trigrams para termos compostos
  4. scispaCy NER quando disponível, léxico curado como fallback

Saída: data/processed/entities_lemmatized.json
  {case_id: {disease: [...], morphology: [...], location: [...], symptom: [...], chemical: []}}

Uso:
    python scripts/extract_entities_local.py --config configs/graph_lemmatized.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.prepare_eval import DataConfig, SplitConfig, prepare_dataset

console = Console()

# ---------------------------------------------------------------------------
# Tabela de normalização: variante → forma canónica
# Cobre as formas adjectivais, plurais e participiais mais comuns
# em dermatologia clínica.
# ---------------------------------------------------------------------------
NORMALIZE: dict[str, str] = {
    # Morfologia — adjectivos/participiais → substantivo
    "erythematous": "erythema",
    "erythemic":    "erythema",
    "erythemas":    "erythema",
    "papular":      "papule",
    "papules":      "papule",
    "papulosquamous": "papule",
    "vesicular":    "vesicle",
    "vesicles":     "vesicle",
    "bullous":      "bulla",
    "bullae":       "bulla",
    "pustular":     "pustule",
    "pustules":     "pustule",
    "nodular":      "nodule",
    "nodules":      "nodule",
    "plaques":      "plaque",
    "macules":      "macule",
    "patches":      "patch",
    "wheals":       "wheal",
    "ulcerative":   "ulcer",
    "ulcerated":    "ulcer",
    "ulcers":       "ulcer",
    "erosive":      "erosion",
    "erosions":     "erosion",
    "fissures":     "fissure",
    "crusted":      "crust",
    "crusts":       "crust",
    "crusty":       "crust",
    "scaly":        "scale",
    "scales":       "scale",
    "scaling":      "scale",
    "squamous":     "scale",
    "desquamating": "desquamation",
    "hyperkeratotic": "hyperkeratosis",
    "lichenified":  "lichenification",
    "excoriated":   "excoriation",
    "excoriations": "excoriation",
    "atrophic":     "atrophy",
    "indurated":    "induration",
    "fibrotic":     "fibrosis",
    "sclerotic":    "sclerosis",
    "comedones":    "comedone",
    "comedonal":    "comedone",
    "milia":        "milium",
    "cysts":        "cyst",
    "abscesses":    "abscess",
    "telangiectatic": "telangiectasia",
    "telangiectases": "telangiectasia",
    "purpuric":     "purpura",
    "petechial":    "petechiae",
    "hyperpigmented": "hyperpigmentation",
    "hypopigmented":  "hypopigmentation",
    "depigmented":    "depigmentation",
    "pigmented":    "pigmentation",
    "dyspigmented": "dyspigmentation",
    "acanthotic":   "acanthosis",
    # Sintomas
    "pruritic":     "pruritus",
    "itchy":        "pruritus",
    "painful":      "pain",
    "tender":       "tenderness",
    "edematous":    "edema",
    "swollen":      "swelling",
    "bleeding":     "hemorrhage",
    "burning":      "burning",
    "alopecic":     "alopecia",
    # Localização — plurais
    "nails":        "nail",
    "fingers":      "finger",
    "toes":         "toe",
    "hands":        "hand",
    "feet":         "foot",
    "legs":         "leg",
    "arms":         "arm",
    "ears":         "ear",
    "lips":         "lip",
    "eyelids":      "eyelid",
    "cheeks":       "cheek",
    "palms":        "palm",
    "soles":        "sole",
    "genitals":     "genital",
    "axillae":      "axilla",
}

# ---------------------------------------------------------------------------
# Léxicos expandidos
# ---------------------------------------------------------------------------

DISEASE_TERMS: set[str] = {
    # Inflamatórias / imunomediadas
    "psoriasis", "eczema", "atopic dermatitis", "contact dermatitis",
    "allergic contact dermatitis", "irritant contact dermatitis",
    "seborrheic dermatitis", "nummular dermatitis", "stasis dermatitis",
    "rosacea", "lupus", "lupus erythematosus", "discoid lupus",
    "systemic lupus erythematosus", "sle",
    "sarcoidosis", "dermatomyositis", "scleroderma", "morphea",
    "vitiligo", "lichen planus", "lichen sclerosus", "lichen striatus",
    "lichen nitidus", "lichen simplex chronicus",
    "pemphigus", "pemphigus vulgaris", "pemphigus foliaceus",
    "pemphigoid", "bullous pemphigoid", "mucous membrane pemphigoid",
    "dermatitis herpetiformis", "linear iga disease",
    "erythema multiforme", "stevens johnson syndrome", "toxic epidermal necrolysis",
    "sweet syndrome", "pyoderma gangrenosum", "behcet disease",
    "pityriasis rosea", "pityriasis rubra pilaris", "pityriasis lichenoides",
    "granuloma annulare", "necrobiosis lipoidica", "erythema nodosum",
    "panniculitis", "vasculitis", "leukocytoclastic vasculitis",
    "polyarteritis nodosa", "urticarial vasculitis",
    "urticaria", "chronic urticaria", "angioedema", "mastocytosis",
    "graft versus host disease", "gvhd",
    # Infecções bacterianas
    "impetigo", "cellulitis", "erysipelas", "folliculitis",
    "furuncle", "carbuncle", "abscess", "necrotizing fasciitis",
    "erythrasma", "pitted keratolysis",
    # Infecções fúngicas
    "tinea", "tinea pedis", "tinea corporis", "tinea capitis",
    "tinea versicolor", "tinea cruris", "tinea unguium",
    "candidiasis", "candida", "onychomycosis",
    "aspergillosis", "sporotrichosis",
    # Infecções virais
    "herpes", "herpes simplex", "herpes zoster", "varicella",
    "molluscum contagiosum", "verruca", "wart", "condyloma",
    "hand foot mouth", "hand foot and mouth disease",
    "parvovirus", "roseola", "rubeola", "rubella",
    # Parasitas
    "scabies", "pediculosis", "cutaneous larva migrans",
    # IST
    "syphilis", "gonorrhea", "hiv", "kaposi sarcoma",
    # Neoplasias benignas
    "seborrheic keratosis", "actinic keratosis", "dermatofibroma",
    "lipoma", "epidermal cyst", "pilomatricoma", "trichoepithelioma",
    "syringoma", "cylindroma", "spiradenoma",
    "nevus", "melanocytic nevus", "spitz nevus", "blue nevus",
    "congenital nevus", "dysplastic nevus",
    "hemangioma", "pyogenic granuloma", "angiokeratoma",
    "cherry angioma", "port wine stain", "venous malformation",
    "xanthoma", "xanthelasma", "fibroma",
    # Neoplasias malignas
    "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
    "merkel cell carcinoma", "dermatofibrosarcoma", "angiosarcoma",
    "mycosis fungoides", "sezary syndrome", "lymphoma",
    "cutaneous t cell lymphoma", "cutaneous b cell lymphoma",
    "leukemia cutis",
    # Cabelo / glândulas
    "acne", "acne vulgaris", "nodulocystic acne", "cystic acne",
    "hidradenitis suppurativa", "rosacea",
    "alopecia", "alopecia areata", "androgenetic alopecia",
    "telogen effluvium", "anagen effluvium", "trichotillomania",
    "hyperhidrosis",
    # Doenças sistêmicas com manifestação cutânea
    "diabetes", "amyloidosis", "porphyria", "calciphylaxis",
    "neurofibromatosis", "tuberous sclerosis", "incontinentia pigmenti",
    # Fotodanos / outros
    "photodermatosis", "polymorphous light eruption", "solar urticaria",
    "drug eruption", "fixed drug eruption", "lichenoid drug reaction",
    "prurigo", "prurigo nodularis", "nodular prurigo",
    "keloid", "hypertrophic scar",
    "ichthyosis", "darier disease", "hailey hailey disease",
    "epidermolysis bullosa",
    # Doenças do tecido conjuntivo
    "rheumatoid arthritis", "psoriatic arthritis",
    "mixed connective tissue disease",
}

CHEMICAL_TERMS: set[str] = {
    # Imunossupressores
    "methotrexate", "cyclosporine", "azathioprine", "mycophenolate",
    "hydroxychloroquine", "chloroquine", "dapsone", "thalidomide",
    "colchicine", "pentoxifylline",
    # Corticosteroides sistémicos
    "prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
    # Corticosteroides tópicos
    "betamethasone", "clobetasol", "triamcinolone", "hydrocortisone",
    "mometasone", "fluocinonide", "fluocinolone", "desonide",
    "halobetasol", "desoximetasone",
    # Retinoides
    "isotretinoin", "acitretin", "tretinoin", "tazarotene", "adapalene",
    "bexarotene", "alitretinoin",
    # Biológicos — anti-IL
    "dupilumab", "tralokinumab", "lebrikizumab",
    "secukinumab", "ixekizumab", "bimekizumab",
    "guselkumab", "risankizumab", "tildrakizumab",
    "ustekinumab",
    # Biológicos — anti-TNF
    "adalimumab", "infliximab", "etanercept", "certolizumab",
    # Biológicos — outros
    "omalizumab", "benralizumab", "mepolizumab",
    "rituximab", "intravenous immunoglobulin", "ivig",
    # Inibidores JAK
    "tofacitinib", "baricitinib", "upadacitinib", "ruxolitinib",
    "abrocitinib",
    # Tópicos calcineurina / outros
    "tacrolimus", "pimecrolimus", "crisaborole", "ruxolitinib",
    "calcipotriol", "calcitriol", "calcipotriene",
    # Antimicrobianos
    "doxycycline", "tetracycline", "minocycline", "clindamycin",
    "erythromycin", "azithromycin", "amoxicillin", "cephalexin",
    "trimethoprim", "sulfamethoxazole",
    "fluconazole", "itraconazole", "terbinafine", "griseofulvin",
    "voriconazole", "ketoconazole",
    "acyclovir", "valacyclovir", "famciclovir",
    # Quimioterápicos cutâneos
    "fluorouracil", "imiquimod", "ingenol mebutate",
    "vismodegib", "cemiplimab", "pembrolizumab", "nivolumab",
    # Outros
    "ivermectin", "permethrin", "benzyl benzoate",
    "zinc oxide", "salicylic acid", "benzoyl peroxide",
    "coal tar", "anthralin",
}

MORPHOLOGY_TERMS: set[str] = {
    # Lesões primárias
    "macule", "patch", "papule", "plaque", "nodule", "tumor",
    "vesicle", "bulla", "pustule", "wheal", "urticaria",
    # Lesões secundárias
    "erosion", "ulcer", "fissure", "atrophy", "scar", "keloid",
    "crust", "scale", "lichenification", "excoriation",
    "desquamation", "hyperkeratosis", "acanthosis",
    # Vascular / pigmentar
    "erythema", "purpura", "petechiae", "telangiectasia",
    "hyperpigmentation", "hypopigmentation", "depigmentation",
    "pigmentation", "dyspigmentation",
    # Folicular
    "comedone", "milium", "cyst", "abscess",
    "induration", "sclerosis", "fibrosis",
    # Outros descritores morfológicos
    "vegetation", "verrucous", "annular", "linear", "serpiginous",
    "reticulate", "targetoid", "umbilicated",
    "crusted", "scabbed", "eroded", "excoriated",
}

LOCATION_TERMS: set[str] = {
    # Cabeça
    "scalp", "face", "forehead", "cheek", "chin", "nose",
    "ear", "neck", "eyelid", "periorbital", "lip", "oral",
    "perioral", "mucosa", "mucosal", "buccal", "gingival",
    "conjunctival", "nasal",
    # Tronco
    "chest", "trunk", "back", "abdomen", "flank", "shoulder",
    "breast", "nipple", "areola",
    # Membros superiores
    "arm", "forearm", "elbow", "wrist", "hand", "finger",
    "palm", "palmar", "periungual", "interdigital",
    # Membros inferiores
    "leg", "thigh", "knee", "shin", "ankle", "foot", "toe",
    "plantar", "sole", "popliteal",
    # Regiões especiais
    "groin", "inguinal", "axilla", "axillary",
    "perianal", "genital", "vulvar", "penile", "scrotal",
    "nail", "subungual",
    # Descritores
    "bilateral", "unilateral", "generalized", "localized", "diffuse",
    "extremities", "lower extremity", "upper extremity",
    "acral", "intertriginous", "flexural", "extensor",
    "photoexposed", "sun-exposed",
}

SYMPTOM_TERMS: set[str] = {
    "pruritus", "pruritic", "itching", "itch",
    "pain", "painful", "burning", "tenderness", "tender",
    "bleeding", "hemorrhage",
    "swelling", "edema",
    "numbness", "dysesthesia", "paresthesia",
    "fever", "fatigue", "malaise",
    "alopecia", "hair loss",
    "photosensitivity", "photophobia",
    "xerosis", "dryness", "dry skin",
}

LOCATION_MULTIWORD: set[str] = {
    "lower extremity", "upper extremity", "lower leg", "lower limb",
    "upper limb", "hair loss", "sun exposed", "photo exposed",
    "palms soles", "hand foot",
}


def normalize_token(tok: str) -> str:
    """Aplica normalização de variante morfológica."""
    return NORMALIZE.get(tok, tok)


def extract_entities(text: str, nlp=None) -> dict[str, list[str]]:
    text_lower = text.lower()

    # Tokeniza e normaliza
    raw_tokens = re.findall(r"\b[a-z]{3,}\b", text_lower)
    tokens     = {normalize_token(t) for t in raw_tokens}
    raw_set    = set(raw_tokens)

    # Bigrams (originais e normalizados)
    bigrams: set[str] = set()
    for i in range(len(raw_tokens) - 1):
        bigrams.add(f"{raw_tokens[i]} {raw_tokens[i+1]}")
        bigrams.add(f"{normalize_token(raw_tokens[i])} {normalize_token(raw_tokens[i+1])}")

    # Trigrams para termos compostos de 3 palavras
    trigrams: set[str] = set()
    for i in range(len(raw_tokens) - 2):
        trigrams.add(f"{raw_tokens[i]} {raw_tokens[i+1]} {raw_tokens[i+2]}")

    entities: dict[str, list[str]] = {
        "disease": [], "chemical": [], "morphology": [],
        "location": [], "symptom": [],
    }

    # --- NER com scispaCy ---
    if nlp is not None:
        doc = nlp(text[:2000])   # limita para velocidade
        for ent in doc.ents:
            val = normalize_token(ent.text.lower().strip())
            if len(val) < 3:
                continue
            if ent.label_ == "DISEASE":
                entities["disease"].append(val)
            elif ent.label_ == "CHEMICAL":
                entities["chemical"].append(val)

    # --- Léxico: disease ---
    hits_d = DISEASE_TERMS & tokens
    hits_d |= DISEASE_TERMS & raw_set
    hits_d |= {t for t in DISEASE_TERMS if " " in t and (t in bigrams or t in text_lower)}
    entities["disease"] = sorted(set(entities["disease"]) | hits_d)

    # --- Léxico: chemical ---
    hits_c = CHEMICAL_TERMS & tokens
    hits_c |= CHEMICAL_TERMS & raw_set
    hits_c |= {t for t in CHEMICAL_TERMS if " " in t and (t in bigrams or t in text_lower)}
    entities["chemical"] = sorted(set(entities["chemical"]) | hits_c)

    # --- Léxico: morphology ---
    entities["morphology"] = sorted(MORPHOLOGY_TERMS & tokens | MORPHOLOGY_TERMS & raw_set)

    # --- Léxico: location ---
    loc = LOCATION_TERMS & tokens | LOCATION_TERMS & raw_set
    loc |= LOCATION_MULTIWORD & bigrams | LOCATION_MULTIWORD & trigrams
    entities["location"] = sorted(loc)

    # --- Léxico: symptom ---
    symp = SYMPTOM_TERMS & tokens | SYMPTOM_TERMS & raw_set
    symp |= {t for t in SYMPTOM_TERMS if " " in t and t in text_lower}
    entities["symptom"] = sorted(symp)

    # Deduplica
    for key in entities:
        entities[key] = sorted(set(entities[key]))

    return entities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/graph_lemmatized.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-extrai mesmo que o cache exista")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_path = Path(cfg["entity_extraction"]["cache_path"])

    if output_path.exists() and not args.force:
        console.print(f"[green]{output_path} já existe. Usa --force para re-extrair.[/]")
        return

    # Tenta carregar scispaCy
    nlp = None
    spacy_model = cfg["entity_extraction"].get("spacy_model", "en_ner_bc5cdr_md")
    if cfg["entity_extraction"].get("use_scispacy", False):
        try:
            import spacy
            nlp = spacy.load(spacy_model)
            console.print(f"[green]scispaCy: {spacy_model}[/]")
        except Exception as e:
            console.print(f"[yellow]scispaCy indisponível ({e}). Usando léxico curado.[/]")

    # Carrega corpus completo
    import pandas as pd
    df = pd.read_csv(cfg["data"]["derm_csv"])
    id_col   = cfg["data"]["id_col"]
    text_col = cfg["data"]["text_col"]
    df = df.dropna(subset=[id_col, text_col]).copy()
    df[id_col] = df[id_col].astype(str)

    ids   = df[id_col].tolist()
    texts = df[text_col].astype(str).tolist()
    console.print(f"Corpus: {len(ids)} casos")

    results: dict[str, dict] = {}

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extraindo entidades...", total=len(ids))
        for doc_id, text in zip(ids, texts):
            results[doc_id] = extract_entities(text, nlp=nlp)
            progress.advance(task)

    # Estatísticas
    n = len(results)
    console.rule("Cobertura")
    for key in ("disease", "morphology", "location", "symptom", "chemical"):
        has = sum(1 for e in results.values() if e.get(key))
        avg = sum(len(e.get(key, [])) for e in results.values()) / n
        console.print(f"  {key:<12}: {has:>5}/{n} ({has/n*100:.0f}%)  média {avg:.1f}/caso")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"\n[green bold]Entidades salvas em {output_path}[/]")


if __name__ == "__main__":
    main()
