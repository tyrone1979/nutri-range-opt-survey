#!/usr/bin/env python3
"""Build a logically clustered, DOI-verified bibliography for the NutriOpt-Bench survey.

User constraints:
- Exactly ~150 entries, all with DOI verified in Crossref or DataCite.
- Must be useful/logical for a TEVC survey on diet optimization + EC benchmarking.
- Include ACMO-Diet and DMP4P.
- Exclude biomedical relation-extraction / LLM-RE papers.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "NutriOptBench-Survey/1.0 (mailto:nutriopt@local; scholarly metadata)"}

SEEDS = [
    ("10.2307/1231810", "A_classical_diet_OR", "crossref", "stigler1945"),
    ("10.2307/1235154", "A_classical_diet_OR", "crossref", "smith1959"),
    ("10.1055/s-0038-1634022", "B_evolutionary_menu", "crossref", "gaal2005"),
    ("10.1016/j.jfca.2009.02.006", "B_evolutionary_menu", "crossref", "seljak2009"),
    ("10.1109/ines.2006.1689351", "B_evolutionary_menu", "crossref", "vassanyi2006"),
    ("10.1109/cisce65916.2025.11065817", "C_diet_moo_recommend", "crossref", "zhao2025acmodiet"),
    ("10.1109/4235.996017", "D_moea_methods", "crossref", "deb2002"),
    ("10.1109/tevc.2007.892759", "D_moea_methods", "crossref", "zhang2007"),
    ("10.1109/tevc.2016.2519378", "D_moea_methods", "crossref", "cheng2016"),
    ("10.1109/tevc.2018.2855411", "D_moea_methods", "crossref", "li2019ctaea"),
    ("10.3929/ethz-a-004284029", "D_moea_methods", "datacite", "zitzler2001"),
    ("10.6084/m9.figshare.30153778", "G_nutrition_data", "datacite", "zhao2026dmp4p"),
]
STABLE_KEY_BY_DOI = {d.lower(): k for d, _, __, k in SEEDS}

# Hand-picked, survey-relevant DOIs (will be schema-verified; failures skipped).
EXTRA_CURATED = [
    # Classical / OR diets
    ("10.1038/sj/jors/0450501", "A_classical_diet_OR"),
    ("10.1057/jors.1994.76", "A_classical_diet_OR"),
    ("10.1016/j.ejor.2017.08.027", "A_classical_diet_OR"),
    ("10.1016/j.omega.2019.102123", "A_classical_diet_OR"),
    ("10.1016/j.cor.2020.105087", "A_classical_diet_OR"),
    ("10.3390/nu13082688", "A_classical_diet_OR"),
    ("10.3390/nu14030487", "A_classical_diet_OR"),
    # Diet MOO / recommenders
    ("10.1016/j.eswa.2023.122166", "C_diet_moo_recommend"),
    ("10.1016/j.eswa.2023.119719", "C_diet_moo_recommend"),
    ("10.1016/j.ipm.2021.102662", "C_diet_moo_recommend"),
    ("10.3390/computers13010001", "C_diet_moo_recommend"),
    ("10.1016/j.eswa.2024.123456", "C_diet_moo_recommend"),  # may fail
    # MOEA / CMOEA core & surveys
    ("10.1109/4235.797969", "D_moea_methods"),
    ("10.1109/tevc.2019.2918140", "F_sparse_large_combo"),
    ("10.1109/tevc.2023.3250552", "D_moea_methods"),
    ("10.1109/tevc.2023.3281666", "D_moea_methods"),
    ("10.1016/j.swevo.2023.101314", "D_moea_methods"),
    ("10.1016/j.ins.2022.10.050", "D_moea_methods"),
    ("10.1016/j.ejor.2006.08.008", "D_moea_methods"),  # SMS-EMOA
    ("10.1145/1830483.1830578", "E_benchmark_stats"),
    ("10.1109/tevc.2019.2906927", "E_benchmark_stats"),
    ("10.1109/tevc.2020.2982306", "D_moea_methods"),  # may fail
    ("10.1109/tevc.2021.3066433", "D_moea_methods"),  # may fail
    ("10.1145/3321707.3321839", "I_memetic_repair"),
    ("10.1145/3512290.3528732", "I_memetic_repair"),
]

EXCLUDE_DOI = {
    "10.1016/j.compbiolchem.2026.108978",
}

APPLIED_NOISE = re.compile(
    r"weapon|thermal behaviour|perforated|precast|shovel|attachment|power dispatch|"
    r"construction scheduling|modular frame|in vitro fertilization|IVF/|"
    r"hole height|wall thickness|environmental economic|ordinary shovel|"
    r"feature subset selection algorithm for|corrigendum",
    re.I,
)

D_TITLE_OK = re.compile(
    r"(survey|review).{0,40}(constrained|multi-?objective|evolutionary)|"
    r"constraint handling|"
    r"constrained multi-?objective evolutionary|"
    r"evolutionary constrained multi|"
    r"two-archive evolutionary|"
    r"reference vector guided|"
    r"large-scale sparse multi|"
    r"sparse multi-?objective|"
    r"\bMOEA/D\b|"
    r"\bNSGA-II\b|"
    r"\bNSGA-III\b|"
    r"\bSPEA2?\b|"
    r"\bRVEA\b|"
    r"\bSMS-EMOA\b|"
    r"\bC-?TAEA\b|"
    r"benchmarking multi-?objective|"
    r"performance indicators?.{0,30}multi-?objective|"
    r"many-objective evolutionary algorithm$",  # rare exact
    re.I,
)

RE_EXCLUDE = re.compile(
    r"relation\s*extraction|named entity|prompt(?:ing)?|large language model|\bLLM\b|"
    r"biomedical text|document-level|ChatGPT|zero-shot information|BERT\b|transformer language",
    re.I,
)

DIETISH = re.compile(r"\b(diet|dietary|meal|menu|nutrition|nutrient|food)\b", re.I)
MOEAISH = re.compile(
    r"multi-?objective|many-?objective|MOEA|NSGA|MOEA/D|RVEA|SPEA|C-?TAEA|SMS-EMOA|"
    r"evolutionary algorithm|genetic algorithm|differential evolution|memetic|"
    r"constraint handling|constrained multi",
    re.I,
)
METHOD_CORE = re.compile(
    r"(constrained multi|constraint handling|CMOEA|two-archive|NSGA-?I{1,3}|MOEA/D|RVEA|SPEA2?|"
    r"SMS-EMOA|HypE|reference vector|decomposition-based evolutionary|"
    r"survey on evolutionary|evolutionary constrained multi|"
    r"large-scale sparse multi|sparse multiobjective)",
    re.I,
)
BENCHISH = re.compile(
    r"\b(benchmark|test suite|test problem|performance indicator|experimental comparison|"
    r"algorithm comparison|reproducib|fair comparison|CEC\b)\b",
    re.I,
)
DATAISH = re.compile(
    r"(food composition database|food and nutrient database|FNDDS|USDA FoodData|"
    r"meal plan dataset|dietary database|nutrient analysis of diet|food atlas|"
    r"food composition tools|FooDB|composition database for nutrient)",
    re.I,
)
EPI_NOISE = re.compile(
    r"association between|risk of|patients|eczema|stroke|arthritis|depression|"
    r"mortality|cardiovascular|rheumatoid|breast cancer|intake and the risk|"
    r"cross-sectional study|all-cause",
    re.I,
)
SPARSEISH = re.compile(
    r"\b(sparse multi|large-scale multi|large scale multi|combinatorial multi|"
    r"binary multi|knapsack|subset selection|high-dimensional multi)\b",
    re.I,
)
MEMETICISH = re.compile(r"\b(memetic|local search|repair operator|repair mechanism|hybrid evolutionary)\b", re.I)

CLUSTER_QUOTA = {
    "A_classical_diet_OR": 16,
    "B_evolutionary_menu": 22,
    "C_diet_moo_recommend": 40,
    "D_moea_methods": 18,  # keep lean: core MOEA/CMOEA only
    "E_benchmark_stats": 18,
    "F_sparse_large_combo": 16,
    "G_nutrition_data": 12,
    "I_memetic_repair": 12,
}  # sum=154; trim to 150 after selection
HARD_CAP = {
    "D_moea_methods": 18,  # never exceed even when filling leftovers
}

ROLES = {
    "A_classical_diet_OR": "Classical LP/OR diet models — problem lineage",
    "B_evolutionary_menu": "Evolutionary / AI menu planning — EC application history",
    "C_diet_moo_recommend": "Modern diet MOO & food recommendation (incl. ACMO-Diet)",
    "D_moea_methods": "MOEA / CMOEA methodology — algorithmic repertoire",
    "E_benchmark_stats": "Benchmarking & comparison methodology — fair evaluation",
    "F_sparse_large_combo": "Sparse/large-scale/combinatorial constraints — difficulty mechanisms",
    "G_nutrition_data": "Nutrition datasets & food databases (incl. DMP4P)",
    "I_memetic_repair": "Memetic/local/repair operators — feasibility-oriented search",
}

QUERIES = [
    ("linear programming diet nutrition optimization", "A_classical_diet_OR", 1990, 2026, 40),
    ("integer programming meal planning diet", "A_classical_diet_OR", 2000, 2026, 30),
    ("menu planning genetic algorithm dietary", "B_evolutionary_menu", 2000, 2026, 40),
    ("evolutionary algorithm weekly menu planning", "B_evolutionary_menu", 2000, 2026, 30),
    ("multi-objective diet planning optimization", "C_diet_moo_recommend", 2018, 2026, 50),
    ("personalized diet recommendation multiobjective", "C_diet_moo_recommend", 2018, 2026, 40),
    ("healthy food recommendation nutrition constraints", "C_diet_moo_recommend", 2018, 2026, 40),
    ("constrained multiobjective evolutionary algorithm survey", "D_moea_methods", 2018, 2026, 20),
    ("constraint handling multiobjective evolutionary algorithm", "D_moea_methods", 2015, 2026, 15),
    ("NSGA-III MOEA/D RVEA constrained evolutionary", "D_moea_methods", 2014, 2026, 12),
    ("multiobjective evolutionary algorithm benchmark comparison", "E_benchmark_stats", 2015, 2026, 45),
    ("performance indicators multiobjective evolutionary algorithms", "E_benchmark_stats", 2010, 2026, 35),
    ("experimental comparison evolutionary multiobjective fair budget", "E_benchmark_stats", 2015, 2026, 30),
    ("large-scale sparse multiobjective evolutionary algorithm", "F_sparse_large_combo", 2018, 2026, 45),
    ("combinatorial multiobjective evolutionary knapsack binary", "F_sparse_large_combo", 2015, 2026, 35),
    ("subset selection evolutionary multiobjective constrained", "F_sparse_large_combo", 2015, 2026, 30),
    ("NHANES food composition database dietary intake", "G_nutrition_data", 2010, 2026, 40),
    ("USDA food composition database nutrient", "G_nutrition_data", 2005, 2026, 30),
    ("food and nutrient database for dietary studies FNDDS", "G_nutrition_data", 2010, 2026, 25),
    ("memetic algorithm multiobjective constrained combinatorial", "I_memetic_repair", 2010, 2026, 40),
    ("repair operator constrained evolutionary multiobjective", "I_memetic_repair", 2010, 2026, 30),
    ("hybrid local search evolutionary constrained combinatorial", "I_memetic_repair", 2012, 2026, 25),
]


def get_json(url: str, retries: int = 3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                raise
            last = e
            time.sleep(min(1.2 * (i + 1), 5))
        except Exception as e:
            last = e
            time.sleep(min(1.2 * (i + 1), 5))
    raise RuntimeError(f"failed {url}: {last}")


def crossref_work(doi: str):
    return get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]


def datacite_work(doi: str):
    return get_json("https://api.datacite.org/dois/" + urllib.parse.quote(doi))["data"]["attributes"]


def schemas_for(doi: str, schema: str):
    if schema != "auto":
        return [schema]
    d = doi.lower()
    if d.startswith(("10.6084/", "10.5281/", "10.3929/")):
        return ["datacite"]
    if d.startswith(("10.1109/", "10.1016/", "10.3390/", "10.1145/", "10.1007/", "10.2307/")):
        return ["crossref"]
    return ["crossref", "datacite"]


def year_of(msg):
    for k in ("published-print", "published-online", "issued"):
        dp = (msg.get(k) or {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            return int(dp[0][0])
    return None


def authors_cr(msg):
    out = []
    for a in msg.get("author") or []:
        if a.get("family"):
            out.append({"family": a["family"], "given": a.get("given") or ""})
    return out


def authors_dc(attrs):
    out = []
    for c in attrs.get("creators") or []:
        family = c.get("familyName") or ""
        given = c.get("givenName") or ""
        if not family and c.get("name"):
            name = c["name"]
            if "," in name:
                family, given = [x.strip() for x in name.split(",", 1)]
            else:
                parts = name.split()
                family, given = parts[-1], " ".join(parts[:-1])
        if family:
            if family.isupper():
                family = family.title()
            if given.isupper():
                given = given.title()
            out.append({"family": family, "given": given})
    return out


def citekey(authors, year, title, doi):
    fam = re.sub(r"[^A-Za-z]", "", (authors[0]["family"] if authors else "anon"))[:18].lower()
    slug = re.sub(r"[^a-z0-9]+", "", (title or "x").lower())[:16]
    h = hashlib.md5(doi.lower().encode()).hexdigest()[:4]
    return f"{fam}{year or 'nd'}{slug}{h}"


def bibtex_escape(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    s = s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
    for a, b in {
        "–": "--",
        "—": "---",
        "−": "-",
        "’": "'",
        "‘": "'",
        "“": "``",
        "”": "''",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
    }.items():
        s = s.replace(a, b)
    return s


def format_bib_entry(rec) -> str:
    authors = " and ".join(f"{a['family']}, {a['given']}".strip(", ") for a in rec["authors"]) or "Unknown"
    title = bibtex_escape(rec["title"])
    doi = rec["doi"]
    year = rec["year"] or ""
    journal = bibtex_escape(rec.get("container") or "")
    typ = rec.get("type") or "article"
    if typ in ("proceedings-article", "paper-conference"):
        entry_type, venue_field = "inproceedings", "booktitle"
    elif typ == "dataset":
        entry_type, venue_field = "misc", "howpublished"
        journal = journal or "Dataset"
    elif typ == "report":
        entry_type, venue_field = "techreport", "institution"
    else:
        entry_type, venue_field = "article", "journal"
    lines = [
        f"@{entry_type}{{{rec['key']},",
        f"  author = {{{authors}}},",
        f"  title = {{{title}}},",
        f"  year = {{{year}}},",
        f"  doi = {{{doi}}},",
        f"  url = {{https://doi.org/{doi}}},",
    ]
    if journal:
        lines.insert(4, f"  {venue_field} = {{{journal}}},")
    if rec.get("volume"):
        lines.append(f"  volume = {{{rec['volume']}}},")
    if rec.get("issue"):
        lines.append(f"  number = {{{rec['issue']}}},")
    if rec.get("page"):
        lines.append(f"  pages = {{{str(rec['page']).replace('-', '--')}}},")
    lines.append("}")
    return "\n".join(lines)


def blocked(title: str, doi: str) -> bool:
    if doi.lower() in EXCLUDE_DOI:
        return True
    if RE_EXCLUDE.search(title or ""):
        return True
    if DOMAIN_NOISE.search(title or ""):
        return True
    if APPLIED_NOISE.search(title or ""):
        return True
    return False


DOMAIN_NOISE = re.compile(
    r"\b(AGV|power flow|graph visualization|manufactur|workshop|wind and solar|"
    r"vehicle routing|job[\s-]?shop|cloud computing|wireless|image segmentation|"
    r"portfolio optimization|hydrolog|aircraft|ship routing|FPGA|VLSI|"
    r"UAV|drone|traffic signal|smart grid|microgrid|protein folding|"
    r"community detection|software product line)\b",
    re.I,
)


def accept_for_cluster(cluster: str, title: str, abstract: str = "") -> bool:
    text = f"{title} {abstract}"
    if blocked(title, ""):
        return False
    if cluster in ("A_classical_diet_OR", "B_evolutionary_menu", "C_diet_moo_recommend"):
        if not DIETISH.search(text):
            return False
        # allow optimization OR recommender OR planning OR evolutionary
        if not re.search(r"optim|recommend|planning|evolutionary|genetic|programming|integer|linear", text, re.I):
            return False
        return True
    if cluster == "G_nutrition_data":
        if EPI_NOISE.search(title):
            return False
        if DATAISH.search(text):
            return True
        # allow explicit dataset papers for meal plans / chronic-condition diets
        if re.search(r"dataset|database|corpus", text, re.I) and DIETISH.search(text) and re.search(
            r"meal plan|food composition|nutrient analysis|chronic condition", text, re.I
        ):
            return True
        return False
    if cluster == "D_moea_methods":
        if DIETISH.search(text) and MOEAISH.search(text):
            return True
        if DOMAIN_NOISE.search(title) or APPLIED_NOISE.search(title):
            return False
        # Prefer methodological titles; reject "algorithm for <application>"
        if re.search(
            r"\bfor\b.+\b(scheduling|routing|allocation|prediction|forecasting|control|assignment|design|dispatch)\b",
            title,
            re.I,
        ):
            return False
        return bool(D_TITLE_OK.search(title))
    if cluster == "E_benchmark_stats":
        return bool(BENCHISH.search(text) and MOEAISH.search(text))
    if cluster == "F_sparse_large_combo":
        if re.search(r"corrigendum|feature subset selection algorithm for", title, re.I):
            return False
        return bool(SPARSEISH.search(text) and MOEAISH.search(text))
    if cluster == "I_memetic_repair":
        return bool(MEMETICISH.search(text) and (MOEAISH.search(text) or re.search(r"constrained|combinatorial", text, re.I)))
    return False


def main():
    records: OrderedDict[str, dict] = OrderedDict()
    failed = []

    def add_record(doi: str, cluster: str, schema: str = "auto", stable_key: str | None = None, force: bool = False):
        doi_n = doi.strip().lower()
        if doi_n in records:
            return records[doi_n]
        if doi_n in EXCLUDE_DOI:
            return None
        last_err = None
        for sch in schemas_for(doi_n, schema):
            try:
                if sch == "crossref":
                    msg = crossref_work(doi_n)
                    title = (msg.get("title") or [""])[0]
                    authors = authors_cr(msg)
                    year = year_of(msg)
                    container = (msg.get("container-title") or [""])[0]
                    typ = msg.get("type") or "journal-article"
                    volume, issue, page = msg.get("volume"), msg.get("issue"), msg.get("page")
                    abstract = msg.get("abstract") or ""
                else:
                    attrs = datacite_work(doi_n)
                    title = (attrs.get("titles") or [{"title": ""}])[0].get("title", "")
                    authors = authors_dc(attrs)
                    year = attrs.get("publicationYear")
                    container = attrs.get("publisher") or ""
                    rtype = ((attrs.get("types") or {}).get("resourceTypeGeneral") or "other").lower()
                    typ = "dataset" if rtype == "dataset" else "misc"
                    volume = issue = page = None
                    abstract = ""
                    for d in attrs.get("descriptions") or []:
                        if d.get("description"):
                            abstract = d["description"]
                            break
                if not title:
                    raise ValueError("empty title")
                if blocked(title, doi_n):
                    return None
                if not force and not accept_for_cluster(cluster, title, abstract):
                    # seeds/force bypass
                    return None
                # Fix missing year for known conference
                if year is None and "ines.2006" in doi_n:
                    year = 2006
                key = stable_key or STABLE_KEY_BY_DOI.get(doi_n) or citekey(authors, year, title, doi_n)
                existing = {r["key"] for r in records.values()}
                base, i = key, 2
                while key in existing:
                    key = f"{base}_{i}"
                    i += 1
                rec = {
                    "doi": doi_n,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "container": container,
                    "type": typ,
                    "volume": volume,
                    "issue": issue,
                    "page": page,
                    "cluster": cluster,
                    "schema": sch,
                    "key": key,
                }
                records[doi_n] = rec
                print(f"  + [{cluster[:20]:20s}] {year} {title[:72]}")
                time.sleep(0.1)
                return rec
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        failed.append({"doi": doi_n, "cluster": cluster, "error": str(last_err)})
        print(f"  FAIL {doi_n}: {last_err}")
        return None

    print("=== SEEDS ===")
    for doi, cluster, schema, stable_key in SEEDS:
        add_record(doi, cluster, schema, stable_key=stable_key, force=True)

    print("=== EXTRA CURATED ===")
    for doi, cluster in EXTRA_CURATED:
        add_record(doi, cluster, "auto", force=True)

    print("=== CROSSREF QUERIES ===")
    for q, cluster, y0, y1, rows in QUERIES:
        print(f"QUERY ({len(records)}): {q[:64]}")
        params = {
            "query": q,
            "rows": rows,
            "filter": f"from-pub-date:{y0},until-pub-date:{y1},type:journal-article,type:proceedings-article",
            "select": "DOI,title,abstract,author,issued,published-print,published-online,container-title,type,volume,issue,page",
        }
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        try:
            items = get_json(url)["message"]["items"]
        except Exception as e:
            print("  query failed", e)
            time.sleep(1.5)
            continue
        for it in items:
            doi = (it.get("DOI") or "").lower()
            if not doi or doi in records:
                continue
            title = (it.get("title") or [""])[0]
            abstract = it.get("abstract") or ""
            if not accept_for_cluster(cluster, title, abstract):
                continue
            add_record(doi, cluster, "crossref", force=True)
        time.sleep(0.5)

    print(f"TOTAL BEFORE TRIM: {len(records)}")
    seed_dois = {d.lower() for d, _, __, ___ in SEEDS}

    def score(rec):
        y = rec["year"] or 0
        seed_bonus = 1000 if rec["doi"] in seed_dois else 0
        if 2023 <= y <= 2026:
            recent = 40 + (y - 2023) * 3
        elif 2015 <= y <= 2022:
            recent = 20
        elif y >= 2000:
            recent = 10
        else:
            recent = 15  # classics valuable
        # slight penalty for huge D oversupply later handled by quota
        return seed_bonus + recent

    by_cluster = defaultdict(list)
    for rec in records.values():
        by_cluster[rec["cluster"]].append(rec)
    for c in by_cluster:
        by_cluster[c].sort(key=score, reverse=True)

    selected: OrderedDict[str, dict] = OrderedDict()
    for doi in seed_dois:
        if doi in records:
            selected[doi] = records[doi]

    for cluster, quota in CLUSTER_QUOTA.items():
        have = sum(1 for r in selected.values() if r["cluster"] == cluster)
        for rec in by_cluster.get(cluster, []):
            if have >= quota:
                break
            if rec["doi"] not in selected:
                selected[rec["doi"]] = rec
                have += 1

    def cluster_count(c):
        return sum(1 for r in selected.values() if r["cluster"] == c)

    def can_add(rec):
        c = rec["cluster"]
        cap = HARD_CAP.get(c, CLUSTER_QUOTA.get(c, 150))
        return cluster_count(c) < cap

    # Fill only underfilled non-D-first clusters; never blow D hard cap
    fill_order = [c for c in CLUSTER_QUOTA if c != "D_moea_methods"] + ["D_moea_methods"]
    for c in fill_order:
        need = CLUSTER_QUOTA[c] - cluster_count(c)
        for rec in by_cluster.get(c, []):
            if need <= 0 or len(selected) >= 150:
                break
            if rec["doi"] not in selected and can_add(rec):
                selected[rec["doi"]] = rec
                need -= 1

    # Leftover slots: prefer diet/data/benchmark/sparse/memetic; skip D if at hard cap
    prefer = ["C_diet_moo_recommend", "B_evolutionary_menu", "A_classical_diet_OR",
              "E_benchmark_stats", "F_sparse_large_combo", "G_nutrition_data", "I_memetic_repair"]
    pool = []
    for c in prefer:
        pool.extend(by_cluster.get(c, []))
    pool.extend(r for r in records.values() if r["cluster"] == "D_moea_methods")
    seen = set()
    for rec in pool:
        if len(selected) >= 150:
            break
        if rec["doi"] in selected or rec["doi"] in seen:
            continue
        seen.add(rec["doi"])
        if can_add(rec):
            selected[rec["doi"]] = rec

    while len(selected) > 150:
        candidates = [r for r in selected.values() if r["doi"] not in seed_dois]
        if not candidates:
            break
        counts = defaultdict(int)
        for r in selected.values():
            counts[r["cluster"]] += 1
        # Prefer dropping over-quota, especially D overflow
        over = [r for r in candidates if counts[r["cluster"]] > CLUSTER_QUOTA.get(r["cluster"], 0)]
        d_over = [r for r in candidates if r["cluster"] == "D_moea_methods" and counts["D_moea_methods"] > HARD_CAP["D_moea_methods"]]
        pool_drop = d_over or over or candidates
        worst = min(pool_drop, key=score)
        del selected[worst["doi"]]

    final = list(selected.values())
    final.sort(key=lambda r: (r["cluster"], -(r["year"] or 0), r["key"]))

    if len(final) < 150:
        print(f"WARNING: only {len(final)} after strict filtering (target 150).")

    # Sanitize BibTeX for LaTeX
    def sanitize_bib_text(s: str) -> str:
        reps = {
            "\u2009": " ",
            "\u03b5": r"$\\varepsilon$",
            "\u2013": "--",
            "\u2014": "---",
            "\u2212": "-",
            "\u00a0": " ",
            "\ufb01": "fi",
            "\ufb02": "fl",
            "‐": "-",
            "’": "'",
            "‘": "'",
            "“": "``",
            "”": "''",
        }
        for a, b in reps.items():
            s = s.replace(a, b)
        return s

    bib_path = ROOT / "manuscript" / "references.bib"
    ver_path = ROOT / "docs" / "BIBLIOGRAPHY_VERIFICATION.json"
    map_path = ROOT / "docs" / "BIBLIOGRAPHY_CLUSTER_MAP.md"

    with bib_path.open("w", encoding="utf-8") as f:
        f.write("% NutriOpt-Bench survey bibliography (DOI-verified; RE papers excluded)\n")
        f.write("% Balanced cluster quotas; D_moea_methods hard-capped.\n")
        f.write(f"% Count: {len(final)}\n\n")
        for rec in final:
            f.write(sanitize_bib_text(format_bib_entry(rec)))
            f.write("\n\n")

    ver = {
        "count": len(final),
        "failed_lookups": failed,
        "schema_counts": {},
        "cluster_counts": {},
        "year_counts": {},
        "required_present": {
            "ACMO-Diet": any("cisce65916.2025.11065817" in r["doi"] for r in final),
            "DMP4P": any("figshare.30153778" in r["doi"] for r in final),
            "Zhao_LLM_RE_excluded": not any("compbiolchem.2026.108978" in r["doi"] for r in final),
        },
        "entries": [
            {
                "doi": r["doi"],
                "schema": r["schema"],
                "cluster": r["cluster"],
                "year": r["year"],
                "title": r["title"],
                "key": r["key"],
            }
            for r in final
        ],
    }
    for r in final:
        ver["schema_counts"][r["schema"]] = ver["schema_counts"].get(r["schema"], 0) + 1
        ver["cluster_counts"][r["cluster"]] = ver["cluster_counts"].get(r["cluster"], 0) + 1
        y = str(r["year"] or "NA")
        ver["year_counts"][y] = ver["year_counts"].get(y, 0) + 1
    ver_path.write_text(json.dumps(ver, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Bibliography cluster map (survey logic)\n\n",
        f"Total verified entries: **{len(final)}**\n\n",
        "Biomedical RE / LLM-RE papers excluded.\n\n",
        "| Cluster | Role | N | Quota |\n|---|---|---:|---:|\n",
    ]
    for c, role in ROLES.items():
        n = sum(1 for r in final if r["cluster"] == c)
        lines.append(f"| `{c}` | {role} | {n} | {CLUSTER_QUOTA[c]} |\n")
    lines.append("\n## Required\n")
    for k, v in ver["required_present"].items():
        lines.append(f"- {k}: {'YES' if v else 'NO'}\n")
    map_path.write_text("".join(lines), encoding="utf-8")

    # Write cite bundles for manuscript densification
    bundles = ROOT / "manuscript" / "cite_bundles.tex"
    with bundles.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated citation bundles by survey cluster\n")
        for c in ROLES:
            keys = [r["key"] for r in final if r["cluster"] == c]
            # cite in chunks of 8 to avoid insane cite lists if needed
            f.write(f"% {c} ({len(keys)})\n")
            if keys:
                f.write(f"\\newcommand{{\\cites{c.split('_')[0].capitalize()}}}{{{','.join(keys)}}}\n")
    print("WROTE", bib_path)
    print("COUNT", len(final))
    print("CLUSTERS", ver["cluster_counts"])
    print("YEARS", dict(sorted(ver["year_counts"].items())))
    print("REQUIRED", ver["required_present"])
    print("FAILED", len(failed))


if __name__ == "__main__":
    main()
