import os, time, json, re, argparse, hashlib, subprocess, tempfile, shutil
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

# -------- (optional) token count ----------
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text or ""))
except Exception:
    def count_tokens(text): return len((text or "").split())

# ================== 공통 Element 스키마 ======================
@dataclass
class Element:
    type: str
    text: str
    page: int
    meta: Dict[str, Any]

# ================== Parser 구현들 ============================
class BaseParser:
    name = "base"
    def parse(self, pdf_path: str) -> List[Element]:
        raise NotImplementedError

class UnstructuredParser(BaseParser):
    name = "unstructured"
    def __init__(self, use_ocr=False, ocr_langs="kor+eng"):
        self.use_ocr = use_ocr
        self.ocr_langs = ocr_langs
    def parse(self, pdf_path: str) -> List[Element]:
        from unstructured.partition.pdf import partition_pdf
        els = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            ocr_languages=self.ocr_langs if self.use_ocr else None
        )
        out = []
        for e in els:
            etype = getattr(e, "category", "raw").lower()
            map_type = {
                "title":"title","header":"heading","footer":"footnote","narrative_text":"paragraph",
                "list_item":"list","table":"table","figure":"raw","page_number":"raw","caption":"paragraph"
            }.get(etype, "paragraph")
            md = getattr(e, "metadata", {}) or {}
            page = int(getattr(md, "page_number", 1) or 1) if not isinstance(md, dict) else int(md.get("page_number", 1) or 1)
            text = getattr(e, "text", "") or ""
            out.append(Element(type=map_type, text=text, page=page, meta={"raw_type": etype, "ocr": self.use_ocr}))
        return out

class PyMuPDFParser(BaseParser):
    name = "pymupdf"
    def parse(self, pdf_path: str) -> List[Element]:
        import fitz
        out = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text")
                blocks = [b.strip() for b in (text or "").split("\n\n") if b.strip()]
                for b in blocks:
                    if re.match(r"^(제?\d+조|\d+\.\s|[A-Z][A-Z0-9\s]{3,}$)", b):
                        out.append(Element("heading", b, i, {}))
                    elif re.match(r"^\|.*\|$", b) or "│" in b or "┼" in b:
                        out.append(Element("table", b, i, {"heuristic":"ascii_table"}))
                    else:
                        out.append(Element("paragraph", b, i, {}))
        return out

class PDFMinerParser(BaseParser):
    name = "pdfminer"
    def parse(self, pdf_path: str) -> List[Element]:
        from pdfminer.high_level import extract_text
        txt = extract_text(pdf_path) or ""
        pages = re.split(r"\f", txt) if "\f" in txt else [txt]
        out = []
        for i, ptxt in enumerate(pages, start=1):
            chunks = [c.strip() for c in re.split(r"\n{2,}", ptxt) if c.strip()]
            for c in chunks:
                if re.match(r"^(제?\d+조|\d+\.\s|[A-Z][A-Z0-9\s]{3,}$)", c):
                    out.append(Element("heading", c, i, {}))
                elif re.search(r"\t{2,}", c) or re.search(r"\s{4,}\S+\s{2,}\S+", c):
                    out.append(Element("table", c, i, {"heuristic":"spacing"}))
                else:
                    out.append(Element("paragraph", c, i, {}))
        return out

class PDFPlumberParser(BaseParser):
    name = "pdfplumber"
    def parse(self, pdf_path: str) -> List[Element]:
        import pdfplumber
        out: List[Element] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                blocks = [b.strip() for b in re.split(r"\n{2,}", txt) if b.strip()]
                for b in blocks:
                    if re.match(r"^(제?\d+조|\d+\.\s|[A-Z][A-Z0-9\s]{3,}$)", b):
                        out.append(Element("heading", b, i, {}))
                    else:
                        out.append(Element("paragraph", b, i, {}))
                try:
                    tables = page.extract_tables() or []
                    for t in tables:
                        rows = [" | ".join([c or "" for c in row]) for row in t]
                        table_text = "\n".join(rows).strip()
                        if table_text:
                            out.append(Element("table", table_text, i, {"source": "pdfplumber"}))
                except Exception:
                    pass
        return out

class LlamaIndexParser(BaseParser):
    name = "llamaindex"
    def parse(self, pdf_path: str) -> List[Element]:
        from llama_index.core import Document
        from llama_index.readers.file import PDFReader
        reader = PDFReader()
        docs: List[Document] = reader.load_data(file=pdf_path)
        out: List[Element] = []
        for d in docs:
            text = getattr(d, "text", "") or ""
            md = getattr(d, "metadata", {}) or {}
            page = int(md.get("page_label") or md.get("page") or 1)
            chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]
            for c in chunks:
                if re.match(r"^(제?\d+조|\d+\.\s|[A-Z][A-Z0-9\s]{3,}$)", c):
                    out.append(Element("heading", c, page, {"src": "llamaindex"}))
                elif re.search(r"\t{2,}", c) or re.search(r"\s{4,}\S+\s{2,}\S+", c):
                    out.append(Element("table", c, page, {"src": "llamaindex", "heuristic": "spacing"}))
                else:
                    out.append(Element("paragraph", c, page, {"src": "llamaindex"}))
        return out

# [OCR+] 추가: Unstructured OCR 전용 파서
class UnstructuredOCRParser(UnstructuredParser):
    name = "unstructured_ocr"
    def __init__(self, ocr_langs="kor+eng"):
        super().__init__(use_ocr=True, ocr_langs=ocr_langs)

# ================== 품질 지표(기존+보강) ======================
def quality_metrics(elements: List[Element]) -> Dict[str, Any]:
    n = len(elements)
    n_tables = sum(1 for e in elements if e.type == "table")
    n_headings = sum(1 for e in elements if e.type in ("title","heading"))
    n_paras = sum(1 for e in elements if e.type == "paragraph")
    text_all = "\n".join(e.text for e in elements)
    tok = count_tokens(text_all)
    avg_chunk_tok = (sum(count_tokens(e.text) for e in elements) / max(n,1))
    clause_like = sum(1 for e in elements if re.search(r"제\s?\d+\s?조", e.text))
    clause_ratio = clause_like / max(n_headings, 1)
    return {
        "n_elements": n,
        "n_tables": n_tables,
        "n_headings": n_headings,
        "n_paragraphs": n_paras,
        "total_tokens": tok,
        "avg_chunk_tokens": round(avg_chunk_tok, 2),
        "clause_heading_ratio": round(clause_ratio, 3)
    }

def extra_metrics(elements: List[Element], pdf_path: str, elapsed_sec: float) -> Dict[str, Any]:
    try:
        import fitz
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
    except Exception:
        total_pages = max([e.page for e in elements] + [1])

    pages_seen = len(set(e.page for e in elements))
    page_coverage = round(pages_seen / max(total_pages, 1), 3)

    order_breaks = 0
    last = -1
    for e in sorted(elements, key=lambda x: (x.page, 0)):
        if last > e.page:
            order_breaks += 1
        last = e.page

    import statistics as stats
    hashes, lens = [], []
    short_or_punct = 0
    for e in elements:
        t = (e.text or "").strip()
        if len(t) < 5 or re.fullmatch(r"[\W_]+", t or ""):
            short_or_punct += 1
        hashes.append(hashlib.md5(t.encode("utf-8")).hexdigest())
        try:
            lens.append(count_tokens(e.text))
        except Exception:
            lens.append(len((e.text or "").split()))
    dup_ratio = round(1 - (len(set(hashes)) / max(len(hashes), 1)), 3)
    noise_ratio = round(short_or_punct / max(len(elements), 1), 3)
    std_chunk_tokens = round(stats.pstdev(lens) if lens else 0.0, 2)

    elapsed_per_page = round(elapsed_sec / max(total_pages, 1), 3)

    return {
        "total_pages": total_pages,
        "page_coverage": page_coverage,
        "order_breaks": order_breaks,
        "duplicate_ratio": dup_ratio,
        "noise_ratio": noise_ratio,
        "std_chunk_tokens": std_chunk_tokens,
        "elapsed_per_page": elapsed_per_page,
    }

def table_metrics(elements: List[Element]) -> Dict[str, Any]:
    tables = [e for e in elements if e.type == "table"]
    import re as _re
    row_counts, numeric_cells, cell_counts = [], 0, 0
    for t in tables:
        rows = [r for r in (t.text or "").splitlines() if r.strip()]
        row_counts.append(len(rows))
        for r in rows:
            cells = [c.strip() for c in _re.split(r"\s+\|\s+|\t|\s{2,}", r) if c.strip()]
            cell_counts += len(cells)
            numeric_cells += sum(1 for c in cells if _re.search(r"\d", c))
    return {
        "avg_table_rows": round(sum(row_counts)/len(row_counts), 2) if row_counts else 0.0,
        "numeric_cell_ratio": round(numeric_cells / cell_counts, 3) if cell_counts else 0.0
    }

# ================== OCR 유틸 (pre, fallback) ==================
# [OCR+] pre-OCR: ocrmypdf로 검색가능 PDF 생성
def run_ocrmypdf(input_pdf: str, lang: str = "kor+eng") -> str:
    tmpdir = tempfile.mkdtemp(prefix="ocr_")
    out_pdf = os.path.join(tmpdir, "ocr.pdf")
    cmd = ["ocrmypdf", "--skip-text", "-l", lang, input_pdf, out_pdf]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_pdf
    except Exception as e:
        # 실패시 원본으로 폴백
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"ocrmypdf failed: {e}")

# [OCR+] fallback OCR: 텍스트 빈약 페이지만 Tesseract로 보충
def ocr_pages_if_needed(pdf_path: str, elements: List[Element], lang: str = "kor+eng", token_threshold: int = 40) -> List[Element]:
    import fitz, pytesseract
    from PIL import Image

    # 페이지별 토큰 합 계산
    from collections import defaultdict
    page_tok = defaultdict(int)
    for e in elements:
        page_tok[e.page] += count_tokens(e.text)

    out = list(elements)
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            if page_tok[i] >= token_threshold:
                continue  # 충분히 텍스트가 있음
            # 페이지 렌더 후 OCR
            pix = page.get_pixmap(dpi=300, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            txt = pytesseract.image_to_string(img, lang=lang).strip()
            if txt:
                out.append(Element("paragraph", txt, i, {"ocr": "tesseract", "dpi": 300}))
    return out

# ================== 실행/리포트 ==============================
PARSERS = {
    "unstructured": lambda: UnstructuredParser(use_ocr=False),
    "pymupdf": lambda: PyMuPDFParser(),
    "pdfminer": lambda: PDFMinerParser(),
    "pdfplumber": lambda: PDFPlumberParser(),
    "llamaindex": lambda: LlamaIndexParser(),
    # [OCR+] 추가 파서
    "unstructured_ocr": lambda: UnstructuredOCRParser(ocr_langs=OCR_LANG),
}

def run_once(parser_name: str, pdf_path: str, out_dir: str,
             ocr_mode: str = "none", ocr_lang: str = "kor+eng", ocr_threshold: int = 40) -> Dict[str, Any]:
    parser = PARSERS[parser_name]()
    pdf_used = pdf_path

    # [OCR+] pre-OCR 경로: 문서 전체를 먼저 OCR
    if ocr_mode == "pre":
        try:
            pdf_used = run_ocrmypdf(pdf_path, lang=ocr_lang)
        except Exception as e:
            # pre-OCR 실패 시 원본으로 진행(로그만 남김)
            print(f"[warn] ocrmypdf failed: {e}. use original pdf.")

    t0 = time.time()
    elements = parser.parse(pdf_used)

    # [OCR+] fallback 경로: 텍스트 부족 페이지만 OCR 보충
    if ocr_mode == "fallback":
        try:
            elements = ocr_pages_if_needed(pdf_path, elements, lang=ocr_lang, token_threshold=ocr_threshold)
        except Exception as e:
            print(f"[warn] fallback OCR failed: {e}")

    elapsed = time.time() - t0

    # 결과 저장(JSONL)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    jsonl_path = os.path.join(out_dir, f"{base}.{parser_name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for e in elements:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    metrics = quality_metrics(elements)
    metrics.update({
        "parser": parser_name,
        "pdf": os.path.basename(pdf_path),
        "elapsed_sec": round(elapsed, 3),
        "jsonl": jsonl_path
    })
    metrics.update(extra_metrics(elements, pdf_path, elapsed))
    metrics.update(table_metrics(elements))
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="실험할 PDF 폴더")
    ap.add_argument("--out_dir", default="./parser_outputs", help="출력 폴더")
    ap.add_argument("--parsers", default="unstructured,pymupdf,pdfminer,pdfplumber,llamaindex,unstructured_ocr",
                    help="쉼표구분: unstructured,pymupdf,pdfminer,pdfplumber,llamaindex,unstructured_ocr")
    # [OCR+] 옵션
    ap.add_argument("--ocr_mode", default="none", choices=["none","fallback","pre"],
                    help="none: OCR 미사용, fallback: 텍스트 부족 페이지만 Tesseract, pre: 문서 전체 ocrmypdf 후 파싱")
    ap.add_argument("--ocr_lang", default="kor+eng", help="예: 'kor', 'eng', 'kor+eng'")
    ap.add_argument("--ocr_threshold", type=int, default=40, help="fallback에서 텍스트 부족 판단 토큰 기준")
    args = ap.parse_args()

    # [OCR+] 전역 접근을 위해 저장
    global OCR_LANG
    OCR_LANG = args.ocr_lang

    pdfs = [os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir)
            if f.lower().endswith(".pdf")]
    parsers = [p.strip() for p in args.parsers.split(",") if p.strip()]

    rows = []
    for pdf_path in tqdm(pdfs, desc="PDFs"):
        for p in parsers:
            try:
                m = run_once(p, pdf_path, args.out_dir,
                             ocr_mode=args.ocr_mode, ocr_lang=args.ocr_lang, ocr_threshold=args.ocr_threshold)
            except Exception as e:
                m = {"parser": p, "pdf": os.path.basename(pdf_path), "error": str(e)}
            rows.append(m)

    df = pd.DataFrame(rows)
    rep_csv = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(rep_csv, index=False, encoding="utf-8-sig")

    def score(row):
        return (
            2.0 * row.get("n_tables", 0)
          + 1.5 * row.get("clause_heading_ratio", 0)
          + 0.5 * row.get("n_headings", 0)
          + 1.0 * row.get("page_coverage", 0)
          + 0.5 * row.get("numeric_cell_ratio", 0)
          - 0.2 * row.get("elapsed_per_page", 0)
          - 0.5 * row.get("noise_ratio", 0)
          - 0.5 * row.get("duplicate_ratio", 0)
        )

    if not df.empty:
        ok = df[df.get("error").isna()] if "error" in df.columns else df.copy()
        if not ok.empty:
            ok["score"] = ok.apply(score, axis=1)
            best = (ok.sort_values(["pdf","score"], ascending=[True,False])
                      .groupby("pdf").head(1))
            best_csv = os.path.join(args.out_dir, "winners_by_pdf.csv")
            best.to_csv(best_csv, index=False, encoding="utf-8-sig")

            overall = (ok.groupby("parser")
                         .agg({"score":"mean",
                               "elapsed_sec":"mean",
                               "elapsed_per_page":"mean",
                               "n_tables":"sum",
                               "n_headings":"sum",
                               "total_tokens":"mean",
                               "page_coverage":"mean",
                               "numeric_cell_ratio":"mean"})
                         .sort_values("score", ascending=False))
            overall_csv = os.path.join(args.out_dir, "overall_ranking.csv")
            overall.to_csv(overall_csv, encoding="utf-8-sig")

            print("\n[요약] 전체 우승자 랭킹:")
            print(overall.reset_index().to_string(index=False))
            print(f"\n리포트 저장: {rep_csv}")
            print(f"PDF별 우승자: {best_csv}")
            print(f"전체 랭킹: {overall_csv}")
        else:
            print("\n리포트 생성 완료(모두 에러). summary.csv를 확인하세요.")
    else:
        print("\n리포트 생성 완료(빈 결과). summary.csv를 확인하세요.")

if __name__ == "__main__":
    main()