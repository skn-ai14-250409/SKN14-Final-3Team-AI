import os, time, json, re, argparse, hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

# -------- (optional) token count for 길이/밀도 지표 ----------
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text or ""))
except Exception:
    def count_tokens(text): return len((text or "").split())

# ================== 공통 Element 스키마 ======================
# 모든 파서가 추출한 결과를 표준화하기 위한 공통 형식임. 어떤 파서를 쓰든 결과물은 항상 밑의 정보를 가진 Elemnet객체의 리스트가 됨
@dataclass
class Element:
    type: str          # "title" | "heading" | "paragraph" | "list" | "table" | "footnote" | "raw"
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
    def __init__(self, use_ocr=False):
        self.use_ocr = use_ocr
    def parse(self, pdf_path: str) -> List[Element]:
        from unstructured.partition.pdf import partition_pdf
        els = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            ocr_languages="kor+eng" if self.use_ocr else None
        )
        out = []
        for e in els:
            etype = getattr(e, "category", "raw").lower()
            # 맵핑
            map_type = {
                "title":"title","header":"heading","footer":"footnote","narrative_text":"paragraph",
                "list_item":"list","table":"table","figure":"raw","page_number":"raw","caption":"paragraph"
            }.get(etype, "paragraph")
            page = int(getattr(e.metadata, "page_number", 1) or 1)
            text = getattr(e, "text", "") or ""
            out.append(Element(type=map_type, text=text, page=page, meta={"raw_type": etype}))
        return out

class PyMuPDFParser(BaseParser):
    name = "pymupdf"
    def parse(self, pdf_path: str) -> List[Element]:
        import fitz
        out = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text")  # 단순 추출
                # 간단한 헤딩 탐지(굵은 패턴/조항 번호 정규식)
                headings = []
                blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
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
        # 페이지 구분이 약하므로 간단히 페이지 힌트 추정
        pages = re.split(r"\f", txt) if "\f" in txt else [txt]
        out = []
        for i, ptxt in enumerate(pages, start=1):
            chunks = [c.strip() for c in re.split(r"\n{2,}", ptxt) if c.strip()]
            for c in chunks:
                # 간단 헤딩/표 휴리스틱
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
                # 1) 텍스트를 문단 단위로 (두 줄 이상 띄어쓰기를 문단 경계로)
                txt = page.extract_text() or ""
                blocks = [b.strip() for b in re.split(r"\n{2,}", txt) if b.strip()]
                for b in blocks:
                    if re.match(r"^(제?\d+조|\d+\.\s|[A-Z][A-Z0-9\s]{3,}$)", b):
                        out.append(Element("heading", b, i, {}))
                    else:
                        out.append(Element("paragraph", b, i, {}))
                # 2) 테이블 추출 (셀을 '|'로 합쳐 간단 직렬화)
                try:
                    tables = page.extract_tables() or []
                    for t in tables:
                        rows = [" | ".join([c or "" for c in row]) for row in t]
                        table_text = "\n".join(rows).strip()
                        if table_text:
                            out.append(Element("table", table_text, i, {"source": "pdfplumber"}))
                except Exception as _:
                    pass
        return out
    
# (라이트하게 PDFReader로 문서 → 청크 변환. 메타에서 페이지번호를 최대한 보존)
class LlamaIndexParser(BaseParser):
    name = "llamaindex"
    def parse(self, pdf_path: str) -> List[Element]:
        # llama-index 0.10+ 계열
        from llama_index.core import Document
        from llama_index.readers.file import PDFReader

        reader = PDFReader()
        docs: List[Document] = reader.load_data(file=pdf_path)  # 문서 리스트(페이지 메타 포함 가능)

        out: List[Element] = []
        for d in docs:
            text = d.text or ""
            # 페이지 정보 추정: page_label 또는 page / metadata dict
            md = getattr(d, "metadata", {}) or {}
            page = int(md.get("page_label") or md.get("page") or 1)

            # 간단 휴리스틱으로 heading/table/paragraph 분류
            chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]
            for c in chunks:
                if re.match(r"^(제?\d+조|\d+\.\s|[A-Z][A-Z0-9\s]{3,}$)", c):
                    out.append(Element("heading", c, page, {"src": "llamaindex"}))
                elif re.search(r"\t{2,}", c) or re.search(r"\s{4,}\S+\s{2,}\S+", c):
                    out.append(Element("table", c, page, {"src": "llamaindex", "heuristic": "spacing"}))
                else:
                    out.append(Element("paragraph", c, page, {"src": "llamaindex"}))
        return out


# ================== 품질 지표 계산 ===========================
def quality_metrics(elements: List[Element]) -> Dict[str, Any]:
    n = len(elements)
    n_tables = sum(1 for e in elements if e.type == "table")
    n_headings = sum(1 for e in elements if e.type in ("title","heading"))
    n_paras = sum(1 for e in elements if e.type == "paragraph")
    text_all = "\n".join(e.text for e in elements)
    tok = count_tokens(text_all)
    avg_chunk_tok = (sum(count_tokens(e.text) for e in elements) / max(n,1))
    # 금융 문서(조항) 헤딩 패턴 비율
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

# ================== 실행/리포트 ==============================
PARSERS = {
    "unstructured": lambda: UnstructuredParser(use_ocr=False),
    "pymupdf": lambda: PyMuPDFParser(),
    "pdfminer": lambda: PDFMinerParser(),
    "pdfplumber": lambda: PDFPlumberParser(),      # 추가
    "llamaindex": lambda: LlamaIndexParser(),      # 추가
}

def run_once(parser_name: str, pdf_path: str, out_dir: str) -> Dict[str, Any]:
    parser = PARSERS[parser_name]()
    t0 = time.time()
    elements = parser.parse(pdf_path)
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
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="실험할 PDF 폴더")
    ap.add_argument("--out_dir", default="./parser_outputs", help="출력 폴더")
    ap.add_argument("--parsers", default="unstructured,pymupdf,pdfminer",
                    help="쉼표구분: unstructured,pymupdf,pdfminer")
    args = ap.parse_args()

    pdfs = [os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir)
            if f.lower().endswith(".pdf")]
    parsers = [p.strip() for p in args.parsers.split(",") if p.strip()]

    rows = []
    for pdf_path in tqdm(pdfs, desc="PDFs"):
        for p in parsers:
            try:
                m = run_once(p, pdf_path, args.out_dir)
            except Exception as e:
                m = {"parser": p, "pdf": os.path.basename(pdf_path), "error": str(e)}
            rows.append(m)

    df = pd.DataFrame(rows)
    rep_csv = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(rep_csv, index=False, encoding="utf-8-sig")

    # 간단한 승자 선정(가중치 예시): 테이블탐지↑, 조항헤딩↑, 속도↓
    def score(row):
        return (2.0 * row.get("n_tables",0)
                + 1.5 * row.get("clause_heading_ratio",0)
                + 0.5 * row.get("n_headings",0)
                - 0.2 * row.get("elapsed_sec",10))

    if not df.empty and "error" not in df.columns:
        df["score"] = df.apply(score, axis=1)
        best = (df.sort_values(["pdf","score"], ascending=[True,False])
                  .groupby("pdf").head(1))
        best_csv = os.path.join(args.out_dir, "winners_by_pdf.csv")
        best.to_csv(best_csv, index=False, encoding="utf-8-sig")

        # 전체 우승자
        overall = (df.groupby("parser")
                     .agg({"score":"mean","elapsed_sec":"mean","n_tables":"sum",
                           "n_headings":"sum","total_tokens":"mean"})
                     .sort_values("score", ascending=False))
        overall_csv = os.path.join(args.out_dir, "overall_ranking.csv")
        overall.to_csv(overall_csv, encoding="utf-8-sig")
        print("\n[요약] 전체 우승자 랭킹:")
        print(overall.reset_index().to_string(index=False))
        print(f"\n리포트 저장: {rep_csv}")
        print(f"PDF별 우승자: {best_csv}")
        print(f"전체 랭킹: {overall_csv}")
    else:
        print("\n리포트 생성 완료(일부 에러 존재). summary.csv를 확인하세요.")

if __name__ == "__main__":
    main()