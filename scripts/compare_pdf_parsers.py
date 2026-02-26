#!/usr/bin/env python3
"""
PDF 解析器对比工具
对比当前方案 (pdfplumber) 与 Docling/MinerU 的效果

使用方法：
    python scripts/compare_pdf_parsers.py <pdf_file>
    
或者测试所有 PDF：
    python scripts/compare_pdf_parsers.py --all
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_with_pdfplumber(pdf_path: str) -> tuple[str, list]:
    """使用当前方案 (pdfplumber) 提取"""
    import pdfplumber
    
    text_parts = []
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 提取文本
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            
            # 提取表格
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 1:
                    tables.append(table)
    
    return "\n".join(text_parts), tables


def extract_with_docling(pdf_path: str) -> tuple[str, list]:
    """使用 Docling 提取 (如果安装了)"""
    try:
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        text = result.document.export_to_markdown()
        
        # Docling 的表格会在 markdown 中
        tables = []  # Docling 将表格嵌入到 markdown 中
        
        return text, tables
        
    except ImportError:
        return None, None


def extract_with_mineru(pdf_path: str) -> tuple[str, list]:
    """使用 MinerU 提取 (如果安装了)"""
    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
        import json
        
        # MinerU 需要较复杂的设置
        # 这里只是示意，实际使用需要配置模型路径等
        
        return None, None  # 需要 GPU 和模型配置
        
    except ImportError:
        return None, None


def analyze_extraction_quality(text: str, tables: list, pdf_path: str) -> dict:
    """分析提取质量"""
    results = {
        "text_length": len(text),
        "table_count": len(tables),
        "issues": [],
        "scores": {},
    }
    
    # 1. 基本统计
    word_count = len(text.split())
    results["word_count"] = word_count
    
    # 2. 检查关键财务词汇是否存在
    financial_keywords = [
        "revenue", "income", "assets", "liabilities", "cash",
        "capital expenditure", "property", "equipment",
        "fiscal year", "consolidated", "statements",
    ]
    
    found_keywords = sum(1 for kw in financial_keywords if kw.lower() in text.lower())
    results["scores"]["keyword_coverage"] = found_keywords / len(financial_keywords)
    
    # 3. 检查数字提取
    import re
    numbers = re.findall(r'\$[\d,]+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?', text)
    results["number_count"] = len(numbers)
    
    if len(numbers) < 50:
        results["issues"].append("数字提取过少，可能丢失了表格数据")
    
    # 4. 检查表格质量
    total_cells = sum(len(row) for table in tables for row in table)
    results["total_table_cells"] = total_cells
    
    # 表格中的数字
    table_numbers = 0
    for table in tables:
        for row in table:
            for cell in row:
                if cell and re.search(r'\d', str(cell)):
                    table_numbers += 1
    results["table_number_cells"] = table_numbers
    
    if tables and table_numbers < total_cells * 0.3:
        results["issues"].append("表格中数字占比低，可能提取有误")
    
    # 5. 检查常见解析错误
    # 字符粘连
    long_words = [w for w in text.split() if len(w) > 30]
    if len(long_words) > 10:
        results["issues"].append(f"发现 {len(long_words)} 个超长词，可能有字符粘连")
    
    # 乱码检测
    import string
    non_printable = sum(1 for c in text if c not in string.printable and c not in '\n\r\t ')
    if non_printable > len(text) * 0.01:
        results["issues"].append(f"发现 {non_printable} 个不可打印字符，可能有编码问题")
    
    # 6. 计算总体评分
    scores = results["scores"]
    scores["text_extraction"] = min(1.0, word_count / 5000)  # 假设 5000 词是正常
    scores["table_extraction"] = min(1.0, total_cells / 500)  # 假设 500 格是正常
    scores["number_preservation"] = min(1.0, len(numbers) / 200)  # 假设 200 个数字正常
    
    results["overall_score"] = sum(scores.values()) / len(scores)
    
    return results


def compare_single_pdf(pdf_path: str):
    """对比单个 PDF 的解析效果"""
    print(f"\n{'='*70}")
    print(f"  文件: {Path(pdf_path).name}")
    print(f"{'='*70}")
    
    # 1. pdfplumber (当前方案)
    print("\n  [pdfplumber] 正在提取...")
    text_plumber, tables_plumber = extract_with_pdfplumber(pdf_path)
    results_plumber = analyze_extraction_quality(text_plumber, tables_plumber, pdf_path)
    
    print(f"    文本长度: {results_plumber['text_length']:,} 字符")
    print(f"    词数: {results_plumber['word_count']:,}")
    print(f"    表格数: {results_plumber['table_count']}")
    print(f"    数字数: {results_plumber['number_count']}")
    print(f"    总评分: {results_plumber['overall_score']:.2f}")
    
    if results_plumber["issues"]:
        print("    ⚠️  问题:")
        for issue in results_plumber["issues"]:
            print(f"       - {issue}")
    
    # 2. Docling (如果可用)
    print("\n  [Docling] 正在提取...")
    text_docling, tables_docling = extract_with_docling(pdf_path)
    
    if text_docling:
        results_docling = analyze_extraction_quality(text_docling, tables_docling or [], pdf_path)
        print(f"    文本长度: {results_docling['text_length']:,} 字符")
        print(f"    总评分: {results_docling['overall_score']:.2f}")
        
        # 对比
        if results_docling["overall_score"] > results_plumber["overall_score"] + 0.1:
            print("    ✅ Docling 效果更好，建议使用")
        elif results_docling["overall_score"] < results_plumber["overall_score"] - 0.1:
            print("    ✅ pdfplumber 效果更好，保持现状")
        else:
            print("    ➡️  两者效果相近")
    else:
        print("    ⏭️  Docling 未安装，跳过")
        print("       安装: pip install docling")
    
    # 3. 显示示例内容
    print("\n  内容预览 (pdfplumber):")
    preview = text_plumber[:500].replace("\n", " ")
    print(f"    {preview}...")
    
    if tables_plumber:
        print(f"\n  表格预览 (第1个表格前3行):")
        for row in tables_plumber[0][:3]:
            print(f"    {row}")
    
    return results_plumber


def find_problematic_pdfs(data_dir: str):
    """找出可能有解析问题的 PDF"""
    print("\n" + "=" * 70)
    print("  扫描所有 PDF，找出可能有问题的文件")
    print("=" * 70)
    
    pdf_files = list(Path(data_dir).rglob("*.pdf"))
    print(f"\n  找到 {len(pdf_files)} 个 PDF 文件")
    
    problematic = []
    good = []
    
    for i, pdf_path in enumerate(pdf_files):
        print(f"\r  处理中: {i+1}/{len(pdf_files)} - {pdf_path.name[:40]}...", end="", flush=True)
        
        try:
            text, tables = extract_with_pdfplumber(str(pdf_path))
            results = analyze_extraction_quality(text, tables, str(pdf_path))
            
            if results["issues"] or results["overall_score"] < 0.5:
                problematic.append({
                    "path": str(pdf_path),
                    "score": results["overall_score"],
                    "issues": results["issues"],
                })
            else:
                good.append({
                    "path": str(pdf_path),
                    "score": results["overall_score"],
                })
        except Exception as e:
            problematic.append({
                "path": str(pdf_path),
                "score": 0,
                "issues": [f"解析失败: {e}"],
            })
    
    print("\r" + " " * 80)  # 清除进度行
    
    # 按问题严重程度排序
    problematic.sort(key=lambda x: x["score"])
    
    print(f"\n  ✅ 解析良好: {len(good)} 个")
    print(f"  ⚠️  可能有问题: {len(problematic)} 个")
    
    if problematic:
        print("\n  问题文件列表 (按评分从低到高):")
        for item in problematic[:10]:  # 只显示前10个
            print(f"\n    {Path(item['path']).name}")
            print(f"      评分: {item['score']:.2f}")
            for issue in item["issues"]:
                print(f"      - {issue}")
        
        if len(problematic) > 10:
            print(f"\n    ... 还有 {len(problematic) - 10} 个问题文件")
        
        print("\n  建议:")
        print("    如果问题文件较多，考虑使用 Docling 或 MinerU")
        print("    运行: python scripts/compare_pdf_parsers.py <问题文件路径>")
        print("    对比不同解析器的效果")
    
    return problematic


def main():
    parser = argparse.ArgumentParser(description="PDF 解析器对比工具")
    parser.add_argument("pdf_path", nargs="?", help="要分析的 PDF 文件路径")
    parser.add_argument("--all", action="store_true", help="扫描所有 PDF 文件")
    parser.add_argument("--data-dir", default="data", help="数据目录 (默认: data)")
    
    args = parser.parse_args()
    
    if args.all:
        data_dir = project_root / args.data_dir
        if not data_dir.exists():
            print(f"错误: 数据目录不存在 - {data_dir}")
            sys.exit(1)
        find_problematic_pdfs(str(data_dir))
    elif args.pdf_path:
        if not Path(args.pdf_path).exists():
            print(f"错误: 文件不存在 - {args.pdf_path}")
            sys.exit(1)
        compare_single_pdf(args.pdf_path)
    else:
        print("使用方法:")
        print("  python scripts/compare_pdf_parsers.py <pdf_file>   # 分析单个 PDF")
        print("  python scripts/compare_pdf_parsers.py --all        # 扫描所有 PDF")


if __name__ == "__main__":
    main()
