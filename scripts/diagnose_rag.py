#!/usr/bin/env python3
"""
RAG 系统诊断工具
用于定位问题出在哪个环节：PDF解析 / 分块 / 检索 / 生成

使用方法：
    cd Flex-Practicum-Project-2026
    python scripts/diagnose_rag.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import get_collection, embed_text


# ===========================================================================
# 1. PDF 解析质量检测
# ===========================================================================
def diagnose_pdf_parsing():
    """检查 PDF 解析是否正确提取了关键财务数据"""
    print("\n" + "=" * 70)
    print("  1. PDF 解析质量诊断")
    print("=" * 70)
    
    collection = get_collection()
    if collection.count() == 0:
        print("  ❌ ChromaDB 为空，请先运行 build_chromadb.py")
        return False
    
    # 搜索应该存在的关键内容
    test_queries = [
        # CapEx 相关（必须能找到）
        ("CapEx 数据", "purchases of property and equipment capital expenditure"),
        ("现金流量表", "consolidated statements of cash flows investing activities"),
        ("资产负债表", "total assets total liabilities balance sheet"),
        
        # 公司特定
        ("Flex 数据", "Flex Ltd revenue operating income"),
        ("Jabil 数据", "Jabil Inc capital expenditure property equipment"),
    ]
    
    issues = []
    for label, query in test_queries:
        query_emb = embed_text(query)
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"][0]:
            issues.append(f"❌ {label}: 未找到任何结果")
            continue
        
        best_sim = 1 - results["distances"][0][0]
        best_doc = results["documents"][0][0][:200]
        best_meta = results["metadatas"][0][0]
        
        status = "✅" if best_sim > 0.3 else "⚠️"
        print(f"\n  {status} {label}")
        print(f"     相似度: {best_sim:.3f}")
        print(f"     来源: [{best_meta.get('company', '?')}] {best_meta.get('source_file', '?')}")
        print(f"     内容: {best_doc}...")
        
        if best_sim < 0.3:
            issues.append(f"⚠️ {label}: 相似度过低 ({best_sim:.3f})")
    
    if issues:
        print(f"\n  发现 {len(issues)} 个潜在问题:")
        for issue in issues:
            print(f"    {issue}")
        return False
    
    print("\n  ✅ PDF 解析质量良好")
    return True


# ===========================================================================
# 2. 表格提取质量检测
# ===========================================================================
def diagnose_table_extraction():
    """检查表格是否被正确提取和序列化"""
    print("\n" + "=" * 70)
    print("  2. 表格提取质量诊断")
    print("=" * 70)
    
    collection = get_collection()
    
    # 检查是否有表格类型的 chunk
    results = collection.get(
        where={"chunk_type": "table"},
        include=["documents", "metadatas"],
        limit=5,
    )
    
    if not results["documents"]:
        print("  ⚠️ 未找到表格类型的 chunk")
        print("     可能原因: 表格未被识别，或使用了旧版分块")
        
        # 尝试搜索表格内容
        query_emb = embed_text("consolidated statements cash flows capex property equipment")
        search_results = collection.query(
            query_embeddings=[query_emb],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        print("\n  尝试搜索表格内容:")
        for doc, meta in zip(search_results["documents"][0], search_results["metadatas"][0]):
            has_table = "|" in doc or "---" in doc
            status = "📊" if has_table else "📄"
            print(f"    {status} [{meta.get('company')}] {meta.get('chunk_type', 'unknown')}")
            if has_table:
                # 显示表格片段
                lines = [l for l in doc.split("\n") if "|" in l][:3]
                for line in lines:
                    print(f"       {line[:80]}")
        return False
    
    print(f"  ✅ 找到 {len(results['documents'])} 个表格 chunk")
    
    for i, (doc, meta) in enumerate(zip(results["documents"][:3], results["metadatas"][:3])):
        print(f"\n  表格 {i+1}: [{meta.get('company')}] {meta.get('table_type', 'unknown')}")
        print(f"     上下文: {meta.get('table_context', '')[:50]}")
        # 显示表格前几行
        lines = doc.split("\n")[:5]
        for line in lines:
            print(f"     {line[:70]}")
    
    return True


# ===========================================================================
# 3. 父子文档结构检测
# ===========================================================================
def diagnose_parent_child():
    """检查父子文档结构是否正确"""
    print("\n" + "=" * 70)
    print("  3. 父子文档结构诊断")
    print("=" * 70)
    
    collection = get_collection()
    
    # 统计各类型 chunk
    chunk_types = {"child": 0, "parent": 0, "table": 0, "legacy": 0, "unknown": 0}
    
    # 获取所有 chunk 的元数据
    all_results = collection.get(
        include=["metadatas"],
        limit=10000,
    )
    
    for meta in all_results["metadatas"]:
        ctype = meta.get("chunk_type", "unknown")
        if ctype in chunk_types:
            chunk_types[ctype] += 1
        else:
            chunk_types["unknown"] += 1
    
    print(f"\n  Chunk 类型分布:")
    for ctype, count in chunk_types.items():
        if count > 0:
            print(f"    {ctype:<10}: {count:>5}")
    
    # 检查是否使用了新的父子结构
    if chunk_types["child"] == 0 and chunk_types["parent"] == 0:
        print("\n  ⚠️ 未使用父子文档结构")
        print("     建议: 重新运行 build_chromadb.py 使用增强版分块")
        return False
    
    # 验证父子关系
    child_results = collection.get(
        where={"chunk_type": "child"},
        include=["metadatas"],
        limit=10,
    )
    
    valid_refs = 0
    for meta in child_results["metadatas"]:
        if meta.get("parent_id") and meta.get("parent_preview"):
            valid_refs += 1
    
    if valid_refs == len(child_results["metadatas"]):
        print(f"\n  ✅ 父子关系验证通过 ({valid_refs}/{len(child_results['metadatas'])})")
        return True
    else:
        print(f"\n  ⚠️ 部分 child 缺少 parent 引用 ({valid_refs}/{len(child_results['metadatas'])})")
        return False


# ===========================================================================
# 4. 检索质量检测
# ===========================================================================
def diagnose_retrieval():
    """测试检索是否能找到正确答案"""
    print("\n" + "=" * 70)
    print("  4. 检索质量诊断")
    print("=" * 70)
    
    # 测试问题 - 这些应该能在文档中找到答案
    test_cases = [
        {
            "query": "What was Flex's capital expenditure in fiscal year 2024?",
            "expected_keywords": ["flex", "capital", "expenditure", "property", "equipment"],
            "expected_company": "Flex",
        },
        {
            "query": "Compare Jabil and Celestica revenue",
            "expected_keywords": ["revenue", "net sales"],
            "expected_company": None,  # Should find multiple companies
        },
        {
            "query": "What are Sanmina's manufacturing facilities?",
            "expected_keywords": ["sanmina", "facility", "plant", "manufacturing"],
            "expected_company": "Sanmina",
        },
    ]
    
    from backend.rag.retriever import search_documents
    
    issues = []
    for case in test_cases:
        print(f"\n  查询: {case['query'][:50]}...")
        
        # 暂时禁用 reranking 以测试原始检索
        docs = search_documents(case["query"], n_results=5, use_reranking=False)
        
        if not docs:
            issues.append(f"❌ 未找到任何结果: {case['query'][:30]}...")
            continue
        
        # 检查结果质量
        top_doc = docs[0]
        content_lower = top_doc["content"].lower()
        
        found_keywords = sum(1 for kw in case["expected_keywords"] if kw in content_lower)
        keyword_ratio = found_keywords / len(case["expected_keywords"])
        
        company_match = True
        if case["expected_company"]:
            company_match = top_doc["company"] == case["expected_company"]
        
        status = "✅" if keyword_ratio > 0.3 and company_match else "⚠️"
        print(f"    {status} Top 结果: [{top_doc['company']}] {top_doc['source']}")
        print(f"       相似度: {top_doc['similarity']:.3f}")
        print(f"       关键词匹配: {found_keywords}/{len(case['expected_keywords'])}")
        
        if keyword_ratio < 0.3:
            issues.append(f"⚠️ 关键词匹配率低: {case['query'][:30]}...")
        if not company_match:
            issues.append(f"⚠️ 公司不匹配: 期望 {case['expected_company']}, 得到 {top_doc['company']}")
    
    if issues:
        print(f"\n  发现 {len(issues)} 个检索问题:")
        for issue in issues:
            print(f"    {issue}")
        return False
    
    print("\n  ✅ 检索质量良好")
    return True


# ===========================================================================
# 5. 端到端测试
# ===========================================================================
def diagnose_end_to_end():
    """完整的问答测试"""
    print("\n" + "=" * 70)
    print("  5. 端到端问答测试")
    print("=" * 70)
    
    try:
        from backend.rag.pipeline import process_query_sync
    except ImportError as e:
        print(f"  ❌ 无法导入 pipeline: {e}")
        return False
    
    test_query = "What was Flex's capital expenditure in the most recent fiscal year?"
    
    print(f"\n  测试问题: {test_query}")
    print("  正在处理...")
    
    try:
        result = process_query_sync(
            query=test_query,
            mode="rag",
            use_reranking=False,  # 先测试无 reranking
        )
        
        print(f"\n  响应长度: {len(result.get('response', ''))} 字符")
        print(f"  检索到的源: {len(result.get('sources', []))} 个")
        
        response = result.get("response", "")
        
        # 检查响应质量
        has_number = any(c.isdigit() for c in response)
        has_capex = "capex" in response.lower() or "capital" in response.lower()
        has_flex = "flex" in response.lower()
        
        if has_number and has_capex and has_flex:
            print("  ✅ 响应包含关键信息 (数字 + CapEx + Flex)")
            print(f"\n  响应预览:\n  {response[:500]}...")
            return True
        else:
            print("  ⚠️ 响应可能不完整")
            print(f"    - 包含数字: {has_number}")
            print(f"    - 包含 CapEx: {has_capex}")
            print(f"    - 包含 Flex: {has_flex}")
            print(f"\n  响应预览:\n  {response[:500]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===========================================================================
# 主诊断函数
# ===========================================================================
def run_diagnostics():
    """运行所有诊断"""
    print("\n" + "=" * 70)
    print("  RAG 系统诊断工具")
    print("  用于定位问题出在哪个环节")
    print("=" * 70)
    
    results = {}
    
    # 1. PDF 解析
    results["pdf_parsing"] = diagnose_pdf_parsing()
    
    # 2. 表格提取
    results["table_extraction"] = diagnose_table_extraction()
    
    # 3. 父子结构
    results["parent_child"] = diagnose_parent_child()
    
    # 4. 检索质量
    results["retrieval"] = diagnose_retrieval()
    
    # 5. 端到端
    results["end_to_end"] = diagnose_end_to_end()
    
    # 总结
    print("\n" + "=" * 70)
    print("  诊断总结")
    print("=" * 70)
    
    all_pass = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 需关注"
        print(f"  {name:<20}: {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "-" * 70)
    if all_pass:
        print("  🎉 所有检测通过！RAG 系统运行正常。")
    else:
        print("  ⚠️  部分检测未通过，请根据上述信息定位问题：")
        print()
        print("  问题定位指南:")
        print("  - pdf_parsing 失败     → PDF 解析有问题，考虑使用 Docling/MinerU")
        print("  - table_extraction 失败 → 表格未正确提取，检查 serialize_table_enhanced")
        print("  - parent_child 失败    → 需要重新运行 build_chromadb.py")
        print("  - retrieval 失败       → 检索配置有问题，查看 retriever.py")
        print("  - end_to_end 失败      → LLM 生成有问题，检查 API key 和 prompt")
    
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    run_diagnostics()
