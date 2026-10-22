"""
人工校验辅助脚本（51 条分层抽样方案）

子命令：
  export   导出 51 条待校验 CSV（高分歧文本 21 条 + 负例 30 条）
  verify   跑负例检索 similarity 验证（自动检测假负例）
  merge    校验完 CSV 后一键合并修正回 eval_dataset.json
  stats    算错误率与维度统计

CSV 格式：
  id,query,expected_answer,risk_type,review_decision,fix_notes
  review_decision: accept / fix / reject
  fix_notes: 修正后的 expected_answer（仅 fix 时填）

用法示例：
  python experiments/manual_review_helper.py export
  python experiments/manual_review_helper.py verify
  # 人工编辑 data/stats/manual_review_51.csv 打标签
  python experiments/manual_review_helper.py merge
  python experiments/manual_review_helper.py stats
"""
import sys
import json
import time
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_PATH = PROJECT_ROOT / "data/stats/eval_dataset.json"
MERGED_PATH = PROJECT_ROOT / "data/stats/text_subset_merged.json"
REVIEW_CSV = PROJECT_ROOT / "data/stats/manual_review_51.csv"
VERIFY_REPORT = PROJECT_ROOT / "data/stats/negative_verify_report.json"

# 高分歧阈值（与 merge_three_answers.py 对齐）
LOW_AGREEMENT_THRESHOLD = 0.6


def cmd_export():
    """导出 51 条待校验 CSV"""
    with open(MERGED_PATH, encoding="utf-8") as f:
        merged = json.load(f)
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_ds = json.load(f)

    # 高分歧文本：agreement_score < 0.6
    high_risk = [
        s for s in merged["samples"]
        if s["arbitration"].get("agreement_score", 0) < LOW_AGREEMENT_THRESHOLD
    ]
    print(f"[导出] 高分歧文本: {len(high_risk)} 条")

    # 负例
    negatives = eval_ds.get("negative_queries", [])
    print(f"[导出] 负例: {len(negatives)} 条")

    # 写 CSV
    with open(REVIEW_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "query", "expected_answer", "risk_type",
            "review_decision", "fix_notes",
            "conflict_points", "agreement_score", "category",
        ])
        # 高分歧文本
        for s in high_risk:
            arb = s["arbitration"]
            merged_answer = arb.get("merged_answer", "")
            if isinstance(merged_answer, list):
                merged_answer = "; ".join(merged_answer)
            conflicts = " | ".join(arb.get("conflict_points", []))
            w.writerow([
                s["id"], s["query"], merged_answer,
                f"高分歧(score={arb.get('agreement_score',0):.2f})",
                "",  # review_decision 待人工填
                "",  # fix_notes 待人工填
                conflicts,
                arb.get("agreement_score", 0),
                "",
            ])
        # 负例
        for n in negatives:
            w.writerow([
                n["id"], n["query"], n.get("expected_answer", ""),
                f"负例[{n.get('category','')}]",
                "", "",
                "",
                "",
                n.get("category", ""),
            ])

    print(f"[导出] 已写入: {REVIEW_CSV}")
    print(f"[导出] 共 {len(high_risk)+len(negatives)} 条")
    print(f"[导出] 下一步: 用 Excel/Numbers 打开 CSV，在 review_decision 列填 accept/fix/reject")
    print(f"[导出] 若填 fix，在 fix_notes 列写正确 expected_answer")


def cmd_verify():
    """跑负例检索 similarity 验证，检测假负例"""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from app import state
    from app.core.retrieval import hybrid_retrieve

    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_ds = json.load(f)
    negatives = eval_ds.get("negative_queries", [])
    print(f"[验证] 负例数: {len(negatives)}")

    print("[验证] 加载 Qdrant 资源...")
    state.load_resources()
    print(f"[验证] metadata 共 {len(state.get_metadata())} 条")

    results = []
    t0 = time.time()
    for i, n in enumerate(negatives, 1):
        refs = hybrid_retrieve(
            n["query"], state.get_qdrant_client(),
            k=3, allowed_types={"text", "diverse_question"},
            enable_rewrite=True, enable_rerank=True,
        )
        top_sim = max([r.get("similarity", 0) for r in refs], default=0)
        top_ref = refs[0] if refs else None
        flag = "SUSPECT" if top_sim > 0.5 else "ok"
        results.append({
            "id": n["id"],
            "query": n["query"],
            "top_similarity": round(top_sim, 4),
            "top_source": top_ref.get("source", "") if top_ref else "",
            "top_text": (top_ref.get("text", "") or top_ref.get("content", ""))[:150] if top_ref else "",
            "flag": flag,
        })
        print(f"  [{i}/{len(negatives)}] sim={top_sim:.3f} {flag} | {n['query'][:40]}")

    print(f"\n[验证] 耗时 {time.time()-t0:.1f}s")
    suspects = [r for r in results if r["flag"] == "SUSPECT"]
    print(f"[验证] 假负例嫌疑: {len(suspects)} 条 (sim > 0.5)")
    for s in suspects:
        print(f"  [{s['id']}] sim={s['top_similarity']:.3f}")
        print(f"    Q: {s['query']}")
        print(f"    Top hit: {s['top_source']}")
        print(f"    Text: {s['top_text'][:80]}")

    with open(VERIFY_REPORT, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "suspects": len(suspects),
            "results": results,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)
    print(f"[验证] 报告已写入: {VERIFY_REPORT}")


def cmd_merge():
    """合并 CSV 校验结果回 eval_dataset.json"""
    if not REVIEW_CSV.exists():
        print(f"[合并] 错误: {REVIEW_CSV} 不存在，请先 export")
        return

    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_ds = json.load(f)

    # 读 CSV
    decisions = {}
    with open(REVIEW_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["id"]
            decision = row.get("review_decision", "").strip().lower()
            fix_notes = row.get("fix_notes", "").strip()
            if decision:
                decisions[rid] = {"decision": decision, "fix_notes": fix_notes}

    print(f"[合并] CSV 决策数: {len(decisions)}")

    # 应用决策
    stats = {"accept": 0, "fix": 0, "reject": 0, "skip": 0}
    reject_ids = set()

    # 1. 文本子集：fix 改 expected_answer，reject 从 eval_dataset.complex_queries 删除
    # merged samples 不在 eval_dataset，需要从 eval_dataset.complex_queries 找对应 V6-Txxx
    complex_by_id = {q["id"]: q for q in eval_ds["complex_queries"]}
    for rid, d in decisions.items():
        if rid.startswith("text_"):
            # 对应 eval_dataset 中的 V6-Txxx
            new_id = f"V6-T{rid.replace('text_','').lstrip('0').zfill(3)}"
            if new_id in complex_by_id:
                if d["decision"] == "fix" and d["fix_notes"]:
                    complex_by_id[new_id]["expected_answer"] = d["fix_notes"]
                    stats["fix"] += 1
                    print(f"  [fix] {new_id}: 更新 expected_answer")
                elif d["decision"] == "reject":
                    reject_ids.add(new_id)
                    stats["reject"] += 1
                    print(f"  [reject] {new_id}: 待删除")
                else:
                    stats["accept"] += 1
            else:
                stats["skip"] += 1

    # 2. 负例：fix 改 expected_answer，reject 删除
    neg_by_id = {q["id"]: q for q in eval_ds.get("negative_queries", [])}
    for rid, d in decisions.items():
        if rid.startswith("N"):
            if rid in neg_by_id:
                if d["decision"] == "fix" and d["fix_notes"]:
                    neg_by_id[rid]["expected_answer"] = d["fix_notes"]
                    stats["fix"] += 1
                    print(f"  [fix] {rid}: 更新 expected_answer")
                elif d["decision"] == "reject":
                    reject_ids.add(rid)
                    stats["reject"] += 1
                    print(f"  [reject] {rid}: 待删除")
                else:
                    stats["accept"] += 1

    # 删除 reject 的样本
    if reject_ids:
        eval_ds["complex_queries"] = [
            q for q in eval_ds["complex_queries"] if q["id"] not in reject_ids
        ]
        if "negative_queries" in eval_ds:
            eval_ds["negative_queries"] = [
                q for q in eval_ds["negative_queries"] if q["id"] not in reject_ids
            ]
        print(f"[合并] 已删除 {len(reject_ids)} 条: {sorted(reject_ids)}")

    # 更新 metadata
    meta = eval_ds.get("metadata", {})
    meta["v6_manual_review_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["v6_review_stats"] = stats
    meta["v6_review_rejected"] = sorted(reject_ids)
    meta["total_complex"] = len(eval_ds["complex_queries"])
    meta["total_negative"] = len(eval_ds.get("negative_queries", []))
    eval_ds["metadata"] = meta

    # 写回
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_ds, f, ensure_ascii=False, indent=2)
    print(f"\n[合并] 完成: {stats}")
    print(f"[合并] 当前总数: {len(eval_ds['complex_queries']) + len(eval_ds.get('cross_modal_queries', [])) + len(eval_ds.get('negative_queries', [])) + len(eval_ds.get('adversarial_queries', []))}")


def cmd_stats():
    """算错误率与维度统计"""
    if not REVIEW_CSV.exists():
        print(f"[统计] 错误: {REVIEW_CSV} 不存在")
        return
    with open(REVIEW_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    if total == 0:
        print("[统计] CSV 为空")
        return
    decisions = [r.get("review_decision", "").strip().lower() for r in rows]
    from collections import Counter
    c = Counter(decisions)
    accept = c.get("accept", 0)
    fix = c.get("fix", 0)
    reject = c.get("reject", 0)
    blank = total - accept - fix - reject
    error_rate = (fix + reject) * 100 // total
    print(f"=== 校验统计 ===")
    print(f"总数: {total}")
    print(f"accept: {accept} ({accept*100//total}%)")
    print(f"fix:    {fix} ({fix*100//total}%)")
    print(f"reject: {reject} ({reject*100//total}%)")
    print(f"未填:  {blank}")
    print(f"错误率: {error_rate}%")
    print()
    if error_rate < 5:
        verdict = "优秀 -> 直接进入 A5 评测"
    elif error_rate < 10:
        verdict = "可接受 -> 修正后进入 A5 评测"
    else:
        verdict = "高 -> 返工重新生成"
    print(f"判定: {verdict}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "export":
        cmd_export()
    elif cmd == "verify":
        cmd_verify()
    elif cmd == "merge":
        cmd_merge()
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
