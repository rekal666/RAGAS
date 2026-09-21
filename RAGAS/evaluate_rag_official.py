"""
RAG 评估脚本 — 使用官方 RAGAS 库
=================================
在 Docker（Linux）中运行，依赖 RAGAS 包已安装。

环境变量（可通过 docker -e 传入）：
  POSTGRES_HOST    默认 pgvector（docker-compose 网络中的容器名）
  POSTGRES_PORT    默认 5432（容器内部端口）
  POSTGRES_USER    默认 root
  POSTGRES_PASSWORD 必填
  POSTGRES_DB      默认 ragdb
  COLLECTION_NAME  默认 doc_collection_1
"""

import json
import os
import sys
import types

# ====== 兼容层：新版 langchain-community 拆分了 vertexai，RAGAS 0.4.3 仍从旧路径导入 ======
_chat_models = types.ModuleType('langchain_community.chat_models')
_chat_models.vertexai = types.ModuleType('langchain_community.chat_models.vertexai')
_chat_models.vertexai.ChatVertexAI = object
sys.modules['langchain_community.chat_models'] = _chat_models
sys.modules['langchain_community.chat_models.vertexai'] = _chat_models.vertexai
# ====================================================================================

from datasets import Dataset
from ragas import evaluate
from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
from ragas.metrics._context_recall import NonLLMContextRecall

context_precision = NonLLMContextPrecisionWithReference()
context_recall = NonLLMContextRecall()

# ============================================================
# 1. 加载测试数据
# ============================================================
DATA_PATH = "/app/rag-test-data.json"
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] 找不到测试数据: {DATA_PATH}")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

test_items = dataset["test_data"]
questions = [item["query"] for item in test_items]
ground_truths = [item["answer"] for item in test_items]

print(f"测试数据: {len(test_items)} 条")
print(f"源文件数: {len(dataset['meta']['source_files'])}")
print()

# ============================================================
# 2. 接入真实 RAG 系统 TODO
# ============================================================
# 这里需要替换为真实的 RAG 检索结果
# 当前用标准答案充当 mock 数据，仅用于演示 RAGAS API 调用
print("[INFO] 使用模拟数据（TODO: 接入真实 RAG 系统后替换）")
print()

mock_answers = ground_truths
mock_contexts = [[gt] for gt in ground_truths]

# ============================================================
# 3. 使用 RAGAS 评估
# ============================================================
# NonLLM 指标需要 reference_contexts 列（必须是列表格式）
mock_reference = [[gt] for gt in ground_truths]

eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": mock_answers,
    "contexts": mock_contexts,
    "reference_contexts": mock_reference,
    "ground_truth": ground_truths,
})

result = evaluate(
    eval_dataset,
    metrics=[
        context_precision,
        context_recall,
    ],
)

# ============================================================
# 4. 输出结果
# ============================================================
print("=" * 60)
print("RAGAS 评估结果")
print("=" * 60)
for metric_name, score in result._repr_dict.items():
    print(f"  {metric_name}: {score:.4f}")
print()
print(f"详细数据已写入 eval_report.csv")

# 保存详细报告
try:
    df = result.to_pandas()
    df.to_csv("/app/eval_report.csv", index=False, encoding="utf-8-sig")
except Exception as e:
    print(f"  保存 CSV 失败（不影响结果）: {e}")
print("\n✅ 评估完成")