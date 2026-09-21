# RAG 评估：从模拟到真实的完整接入指南

> 本文档写给 AI 开发者，当 RAG 系统开发完成后，按此文档改造评估脚本即可直接跑通真实评估。
> 当前状态：模拟数据 → 目标状态：接入真实 RAG 系统

---

## 一、当前已就绪的东西

| 组件 | 位置（服务器 `10.80.153.12`） | 说明 |
|------|-------------------------------|------|
| Docker 评估镜像 | `~/ragas/rag-eval:latest` | 含 RAGAS 0.4.3 + 评估脚本 |
| 评估脚本 | `~/ragas/evaluate_rag.py` | 调用 RAGAS 官方 API |
| 测试数据 | `~/ragas/rag-test-data.json` | 28 条财务问题 + 标准答案 |
| pgvector | 宿主机 5433 端口，Docker 网络 `infra-net` | 向量数据库，待导入文档 |

---

## 二、改造入口

当前 `evaluate_rag.py` 中 **第 50-60 行** 是模拟数据：

```python
# ====== ↓ 把这整块替换为真实的 RAG 检索 + 生成 ======
mock_answers = ground_truths
mock_contexts = [[gt] for gt in ground_truths]
# ====== ↑ 替换这块 ======
```

## 三、替换方案（二选一）

### 方案 A：把 RAG 逻辑直接写在评估脚本里（推荐起步）

```python
# ============ 新增：RAG 引擎 ============
import httpx
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from pgvector.sqlalchemy import Vector

class RAGEngine:
    """RAG 检索引擎：连 pgvector 搜 chunks + 调 LLM 生成回答"""
    
    def __init__(self):
        # pgvector 连接（Docker 内用容器名 pgvector-rag）
        self.db_url = (
            f"postgresql+asyncpg://"
            f"{os.getenv('PG_USER', 'root')}:{os.getenv('PG_PASSWORD', 'QxWCw2zBlN1pGEx0QZahH/6WaaM=')}"
            f"@{os.getenv('PG_HOST', 'pgvector-rag')}:{os.getenv('PG_PORT', '5432')}"
            f"/{os.getenv('PG_DB', 'ragdb')}"
        )
        self.engine = create_async_engine(self.db_url)
        
        # LLM 端点（项目中已配置的 Qwen3 或其他模型）
        self.llm_api = os.getenv('LLM_API', 'http://host.docker.internal:80/llm/qwen36b/v1')
        self.llm_model = os.getenv('LLM_MODEL', '/models/Qwen3.6-35B-A3B')
        
        # 嵌入 API（项目中已配置的 BGE-M3）
        self.embed_api = os.getenv('EMBED_API', 'http://host.docker.internal:80/embeddings/v1/embeddings')
    
    async def embed(self, text: str) -> list:
        """调用 BGE-M3 将文本转为向量"""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.embed_api,
                json={"input": text, "model": "BAAI/bge-m3"}
            )
            return resp.json()["data"][0]["embedding"]
    
    async def retrieve(self, question: str, top_k: int = 10) -> list[str]:
        """从 pgvector 检索最相似的 chunks"""
        q_vec = await self.embed(question)
        sql = """
            SELECT text FROM doc_collection_1
            ORDER BY embedding <=> :query_vec
            LIMIT :top_k
        """
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                sql, {"query_vec": q_vec, "top_k": top_k}
            )
            return [row.text for row in result]
    
    async def query(self, question: str) -> str:
        """检索 + 生成完整流程"""
        contexts = await self.retrieve(question)
        prompt = f"基于以下资料回答问题。如果资料中没有相关信息，请如实说不知道。\n\n资料：\n{chr(10).join(contexts)}\n\n问题：{question}"
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.llm_api}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=60,
            )
            return resp.json()["choices"][0]["message"]["content"]

# ============ 替换模拟数据 ============
rag = RAGEngine()
real_answers = []
real_contexts = []

for q in questions:
    print(f"  [{len(real_answers)+1}/{len(questions)}] {q[:40]}...")
    ctx = await rag.retrieve(q)
    ans = await rag.query(q)
    real_answers.append(ans)
    real_contexts.append(ctx)

mock_answers = real_answers
mock_contexts = real_contexts
```

### 方案 B：封装为独立模块（推荐长期维护）

```
~/ragas/
├── Dockerfile.eval
├── evaluate_rag.py          ← 评估入口（只调 RAGAS，不写业务逻辑）
├── rag-test-data.json
├── rag_engine.py            ← 新增：RAG 引擎封装（方案A的 RAGEngine 类）
└── config.py                ← 新增：连接配置
```

`evaluate_rag.py` 改造为：

```python
from rag_engine import RAGEngine

rag = RAGEngine()
real_answers = []
real_contexts = []

for q in questions:
    ctx = await rag.retrieve(q)
    ans = await rag.query(q)
    real_answers.append(ans)
    real_contexts.append(ctx)
```

---

## 四、环境变量说明

在运行评估时通过 `-e` 传入：

```bash
docker run --rm \
  -v ~/ragas:/app \
  --network=infra-net \
  -e PG_HOST=pgvector-rag \
  -e PG_PASSWORD=QxWCw2zBlN1pGEx0QZahH/6WaaM= \
  -e LLM_API=http://host.docker.internal:80/llm/qwen36b/v1 \
  rag-eval:latest
```

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PG_HOST` | `pgvector-rag` | pgvector 容器名（Docker 网络内） |
| `PG_PORT` | `5432` | pgvector 端口（容器内部） |
| `PG_USER` | `root` | 数据库用户 |
| `PG_PASSWORD` | 无默认 | 数据库密码 |
| `PG_DB` | `ragdb` | 数据库名 |
| `TABLE_NAME` | `doc_collection_1` | 向量表名 |
| `LLM_API` | `http://host.docker.internal:80/llm/qwen36b/v1` | LLM 服务地址 |
| `LLM_MODEL` | `/models/Qwen3.6-35B-A3B` | 模型名 |
| `EMBED_API` | `http://host.docker.internal:80/embeddings/v1/embeddings` | 嵌入服务地址 |

---

## 五、Dockerfile 补充依赖

如果方案 A/B 中新增了 `sqlalchemy`、`asyncpg`、`pgvector`、`httpx` 等依赖，需要在 Dockerfile 中补充：

```dockerfile
# 在现有 RUN pip install ragas==0.4.3 之后追加
RUN pip install --no-cache-dir \
    sqlalchemy[asyncio] \
    asyncpg \
    pgvector \
    httpx
```

然后重新构建：

```bash
docker build -f Dockerfile.eval -t rag-eval:latest .
```

---

## 六、跑真实评估的命令

```bash
# 进入服务器
ssh caojialin@10.80.153.12
cd ~/ragas

# 构建（如果改了 Dockerfile）
docker build -f Dockerfile.eval -t rag-eval:latest .

# 运行（加 infra-net 网络 + 环境变量）
docker run --rm \
  -v ~/ragas:/app \
  --network=infra-net \
  -e PG_PASSWORD=QxWCw2zBlN1pGEx0QZahH/6WaaM= \
  rag-eval:latest
```

---

## 七、预期输出示例（真实数据时）

```
测试数据: 28 条
源文件数: 8

  [1/28] 项目商业计划审批表的操作流程是怎样的？...
  [2/28] ECN审核中如果生产地公司选错了怎么办？...
  ...

============================================================
RAGAS 评估结果
============================================================
  non_llm_context_precision_with_reference: 0.7234
  non_llm_context_recall: 0.6512

分数解读:
  0.72 精准度 → 检索回来的 chunk 里约 7 成是有用的
  0.65 召回率 → 需要的信息约 6 成 5 被检索到了
```

---

## 八、常见问题

**Q: 连不上 pgvector 怎么办？**
A: 确认加了 `--network=infra-net`，且 `PG_HOST=pgvector-rag`（容器名，非 localhost）

**Q: 提示 `ModuleNotFoundError: No module named 'asyncpg'`**
A: Dockerfile 缺少依赖，参考第五节补充后重新构建

**Q: 文档还没入库 pgvector，如何开始？**
A: 先跑项目已有的 `DocumentProcessingPipeline.process()`，将 `RAG-data` 目录下的财务文档导入 pgvector

**Q: 评估结果偏低（如 <0.5）说明什么？**
A: 需要优化检索策略：调整 `chunk_size`、`chunk_overlap`、`top_k`，或更换嵌入模型

---

> 📌 完成 RAG 系统开发后，按本文档替换 evaluate_rag.py 中标记的区域，
> 调整 Dockerfile 补充依赖，即可从模拟评估切换到真实评估。