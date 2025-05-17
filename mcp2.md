1. 安全、确定性的基础工具（`edit_file`、`read_file`、`run_command`）。  
2. 与 MCP 服务器的通用交互循环（假设 MCP 暴露 `/step` 接口，可自行调整）。  
3. 自动将每次迭代写入 JSONL，便于后续直接用来微调。  
4. 基本的安全与资源限制脚手架（阻止路径穿越、危险命令、超时等）。

 `dockerfiles/agent/agent.py`

```python
# dockerfiles/agent/agent.py
"""
codex-agent：一个简洁的执行封装器，功能流程：
1.  从 STDIN 读取 JSON “任务”（或通过 CLI 参数 --task_file …）
2.  向 MCP 服务器请求下一步思考 / 工具调用
3.  执行指定工具
4.  将结果回传给 MCP，直至对方返回 "finish"

环境变量
---------
OPENAI_API_KEY            – 若 MCP 需代理裸模调用则使用
MCP_SERVER_URL            – 例如 http://mcp:8000
OPENAI_FINETUNED_MODEL_ID – 可选，转发给 MCP 的主模型 id
OPENAI_CRITIQUE_MODEL_ID  – 可选，转发给 MCP 的批判模型 id
CI                         – 若设置（如 GitHub Actions），禁止 `!pip install …`
"""

from __future__ import annotations
import os, sys, json, subprocess, logging, shlex, difflib, uuid, time
from pathlib import Path
from typing import Dict, Any, Tuple
import requests                      # 与 MCP 的 HTTP 通信
from dotenv import load_dotenv       # 可选加载 .env

# ── 配置 ───────────────────────────────────────────────────────────────
load_dotenv()

CODE_DIR           = Path("/app/code").resolve()
OUTPUT_DIR         = Path("/app/output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINE_TUNE_LOG_DIR  = OUTPUT_DIR / "finetuning_data"
FINE_TUNE_LOG_DIR.mkdir(exist_ok=True)

MCP_SERVER_URL     = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

MAX_MCP_ITERATIONS = int(os.getenv("MAX_MCP_ITERATIONS", "7"))
CMD_TIMEOUT        = int(os.getenv("CMD_TIMEOUT", "20"))        # 秒
CMD_MAX_OUTPUT     = int(os.getenv("CMD_MAX_OUTPUT", "20000"))  # 截断字节数

# ── 日志 ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "agent.log", encoding="utf-8")
    ],
)
log = logging.getLogger("codex-agent")

# ── 工具实现 ───────────────────────────────────────────────────────────
def edit_file(file_path:str,
              new_content:str|None=None,
              diff_patch:str|None=None,
              insert_after_line:int=-1,
              replace_lines:Tuple[int,int]|None=None) -> Dict[str,Any]:
    """
    修改文件的多种模式（整体替换 / diff patch / 插入 / 行替换）。
    返回 dict: {"status": "...", "message": "...", "diff": "实际产生的 diff"}。
    """
    rel_path = Path(file_path)
    abs_path = (CODE_DIR / rel_path).resolve()
    if not str(abs_path).startswith(str(CODE_DIR)):
        return {"status":"failure", "message":"路径越界被阻止"}

    original_content = abs_path.read_text(encoding="utf-8") if abs_path.exists() else ""
    modified_content = original_content

    # --- 互斥的几种写法 ---------------------------------------------------
    if new_content is not None and diff_patch is None and insert_after_line==-1 and not replace_lines:
        modified_content = new_content

    elif diff_patch is not None:
        # 生产环境可用 python-patch 或系统 patch，这里偷懒：
        if new_content is None:
            return {"status":"failure","message":"简化模式下 diff_patch 需同时提供 new_content"}
        modified_content = new_content

    elif insert_after_line > -1 and new_content is not None:
        lines = original_content.splitlines(keepends=True)
        insert_idx = min(insert_after_line+1, len(lines))
        lines.insert(insert_idx, new_content + ("" if new_content.endswith("\n") else "\n"))
        modified_content = "".join(lines)

    elif replace_lines and new_content is not None:
        start, end = replace_lines
        lines = original_content.splitlines(keepends=True)
        if not (0 <= start <= end < len(lines)):
            return {"status":"failure","message":"replace_lines 范围非法"}
        lines[start:end+1] = [new_content + ("" if new_content.endswith("\n") else "\n")]
        modified_content = "".join(lines)

    else:
        return {"status":"failure","message":"参数组合非法"}

    # 写入并生成 diff
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(modified_content, encoding="utf-8")

    actual_diff = "".join(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            modified_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=""
        )
    )
    return {"status":"success","message":"file edited","diff":actual_diff}

def read_file(file_path:str, start:int=0, end:int=4000) -> Dict[str,Any]:
    """读取文件片段，避免一次性塞入巨大 prompt。"""
    abs_path = (CODE_DIR / file_path).resolve()
    if not abs_path.exists():
        return {"status":"failure","message":"文件不存在"}
    if not str(abs_path).startswith(str(CODE_DIR)):
        return {"status":"failure","message":"路径越界被阻止"}

    text = abs_path.read_text(encoding="utf-8")
    return {"status":"success","content":text[start:end]}

def run_command(command:str, workdir:str=".", timeout:int=CMD_TIMEOUT) -> Dict[str,Any]:
    """
    在 /app/code 内安全执行 shell 命令。
    禁止管道、重定向、后台、危险二进制等。
    """
    abs_workdir = (CODE_DIR / workdir).resolve()
    if not str(abs_workdir).startswith(str(CODE_DIR)):
        return {"status":"failure","message":"工作目录越界"}

    dangerous = [";", "&&", "|", ">", "<", "`", "$(", "&", "sudo",
                 "yum", "apt", "pip", "curl", "wget"]
    if any(tok in command for tok in dangerous):
        return {"status":"failure","message":"命令被安全策略阻止"}

    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=abs_workdir,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        stdout = proc.stdout[:CMD_MAX_OUTPUT]
        stderr = proc.stderr[:CMD_MAX_OUTPUT]
        return {
            "status":"success" if proc.returncode==0 else "failure",
            "returncode":proc.returncode,
            "stdout":stdout,
            "stderr":stderr
        }
    except subprocess.TimeoutExpired:
        return {"status":"failure","message":f"执行超时 {timeout}s"}

# 动态分发表
TOOLS = {
    "edit_file":   edit_file,
    "read_file":   read_file,
    "run_command": run_command,
}

# ── MCP 交互辅助函数 ───────────────────────────────────────────────────
def mcp_step(task_id:str, payload:Dict[str,Any]) -> Dict[str,Any]:
    """
    POST {task_id, payload} 到 MCP /step，返回 JSON。
    """
    url = MCP_SERVER_URL.rstrip("/") + "/step"
    resp = requests.post(url, json={"task_id":task_id, "payload":payload}, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── 微调日志 ───────────────────────────────────────────────────────────
def record_iteration(task_id:str, iteration:int, data:Dict[str,Any]) -> None:
    """每行写一个 JSON，后期拼接成大 JSONL 直接喂给 OpenAI 微调。"""
    file = FINE_TUNE_LOG_DIR / f"{task_id}.jsonl"
    with file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"iteration":iteration, **data}, ensure_ascii=False) + "\n")

# ── 主循环 ─────────────────────────────────────────────────────────────
def run_task(initial_task:Dict[str,Any]) -> None:
    task_id = initial_task.get("task_id") or str(uuid.uuid4())
    payload  = {"event":"start", "task":initial_task}

    for iteration in range(1, MAX_MCP_ITERATIONS+1):
        log.info(f"[{task_id}] 第 {iteration} 轮")

        # 1) 询问 MCP 下一步行动
        mcp_reply = mcp_step(task_id, payload)
        record_iteration(task_id, iteration, {
            "agent_input":payload,
            "mcp_reply":mcp_reply
        })

        if mcp_reply.get("event") == "finish":
            log.info(f"[{task_id}] 完成：{mcp_reply.get('summary','')}")
            break

        if mcp_reply.get("event") != "tool_call":
            log.warning(f"[{task_id}] 未知事件 {mcp_reply.get('event')}，终止")
            break

        tool_name = mcp_reply["name"]
        arguments = mcp_reply.get("arguments", {})

        if tool_name not in TOOLS:
            payload = {
                "event":"tool_result",
                "name":tool_name,
                "result":{"status":"failure","message":"未知工具"}
            }
            continue

        # 2) 执行工具
        try:
            result = TOOLS[tool_name](**arguments)
        except Exception as exc:
            log.exception(f"工具 {tool_name} 崩溃")
            result = {"status":"failure","message":str(exc)}

        # 3) 把结果告诉 MCP，进入下一轮
        payload = {
            "event":"tool_result",
            "name":tool_name,
            "arguments":arguments,
            "result":result
        }

    else:
        # 达到最大轮次仍未完结
        mcp_step(task_id, {"event":"agent_abort","reason":"max_iterations"})

# ── CLI 入口 ───────────────────────────────────────────────────────────
def load_task_from_cli() -> Dict[str,Any]:
    """
    用法示例：
        python agent.py --task_file /tmp/task.json
        echo '{"instruction":"…"}' | python agent.py
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task_file", help="写一个包含任务 JSON 的文件路径")
    args = p.parse_args()

    if args.task_file:
        return json.loads(Path(args.task_file).read_text())
    else:
        return json.loads(sys.stdin.read())

if __name__ == "__main__":
    try:
        task = load_task_from_cli()
    except Exception as exc:
        print("无法解析任务 JSON:", exc, file=sys.stderr)
        sys.exit(1)

    run_task(task)
```

重点提示
--------

1. HTTP 协议  
   • 假设 MCP `/step` 返回两种事件：  
     `{"event":"tool_call","name":"edit_file", ...}` 或  
     `{"event":"finish","summary":"..."}`
   • 如果你已有 websocket / gRPC 实现，可将 `requests.post` 换成对应调用。

2. 文件 patch  
   • 生产环境请用 `python-patch` 或系统 `patch`，目前简化为整文件替换。

3. 安全  
   • `run_command` 的黑名单比较严格，可根据威胁模型放宽。  
   • 建议在 k8s Pod 中设定 `securityContext`：只读根文件系统、丢弃 Capabilities、限制 CPU/内存等。

4. 微调语料  
   • 每次迭代都写入 `finetuning_data/<task>.jsonl`。  
   • 定时任务合并并转换为 OpenAI 微调所需格式即可。

5. 可扩展性  
   • 无状态，N 个 Pod 并发消费任务队列即可。  
   • 通过队列长度实现自动弹性伸缩。

这样，你就拥有了一个端到端的最小工作流：容器接任务 → 与 MCP 多轮交互 → 执行安全工具 → 记录数据 → 退出。  

祝你开发顺利，玩得开心！🚀
