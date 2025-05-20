import openai
import os
import sys
import difflib
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import json
import uuid
from typing import Dict, Any
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# Load .env file from /app (if it exists, for local testing) or rely on Docker env vars
dotenv_path = Path('/app/.env')
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # Can be overridden by user via API
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o") #

CODE_DIR = Path("/app/code").resolve()
OUTPUT_DIR = Path("/app/output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINE_TUNE_LOG_DIR = OUTPUT_DIR / "finetuning_data"
FINE_TUNE_LOG_DIR.mkdir(exist_ok=True)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MAX_MCP_ITERATIONS = int(os.getenv("MAX_MCP_ITERATIONS", "7"))
CMD_TIMEOUT = int(os.getenv("CMD_TIMEOUT", "20"))
CMD_MAX_OUTPUT = int(os.getenv("CMD_MAX_OUTPUT", "20000"))

# --- Logging ---
log_file_path = OUTPUT_DIR / "agent.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, encoding="utf-8")
    ]
)
log = logging.getLogger("codex-agent")

# --- Helper Functions ---
def run_command(command, cwd=None, env=None):
    logging.info(f"Running command: {' '.join(command)} in {cwd or '.'}")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit, we'll check returncode
            cwd=cwd,
            env=env
        )
        if process.stdout:
            logging.info(f"Stdout: {process.stdout.strip()}")
        if process.stderr:
            logging.error(f"Stderr: {process.stderr.strip()}")
        if process.returncode != 0:
            logging.error(f"Command failed with exit code {process.returncode}")
        return process
    except Exception as e:
        logging.error(f"Exception running command {' '.join(command)}: {e}")
        return None


def get_llm_response(prompt_text):
    logging.info("Sending request to OpenAI API...")
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}]
        )
        content = response.choices[0].message.content
        # Strip markdown code block delimiters if present
        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.splitlines()[1:-1])
        logging.info("Received response from OpenAI API.")
        return content
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None

def create_github_pr(repo_path, target_file_rel_path, task_description, original_branch="main"):
    if not GITHUB_TOKEN:
        logging.warning("GITHUB_TOKEN not provided. Skipping PR creation.")
        return False

    # Check if it's a git repo and we have a remote origin
    if not (repo_path / ".git").is_dir():
        logging.warning(f"{repo_path} is not a git repository. Skipping PR creation.")
        return False
    
    remote_url_proc = run_command(["git", "config", "--get", "remote.origin.url"], cwd=repo_path)
    if not remote_url_proc or remote_url_proc.returncode != 0 or not remote_url_proc.stdout.strip():
        logging.warning("No remote 'origin' found or git command failed. Skipping PR creation.")
        return False
    remote_url = remote_url_proc.stdout.strip()
    if "github.com" not in remote_url:
        logging.warning(f"Remote origin '{remote_url}' is not a GitHub URL. Skipping PR creation.")
        return False

    logging.info(f"Attempting to create PR for changes in {target_file_rel_path}")

    # Configure git user
    run_command(["git", "config", "--local", "user.email", "codex-agent@example.com"], cwd=repo_path)
    run_command(["git", "config", "--local", "user.name", "Codex Agent"], cwd=repo_path)

    # Authenticate gh CLI
    with open("/tmp/gh_token", "w") as f:
        f.write(GITHUB_TOKEN)
    auth_proc = run_command(["gh", "auth", "login", "--with-token"], cwd=repo_path, env={**os.environ, "GH_TOKEN": GITHUB_TOKEN})
    # run_command(["gh", "auth", "login", "--with-token", "<", "/tmp/gh_token"], cwd=repo_path) # This doesn't work with subprocess well
    # Instead, pass GITHUB_TOKEN as env var for gh commands
    
    if not auth_proc or auth_proc.returncode != 0:
       logging.error("GitHub CLI authentication failed.")
       # Try to remove the token file anyway
       Path("/tmp/gh_token").unlink(missing_ok=True)
       return False
    Path("/tmp/gh_token").unlink(missing_ok=True) # Clean up token file
    logging.info("GitHub CLI authenticated.")


    # Determine current branch
    current_branch_proc = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    if not current_branch_proc or current_branch_proc.returncode != 0:
        logging.error("Could not determine current branch.")
        original_branch_name = "main" # fallback
    else:
        original_branch_name = current_branch_proc.stdout.strip()
        if not original_branch_name or original_branch_name == "HEAD": # HEAD means detached
            logging.warning(f"Currently on a detached HEAD. Will try to base off 'main' or 'master'.")
            # Try to find a common base branch
            for common_branch in ["main", "master"]:
                check_branch_proc = run_command(["git", "show-branch", f"remotes/origin/{common_branch}"], cwd=repo_path)
                if check_branch_proc and check_branch_proc.returncode == 0:
                    original_branch_name = common_branch
                    logging.info(f"Using '{original_branch_name}' as base branch for PR.")
                    break
            else: # if no common branch found
                logging.error("Could not determine a suitable base branch (main/master). Skipping PR.")
                return False


    # Create a new branch
    new_branch_name = f"codex-agent-patch-{Path(target_file_rel_path).stem}-{os.urandom(3).hex()}"
    logging.info(f"Creating new branch: {new_branch_name} from {original_branch_name}")
    
    # Ensure we are on the original branch and it's up-to-date before branching
    # This part is tricky if the cloned repo is shallow or detached.
    # For simplicity, we'll assume the clone is good enough to branch from.
    # A robust solution might involve fetching before branching.
    run_command(["git", "checkout", "-b", new_branch_name, original_branch_name], cwd=repo_path)


    # Add and commit changes
    run_command(["git", "add", str(Path(target_file_rel_path).as_posix())], cwd=repo_path) # Use as_posix for path
    commit_message = f"AI Agent: {task_description[:50]}\n\nApplied automated changes to {target_file_rel_path} based on task: {task_description}"
    commit_proc = run_command(["git", "commit", "-m", commit_message], cwd=repo_path)
    if not commit_proc or commit_proc.returncode != 0:
        logging.error("Git commit failed. Maybe no changes to commit?")
        run_command(["git", "checkout", original_branch_name], cwd=repo_path) # Go back
        run_command(["git", "branch", "-D", new_branch_name], cwd=repo_path) # Delete new branch
        return False

    # Push the new branch
    push_proc = run_command(["git", "push", "-u", "origin", new_branch_name], cwd=repo_path, env={**os.environ, "GITHUB_TOKEN": GITHUB_TOKEN})
    if not push_proc or push_proc.returncode != 0:
        logging.error(f"Git push failed for branch {new_branch_name}.")
        return False

    # Create Pull Request
    pr_title = f"AI Agent: {task_description[:70]}"
    pr_body = f"This PR was automatically generated by the Codex Agent.\n\n**Task:** {task_description}\n\n**File Modified:** `{target_file_rel_path}`"
    
    pr_command = [
        "gh", "pr", "create",
        "--title", pr_title,
        "--body", pr_body,
        "--base", original_branch_name, # PR against the original branch
        "--head", new_branch_name      # From our new branch
    ]
    # If repo is public, --repo owner/repo might be needed.
    # gh usually infers it from the git remote.

    pr_proc = run_command(pr_command, cwd=repo_path, env={**os.environ, "GITHUB_TOKEN": GITHUB_TOKEN})
    if not pr_proc or pr_proc.returncode != 0:
        logging.error("Failed to create GitHub Pull Request.")
        return False

    logging.info(f"Successfully created Pull Request for branch {new_branch_name}!")
    pr_url = pr_proc.stdout.strip()
    if pr_url:
        logging.info(f"PR URL: {pr_url}")
        with open(OUTPUT_DIR / "pr_url.txt", "w") as f:
            f.write(pr_url)
    return True

# --- Main Execution ---
def main():
    logging.info("Agent script started.")
    if len(sys.argv) < 4:
        logging.error("Usage: python agent.py <task_description> <target_file_relative_path> <is_github_repo_str>")
        sys.exit(1)

    task_description = sys.argv[1]
    target_file_rel_path = sys.argv[2] # e.g. "src/main.py"
    is_github_repo = sys.argv[3].lower() == 'true'
    
    logging.info(f"Task Description: {task_description}")
    logging.info(f"Target File: {target_file_rel_path}")
    logging.info(f"Is GitHub Repo: {is_github_repo}")
    logging.info(f"OpenAI Model: {MODEL_NAME}")
    logging.info(f"GitHub Token Provided: {'Yes' if GITHUB_TOKEN else 'No'}")

    target_file_abs_path = CODE_DIR / target_file_rel_path
    if not target_file_abs_path.is_file():
        logging.error(f"Target file not found: {target_file_abs_path}")
        sys.exit(1)

    # Run setup.sh if it exists in the user's code
    setup_script_path = CODE_DIR / "setup.sh"
    if setup_script_path.exists() and setup_script_path.is_file():
        logging.info(f"Found setup.sh at {setup_script_path}, executing...")
        # Make it executable
        run_command(["chmod", "+x", str(setup_script_path)], cwd=CODE_DIR)
        # Execute it, redirecting its output to a log file
        setup_log_path = OUTPUT_DIR / "setup.log"
        with open(setup_log_path, "wb") as slf: # binary mode for subprocess output
            setup_proc = subprocess.Popen(
                [str(setup_script_path)], 
                cwd=CODE_DIR, 
                stdout=slf, 
                stderr=subprocess.STDOUT
            )
            setup_proc.communicate() # wait for it to finish
        if setup_proc.returncode == 0:
            logging.info("setup.sh executed successfully.")
        else:
            logging.warning(f"setup.sh exited with code {setup_proc.returncode}. Check setup.log.")
    else:
        logging.info("No setup.sh found in the repository root, or it's not a file.")


    with open(target_file_abs_path, "r", encoding='utf-8') as f:
        original_code = f.read()

    # Read AGENTS.MD if it exists in the user's code
    agents_md_path = CODE_DIR / "AGENTS.md"
    agent_custom_instructions = ""
    if agents_md_path.exists() and agents_md_path.is_file():
        logging.info(f"Found AGENTS.MD at {agents_md_path}, loading custom instructions.")
        with open(agents_md_path, "r", encoding='utf-8') as f:
            agent_custom_instructions = f.read()
    else:
        logging.info("No AGENTS.MD found in repository root, using default prompt structure.")
        # Fallback to example AGENTS.md content if not provided by user
        example_agents_md_path = Path(__file__).parent / "AGENTS.md.example"
        if example_agents_md_path.exists():
            with open(example_agents_md_path, "r", encoding='utf-8') as f:
                agent_custom_instructions = f.read()


    prompt = f"""{agent_custom_instructions}

User's Task Description:
{task_description}

Target File Path: {target_file_rel_path}

Current Code from '{target_file_rel_path}':
```
{original_code}
```

Please provide ONLY the complete, modified code for the file '{target_file_rel_path}'.
Do not include any explanations or conversational text outside the code.
"""

    (OUTPUT_DIR / "prompt.txt").write_text(prompt, encoding='utf-8')
    logging.info(f"Prompt saved to {OUTPUT_DIR / 'prompt.txt'}")

    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Cannot proceed with LLM call.")
        sys.exit(1)

    modified_code = get_llm_response(prompt)

    if modified_code:
        (OUTPUT_DIR / "llm_response.txt").write_text(modified_code, encoding='utf-8')
        logging.info(f"LLM response saved to {OUTPUT_DIR / 'llm_response.txt'}")

        # Generate diff
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            modified_code.splitlines(keepends=True),
            fromfile=f"a/{target_file_rel_path}",
            tofile=f"b/{target_file_rel_path}",
            lineterm=""
        )
        diff_output = "".join(diff)
        (OUTPUT_DIR / "diff.patch").write_text(diff_output, encoding='utf-8')
        logging.info(f"Diff patch saved to {OUTPUT_DIR / 'diff.patch'}")

        # Overwrite the original file with the modified code
        # This is important for the PR creation step
        with open(target_file_abs_path, "w", encoding='utf-8') as f:
            f.write(modified_code)
        logging.info(f"Original file {target_file_abs_path} updated with modified code.")

        # Attempt to create PR if it's a GitHub repo and token is available
        if is_github_repo and GITHUB_TOKEN:
            # The CODE_DIR is the root of the git repo
            # The target_file_abs_path is where the change was made
            # We need the relative path of target_file_abs_path from CODE_DIR
            # which is target_file_rel_path
            create_github_pr(CODE_DIR, target_file_rel_path, task_description)
        else:
            logging.info("Skipping PR creation (not a GitHub repo or no GITHUB_TOKEN).")

    else:
        logging.error("Failed to get a response from LLM or response was empty.")
        (OUTPUT_DIR / "error.txt").write_text("Failed to get LLM response.", encoding='utf-8')

    logging.info("Agent script finished.")

def upload_output_to_cos(output_dir_path: Path, cos_bucket: str, cos_prefix: str):
    logging.info(f"Attempting to upload contents of {output_dir_path} to COS s3://{cos_bucket}/{cos_prefix}")
    # This requires awscli or coscmd to be in the container and configured
    # For awscli (works with S3 and MinIO, and often Tencent COS with S3 compatibility)
    # Ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION are set as env vars
    # or the pod has an IAM role attached (IRSA/Workload Identity).
    # For Tencent COS native: ensure coscmd is installed and configured.

    # Example using aws s3 sync:
    # target_s3_path = f"s3://{cos_bucket}/{cos_prefix}"
    # command = ["aws", "s3", "sync", str(output_dir_path), target_s3_path, "--delete"]
    
    # Example using coscmd (Tencent Cloud):
    # Needs prior `coscmd config -a key -s secret -b default_bucket -r region` or env vars
    # Or pass credentials directly if supported by coscmd version
    target_cos_path = f"cos://{cos_bucket}/{cos_prefix}" # coscmd uses cos:// schema
    # coscmd upload -r /local/path/ cos_path/
    command = ["coscmd", "upload", "-r", str(output_dir_path) + "/", target_cos_path]


    logging.info(f"Executing COS upload command: {' '.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        logging.info(f"COS Upload STDOUT: {process.stdout}")
        if process.stderr:
            logging.warning(f"COS Upload STDERR: {process.stderr}")
        logging.info("Output successfully uploaded to COS.")
    except subprocess.CalledProcessError as e:
        logging.error(f"COS Upload failed. Exit code: {e.returncode}")
        logging.error(f"COS Upload STDOUT: {e.stdout}")
        logging.error(f"COS Upload STDERR: {e.stderr}")
    except FileNotFoundError:
        logging.error(f"COS upload command (e.g., aws or coscmd) not found. Ensure it's in the Docker image PATH.")
# --- RAG 相关 ─────────────────────────────────
def get_relevant_code_snippets(task_description, target_file_content, all_code_docs, max_snippets=3):
    logging.info("RAG: 检索相关代码片段...")
    if not all_code_docs:
        return ""
    query_text = f"{task_description}\n\nContext from target file:\n{target_file_content[:1000]}"
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(all_code_docs + [query_text])
        query_vector = tfidf_matrix[-1]
        doc_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        relevant_indices = similarities.argsort()[-max_snippets*2:][::-1]
        retrieved_snippets_text = []
        added_content = set()
        for i in relevant_indices:
            if similarities[i] > 0.05:
                snippet_content = all_code_docs[i][:300]
                first_few_words = " ".join(snippet_content.split()[:10])
                if first_few_words not in added_content:
                    retrieved_snippets_text.append(f"--- Relevant Snippet {len(retrieved_snippets_text)+1} (Similarity: {similarities[i]:.2f}) ---\n{snippet_content}\n")
                    added_content.add(first_few_words)
                if len(retrieved_snippets_text) >= max_snippets:
                    break
        if retrieved_snippets_text:
            return "\n".join(retrieved_snippets_text)
        else:
            return ""
    except Exception as e:
        logging.error(f"RAG 错误: {e}")
        return ""

def load_codebase_for_rag(code_dir_path: Path, target_file_rel_path: str):
    docs = []
    relevant_extensions = ['.py', '.go', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.cs', '.rb', '.php', '.swift', '.kt', '.rs', '.md', '.txt']
    CHUNK_SIZE_LINES = 30 
    OVERLAP_LINES = 5
    for item in code_dir_path.rglob('*'):
        if item.is_file() and item.suffix.lower() in relevant_extensions:
            try:
                with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content) > 100:
                        lines = content.splitlines()
                        if len(lines) > CHUNK_SIZE_LINES * 1.5 :
                            for i in range(0, len(lines), CHUNK_SIZE_LINES - OVERLAP_LINES):
                                chunk_lines = lines[i : i + CHUNK_SIZE_LINES]
                                if chunk_lines:
                                    chunk_content = "\n".join(chunk_lines)
                                    docs.append(f"File: {item.relative_to(code_dir_path)}\n```\n{chunk_content}\n```")
                        else:
                             docs.append(f"File: {item.relative_to(code_dir_path)}\n```\n{content}\n```")
            except Exception as e:
                logging.warning(f"RAG: 读取文件失败 {item}: {e}")
    return docs

# --- MCP 交互辅助 ─────────────────────────────
def mcp_step(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """使用 fastmcp 客户端实现 MCP 交互"""
    try:
        return mcp_client.step(task_id=task_id, payload=payload)
    except Exception as e:
        log.error(f"fastmcp 交互失败: {str(e)}")
        raise  # 保持与原有错误处理逻辑一致

def record_iteration(task_id:str, iteration:int, data:Dict[str,Any]) -> None:
    file = FINE_TUNE_LOG_DIR / f"{task_id}.jsonl"
    with file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"iteration":iteration, **data}, ensure_ascii=False) + "\n")

# --- MCP/CoT 主循环 ──────────────────────────
import os, sys, json, subprocess, logging, shlex, difflib, uuid, time
from pathlib import Path
from typing import Dict, Any, Tuple
import requests
from dotenv import load_dotenv

# --- 新增基础工具函数（来自 mcp.md 设计） ---
def edit_file(file_path: str,
              new_content: str | None = None,
              diff_patch: str | None = None,
              insert_after_line: int = -1,
              replace_lines: Tuple[int, int] | None = None) -> Dict[str, Any]:
    """修改文件的多种模式（整体替换 / diff patch / 插入 / 行替换）"""
    rel_path = Path(file_path)
    abs_path = (CODE_DIR / rel_path).resolve()
    if not str(abs_path).startswith(str(CODE_DIR)):
        return {"status": "failure", "message": "路径越界被阻止"}

    original_content = abs_path.read_text(encoding="utf-8") if abs_path.exists() else ""
    modified_content = original_content

    # 互斥模式处理
    if new_content is not None and diff_patch is None and insert_after_line == -1 and not replace_lines:
        modified_content = new_content
    elif diff_patch is not None:
        if new_content is None:
            return {"status": "failure", "message": "简化模式下 diff_patch 需同时提供 new_content"}
        modified_content = new_content
    elif insert_after_line > -1 and new_content is not None:
        lines = original_content.splitlines(keepends=True)
        insert_idx = min(insert_after_line + 1, len(lines))
        lines.insert(insert_idx, new_content + ("" if new_content.endswith("\n") else "\n"))
        modified_content = "".join(lines)
    elif replace_lines and new_content is not None:
        start, end = replace_lines
        lines = original_content.splitlines(keepends=True)
        if not (0 <= start <= end < len(lines)):
            return {"status": "failure", "message": "replace_lines 范围非法"}
        lines[start:end + 1] = [new_content + ("" if new_content.endswith("\n") else "\n")]
        modified_content = "".join(lines)
    else:
        return {"status": "failure", "message": "参数组合非法"}

    # 写入并生成 diff
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(modified_content, encoding="utf-8")
    actual_diff = "".join(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        modified_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    ))
    return {"status": "success", "message": "file edited", "diff": actual_diff}

def read_file(file_path: str, start: int = 0, end: int = 4000) -> Dict[str, Any]:
    """安全读取文件片段"""
    abs_path = (CODE_DIR / file_path).resolve()
    if not abs_path.exists():
        return {"status": "failure", "message": "文件不存在"}
    if not str(abs_path).startswith(str(CODE_DIR)):
        return {"status": "failure", "message": "路径越界被阻止"}
    
    text = abs_path.read_text(encoding="utf-8")
    return {"status": "success", "content": text[start:end]}

def run_command(command: str, workdir: str = ".", timeout: int = CMD_TIMEOUT) -> Dict[str, Any]:
    """安全执行 shell 命令（带危险命令过滤）"""
    abs_workdir = (CODE_DIR / workdir).resolve()
    if not str(abs_workdir).startswith(str(CODE_DIR)):
        return {"status": "failure", "message": "工作目录越界"}

    dangerous = [";", "&&", "|", ">", "<", "`", "$(", "&", "sudo", "yum", "apt", "pip", "curl", "wget"]
    if any(tok in command for tok in dangerous):
        return {"status": "failure", "message": "命令被安全策略阻止"}

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
            "status": "success" if proc.returncode == 0 else "failure",
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr
        }
    except subprocess.TimeoutExpired:
        return {"status": "failure", "message": f"执行超时 {timeout}s"}

TOOLS = {
    "edit_file": edit_file,
    "read_file": read_file,
    "run_command": run_command,
}

# --- 重构主流程为 MCP 交互循环 ---
def run_task(initial_task: Dict[str, Any]) -> None:
    task_id = initial_task.get("task_id") or str(uuid.uuid4())
    payload = {"event": "start", "task": initial_task}

    for iteration in range(1, MAX_MCP_ITERATIONS + 1):
        log.info(f"[{task_id}] 第 {iteration} 轮")

        # 1. 向 MCP 请求下一步操作
        mcp_reply = mcp_step(task_id, payload)
        record_iteration(task_id, iteration, {
            "agent_input": payload,
            "mcp_reply": mcp_reply
        })

        if mcp_reply.get("event") == "finish":
            log.info(f"[{task_id}] 完成：{mcp_reply.get('summary', '')}")
            break

        if mcp_reply.get("event") != "tool_call":
            log.warning(f"[{task_id}] 未知事件 {mcp_reply.get('event')}，终止")
            break

        # 2. 执行 MCP 指定的工具
        tool_name = mcp_reply["name"]
        arguments = mcp_reply.get("arguments", {})
        if tool_name not in TOOLS:
            payload = {
                "event": "tool_result",
                "name": tool_name,
                "result": {"status": "failure", "message": "未知工具"}
            }
            continue

        try:
            result = TOOLS[tool_name](**arguments)
        except Exception as exc:
            log.exception(f"工具 {tool_name} 崩溃")
            result = {"status": "failure", "message": str(exc)}

        # 3. 回传工具执行结果
        payload = {
            "event": "tool_result",
            "name": tool_name,
            "arguments": arguments,
            "result": result
        }

    else:
        mcp_step(task_id, {"event": "agent_abort", "reason": "max_iterations"})

# --- 调整 CLI 入口 ---
def load_task_from_cli() -> Dict[str, Any]:
    """支持从 STDIN 或 --task_file 读取 JSON 任务"""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task_file", help="包含任务 JSON 的文件路径")
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