 Agent 增加 **工具使用/函数调用 (Tool Use / Function Calling)** 

以 OpenAI 的函数调用功能为例，
```python
# Mapping of tool names to functions
AVAILABLE_TOOLS = {
    "run_shell_command": tool_run_shell_command,
    "read_file_content": tool_read_file,
    "edit_file_content": ,
    "web_search_doc_and_reference":,
    "call_api":,
    "",
}

# OpenAI function schema for the tools
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Executes a shell command in the specified directory (relative to /app/code) and returns its stdout, stderr, and exit code. Use for linters, tests, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command_string": {
                        "type": "string",
                        "description": "The shell command to execute. E.g., 'pylint main.py' or 'npm test'.",
                    },
                    "target_directory": {
                        "type": "string",
                        "description": "The directory (relative to /app/code) in which to run the command. Defaults to '.' (code root). E.g., 'src' or 'tests'.",
                        "default": "."
                    }
                },
                "required": ["command_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_content",
            "description": "Reads the content of a specified file (relative to /app/code).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file from the root of the codebase (/app/code). E.g., 'src/utils.py'.",
                    }
                },
                "required": ["file_path"],
            },
        },
    }
]
# --- End Tool Definitions ---


def get_llm_response_with_tool_calls(messages_history):
    logging.info(f"Sending request to OpenAI API with {len(messages_history)} messages (tools enabled)...")
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_history,
            tools=TOOLS_SCHEMA,
            tool_choice="auto" # Let the model decide if it wants to use a tool
        )
        return response.choices[0] # Return the full choice object
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None
```
mcp（model-context-protocol）
**核心思路：**

1.  **定义工具 (Functions)：** 在 Agent 端，我们会定义一些可用的工具，比如运行 linter、执行测试命令、甚至进行简单的文件系统操作。
2.  **向 LLM 声明工具：** 在调用 LLM 时，我们会告诉它有哪些工具可用，以及每个工具的参数和描述。
3.  **LLM 决定调用工具：** 如果 LLM 认为需要使用某个工具来回答用户的问题或完成任务，它会返回一个特殊的响应，指明要调用的工具名和参数。
4.  **Agent 执行工具：** Agent 解析 LLM 的响应，执行指定的工具，并将工具的输出结果收集起来。
5.  **将结果反馈给 LLM：** Agent 将工具的输出作为新的上下文，再次调用 LLM，让 LLM 基于工具的结果继续处理任务或给出最终答案。

---

## 🚀 升级 Agent 支持工具使用/函数调用 🚀

### 更新 `dockerfiles/agent/agent.py`

```python
import openai
import os
import sys
import difflib
import logging
import subprocess
import shutil
import json # For function calling and tool results
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer # Keep RAG for now
from sklearn.metrics.pairwise import cosine_similarity     # Keep RAG for now

# --- Configuration (as before) ---
dotenv_path = Path('/app/.env')
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o") # gpt-4, gpt-4-turbo, gpt-3.5-turbo also support func calling
MAX_CONTEXT_SNIPPET_COUNT = 3 # RAG
MAX_SNIPPET_LENGTH = 300    # RAG
MAX_TOOL_CALL_ITERATIONS = 5 # Prevent infinite loops of tool calls

CODE_DIR = Path("/app/code")
OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging (as before) ---
log_file_path = OUTPUT_DIR / "agent.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Helper Functions (run_command, create_github_pr, RAG helpers as before) ---
def run_command(command, cwd=None, env=None, shell=False): # Added shell option
    logging.info(f"Running command: {' '.join(command) if isinstance(command, list) else command} in {cwd or '.'}")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            env=env,
            shell=shell # Be cautious with shell=True
        )
        # Limit output size to prevent flooding logs/LLM context
        stdout_brief = (process.stdout or "").strip()[:1000]
        stderr_brief = (process.stderr or "").strip()[:1000]

        if stdout_brief:
            logging.info(f"Stdout (brief): {stdout_brief}")
        if stderr_brief:
            logging.error(f"Stderr (brief): {stderr_brief}")
        
        if process.returncode != 0:
            logging.error(f"Command failed with exit code {process.returncode}")
        return process # Return the whole process object
    except Exception as e:
        logging.error(f"Exception running command {' '.join(command) if isinstance(command, list) else command}: {e}")
        # Return a mock process object for consistent error handling
        return subprocess.CompletedProcess(command, -1, stderr=str(e), stdout="")


# (Keep create_github_pr, get_relevant_code_snippets, load_codebase_for_rag as before)
def create_github_pr(repo_path, target_file_rel_path, task_description, original_branch="main"):
    # ... (implementation from previous version) ...
    if not GITHUB_TOKEN:
        logging.warning("GITHUB_TOKEN not provided. Skipping PR creation.")
        return False
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
    run_command(["git", "config", "--global", "user.email", "codex-agent@example.com"], cwd=repo_path)
    run_command(["git", "config", "--global", "user.name", "Codex Agent"], cwd=repo_path)
    
    auth_proc = run_command(["gh", "auth", "login", "--with-token"], cwd=repo_path, env={**os.environ, "GITHUB_TOKEN": GITHUB_TOKEN})
    if not auth_proc or auth_proc.returncode != 0:
       logging.error("GitHub CLI authentication failed.")
       return False
    logging.info("GitHub CLI authenticated.")

    current_branch_proc = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    original_branch_name = "main"
    if current_branch_proc and current_branch_proc.returncode == 0 and current_branch_proc.stdout.strip() and current_branch_proc.stdout.strip() != "HEAD":
        original_branch_name = current_branch_proc.stdout.strip()
    else: # Fallback logic for detached HEAD or main/master
        for common_branch in ["main", "master"]:
            check_branch_proc = run_command(["git", "show-branch", f"remotes/origin/{common_branch}"], cwd=repo_path)
            if check_branch_proc and check_branch_proc.returncode == 0:
                original_branch_name = common_branch
                logging.info(f"Using '{original_branch_name}' as base branch for PR.")
                break
        else:
            logging.warning("Could not determine a suitable base branch (main/master). Defaulting to 'main'.")


    new_branch_name = f"codex-agent-patch-{Path(target_file_rel_path).stem}-{os.urandom(3).hex()}"
    logging.info(f"Creating new branch: {new_branch_name} from {original_branch_name}")
    
    checkout_main_cmd = ["git", "checkout", original_branch_name]
    if run_command(checkout_main_cmd, cwd=repo_path).returncode != 0:
        logging.warning(f"Could not checkout {original_branch_name}, attempting to fetch and retry.")
        run_command(["git", "fetch", "origin", original_branch_name], cwd=repo_path)
        if run_command(checkout_main_cmd, cwd=repo_path).returncode != 0:
            logging.error(f"Still could not checkout {original_branch_name}. PR creation might fail.")


    run_command(["git", "checkout", "-b", new_branch_name], cwd=repo_path) # Branch from current (hopefully original_branch_name)
    run_command(["git", "add", str(Path(target_file_rel_path).as_posix())], cwd=repo_path)
    commit_message = f"AI Agent: {task_description[:50]}\n\nApplied automated changes to {target_file_rel_path} based on task: {task_description}"
    commit_proc = run_command(["git", "commit", "-m", commit_message], cwd=repo_path)
    if not commit_proc or commit_proc.returncode != 0:
        logging.error("Git commit failed. Maybe no changes to commit?")
        run_command(["git", "checkout", original_branch_name], cwd=repo_path)
        run_command(["git", "branch", "-D", new_branch_name], cwd=repo_path)
        return False

    push_proc = run_command(["git", "push", "-u", "origin", new_branch_name], cwd=repo_path, env={**os.environ, "GITHUB_TOKEN": GITHUB_TOKEN})
    if not push_proc or push_proc.returncode != 0:
        logging.error(f"Git push failed for branch {new_branch_name}.")
        return False

    pr_title = f"AI Agent: {task_description[:70]}"
    pr_body = f"This PR was automatically generated by the Codex Agent.\n\n**Task:** {task_description}\n\n**File Modified:** `{target_file_rel_path}`"
    pr_command = ["gh", "pr", "create", "--title", pr_title, "--body", pr_body, "--base", original_branch_name, "--head", new_branch_name]
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

def get_relevant_code_snippets(task_description, target_file_content, all_code_docs, max_snippets=MAX_CONTEXT_SNIPPET_COUNT):
    # ... (implementation from previous version) ...
    logging.info("Attempting to find relevant code snippets for RAG...")
    if not all_code_docs:
        logging.warning("No additional code documents provided for RAG context.")
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
                snippet_content = all_code_docs[i][:MAX_SNIPPET_LENGTH]
                first_few_words = " ".join(snippet_content.split()[:10])
                if first_few_words not in added_content:
                    retrieved_snippets_text.append(f"--- Relevant Snippet {len(retrieved_snippets_text)+1} (Similarity: {similarities[i]:.2f}) ---\n{snippet_content}\n")
                    added_content.add(first_few_words)
                if len(retrieved_snippets_text) >= max_snippets:
                    break
        if retrieved_snippets_text:
            logging.info(f"Retrieved {len(retrieved_snippets_text)} relevant snippets.")
            return "\n".join(retrieved_snippets_text)
        else:
            logging.info("No sufficiently relevant snippets found via TF-IDF.")
            return ""
    except Exception as e:
        logging.error(f"Error in TF-IDF based RAG: {e}")
        return ""

def load_codebase_for_rag(code_dir_path: Path, target_file_rel_path: str):
    # ... (implementation from previous version) ...
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
                logging.warning(f"RAG: Could not read or process file {item}: {e}")
    logging.info(f"RAG: Loaded {len(docs)} documents/chunks from codebase.")
    return docs


# --- NEW: Tool Definitions & Execution ---
# Tool: Run a shell command
def tool_run_shell_command(command_string: str, target_directory: str = "."):
    """
    Executes a shell command in the specified directory (relative to the code root /app/code)
    and returns its stdout, stderr, and exit code.
    Use this for linters, tests, or other command-line tools.
    Be very careful with the commands you construct.
    Example: command_string="pylint main.py", target_directory="src"
    """
    logging.info(f"Tool: Attempting to run shell command: '{command_string}' in '{target_directory}'")
    full_target_dir = (CODE_DIR / target_directory).resolve()

    # Security check: Ensure target_directory is within CODE_DIR
    if not str(full_target_dir).startswith(str(CODE_DIR.resolve())):
        logging.error(f"Tool Error: target_directory '{target_directory}' is outside the allowed code directory.")
        return json.dumps({
            "stdout": "",
            "stderr": "Error: target_directory is outside the allowed code directory.",
            "exit_code": -1,
            "error": "Security restriction: target_directory outside allowed scope."
        })

    if not full_target_dir.is_dir():
        logging.error(f"Tool Error: target_directory '{target_directory}' does not exist or is not a directory.")
        return json.dumps({
            "stdout": "",
            "stderr": f"Error: target_directory '{target_directory}' does not exist or is not a directory.",
            "exit_code": -1,
            "error": "Directory not found."
        })

    # SECURITY NOTE: Using shell=True is risky if command_string comes directly from LLM without sanitization.
    # For more complex scenarios, parse the command and arguments properly.
    # Here, we assume the LLM will generate relatively safe commands based on its training.
    # Consider a whitelist of allowed commands or more robust parsing if needed.
    process = run_command(command_string, cwd=str(full_target_dir), shell=True)
    
    return json.dumps({
        "stdout": (process.stdout or "").strip()[:2000], # Limit output length
        "stderr": (process.stderr or "").strip()[:2000], # Limit output length
        "exit_code": process.returncode
    })

# Tool: Read a file's content
def tool_read_file(file_path: str):
    """
    Reads the content of a specified file relative to the code root (/app/code).
    Returns the file content or an error message.
    Example: file_path="src/utils.py"
    """
    logging.info(f"Tool: Attempting to read file: '{file_path}'")
    full_file_path = (CODE_DIR / file_path).resolve()

    if not str(full_file_path).startswith(str(CODE_DIR.resolve())):
        logging.error(f"Tool Error: file_path '{file_path}' is outside the allowed code directory.")
        return json.dumps({"error": "Security restriction: file_path outside allowed scope.", "content": None})

    if not full_file_path.is_file():
        logging.error(f"Tool Error: file '{file_path}' does not exist or is not a file.")
        return json.dumps({"error": "File not found or is not a regular file.", "content": None})
    
    try:
        with open(full_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Limit content length returned to LLM to avoid excessive token usage
        return json.dumps({"content": content[:5000], "note": "Content might be truncated if very long."})
    except Exception as e:
        logging.error(f"Tool Error: Could not read file '{file_path}': {e}")
        return json.dumps({"error": str(e), "content": None})


# Mapping of tool names to functions
AVAILABLE_TOOLS = {
    "run_shell_command": tool_run_shell_command,
    "read_file_content": tool_read_file,
}

# OpenAI function schema for the tools
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Executes a shell command in the specified directory (relative to /app/code) and returns its stdout, stderr, and exit code. Use for linters, tests, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command_string": {
                        "type": "string",
                        "description": "The shell command to execute. E.g., 'pylint main.py' or 'npm test'.",
                    },
                    "target_directory": {
                        "type": "string",
                        "description": "The directory (relative to /app/code) in which to run the command. Defaults to '.' (code root). E.g., 'src' or 'tests'.",
                        "default": "."
                    }
                },
                "required": ["command_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_content",
            "description": "Reads the content of a specified file (relative to /app/code).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file from the root of the codebase (/app/code). E.g., 'src/utils.py'.",
                    }
                },
                "required": ["file_path"],
            },
        },
    }
]
# --- End Tool Definitions ---


def get_llm_response_with_tool_calls(messages_history):
    logging.info(f"Sending request to OpenAI API with {len(messages_history)} messages (tools enabled)...")
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_history,
            tools=TOOLS_SCHEMA,
            tool_choice="auto" # Let the model decide if it wants to use a tool
        )
        return response.choices[0] # Return the full choice object
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None

# --- Main Execution (Modified for Tool Use) ---
def main():
    logging.info("Agent script started (Tools & RAG Enabled).")
    if len(sys.argv) < 4:
        logging.error("Usage: python agent.py <task_description> <target_file_relative_path> <is_github_repo_str>")
        sys.exit(1)

    task_description = sys.argv[1]
    target_file_rel_path = sys.argv[2]
    is_github_repo = sys.argv[3].lower() == 'true'
    
    logging.info(f"Task Description: {task_description}")
    logging.info(f"Target File: {target_file_rel_path}")

    target_file_abs_path = CODE_DIR / target_file_rel_path
    if not target_file_abs_path.is_file():
        logging.error(f"Target file not found: {target_file_abs_path}")
        sys.exit(1)

    # Run setup.sh (as before)
    # ... (setup.sh logic from previous version) ...
    setup_script_path = CODE_DIR / "setup.sh"
    if setup_script_path.exists() and setup_script_path.is_file():
        logging.info(f"Found setup.sh at {setup_script_path}, executing...")
        run_command(["chmod", "+x", str(setup_script_path)], cwd=CODE_DIR)
        setup_log_path = OUTPUT_DIR / "setup.log"
        with open(setup_log_path, "wb") as slf:
            setup_proc = subprocess.Popen([str(setup_script_path)], cwd=CODE_DIR, stdout=slf, stderr=subprocess.STDOUT)
            setup_proc.communicate()
        if setup_proc.returncode == 0: logging.info("setup.sh executed successfully.")
        else: logging.warning(f"setup.sh exited with code {setup_proc.returncode}. Check setup.log.")
    else:
        logging.info("No setup.sh found or it's not a file.")


    with open(target_file_abs_path, "r", encoding='utf-8') as f:
        original_code = f.read()

    all_code_docs_for_rag = load_codebase_for_rag(CODE_DIR, target_file_rel_path)
    relevant_snippets_context = get_relevant_code_snippets(task_description, original_code, all_code_docs_for_rag)
    
    agents_md_path = CODE_DIR / "AGENTS.md"
    agent_custom_instructions = ""
    # ... (AGENTS.MD loading logic from previous version) ...
    if agents_md_path.exists() and agents_md_path.is_file():
        logging.info(f"Found AGENTS.MD at {agents_md_path}, loading custom instructions.")
        with open(agents_md_path, "r", encoding='utf-8') as f:
            agent_custom_instructions = f.read()
    else:
        logging.info("No AGENTS.MD found in repository root, using default prompt structure.")
        # Path(__file__).parent.parent for AGENTS.md.example if it's one level up from agent.py's dir
        example_agents_md_path = Path(__file__).resolve().parent.parent / "AGENTS.md.example"
        if example_agents_md_path.exists():
            with open(example_agents_md_path, "r", encoding='utf-8') as f:
                agent_custom_instructions = f.read()


    system_prompt = f"""{agent_custom_instructions}

You are an AI coding assistant. Your goal is to help the user with their coding task.
You have access to tools like 'run_shell_command' to execute linters, tests, etc., and 'read_file_content' to read other files.
If you need to use a tool, call it. Then, based on the tool's output, decide on the next step.
If the task requires modifying code in '{target_file_rel_path}', your final response should be ONLY the complete, modified code for that file.
Do not include explanations or conversational text outside the code block in your final code modification response.
If the task is to answer a question or provide information, do so clearly.

User's Task Description:
{task_description}

Target File Path for potential modification: {target_file_rel_path}

--- Potentially Relevant Code Snippets from the Codebase (for context) ---
{relevant_snippets_context if relevant_snippets_context else "No highly relevant snippets found by automated search, focus on the target file or use tools to gather more info."}
--- End of Relevant Snippets ---

Current Code from '{target_file_rel_path}':
```
{original_code}
```
"""
    
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please address the following task: {task_description}. The primary file I'm interested in is '{target_file_rel_path}'."}]

    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Cannot proceed.")
        sys.exit(1)

    modified_code_content = None
    tool_call_iterations = 0

    while tool_call_iterations < MAX_TOOL_CALL_ITERATIONS:
        logging.info(f"LLM Iteration {tool_call_iterations + 1}")
        (OUTPUT_DIR / f"prompt_iter_{tool_call_iterations}.json").write_text(json.dumps(messages, indent=2), encoding='utf-8')
        
        choice = get_llm_response_with_tool_calls(messages)
        
        if not choice:
            logging.error("Failed to get a response from LLM.")
            (OUTPUT_DIR / "error.txt").write_text("Failed to get LLM response.", encoding='utf-8')
            sys.exit(1)

        response_message = choice.message
        messages.append(response_message) # Add AI's response to history

        # Check if the model wants to call tools
        if response_message.tool_calls:
            logging.info(f"LLM requested {len(response_message.tool_calls)} tool_calls.")
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logging.info(f"Tool call: {function_name} with args {function_args}")
                
                if function_name in AVAILABLE_TOOLS:
                    tool_function = AVAILABLE_TOOLS[function_name]
                    try:
                        function_response = tool_function(**function_args)
                    except Exception as e:
                        logging.error(f"Error executing tool {function_name}: {e}")
                        function_response = json.dumps({"error": f"Failed to execute tool {function_name}: {str(e)}"})
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })
                    logging.info(f"Tool {function_name} response (first 200 chars): {function_response[:200]}")
                else:
                    logging.error(f"Unknown tool requested: {function_name}")
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": f"Tool '{function_name}' not found."}),
                    })
            tool_call_iterations += 1
            # Continue loop to let LLM process tool results
        else:
            # No tool calls, LLM should have provided the final answer or modified code
            final_content = response_message.content
            if final_content:
                 # Heuristic: If the LLM's response looks like code (e.g., contains common code keywords, newlines)
                 # and the task might involve code modification, assume it's the modified code.
                 # A more robust way is to explicitly ask the LLM to delimit the code block.
                if "def " in final_content or "function " in final_content or "class " in final_content or "public static void main" in final_content or "=> {" in final_content or "```" in final_content:
                    logging.info("LLM provided content that looks like code. Assuming it's the modified code for the target file.")
                    # Strip markdown if present
                    if final_content.strip().startswith("```") and final_content.strip().endswith("```"):
                        modified_code_content = "\n".join(final_content.strip().splitlines()[1:-1])
                    else:
                        modified_code_content = final_content
                else:
                    # If it doesn't look like code, it might be an explanation or answer to a question
                    logging.info("LLM provided a textual response (not primarily code):")
                    logging.info(final_content)
                    # Save this as a general response if no code modification is expected
                    (OUTPUT_DIR / "llm_answer.txt").write_text(final_content, encoding='utf-8')
                    # If the task was *only* to modify code, and this isn't code, it might be an issue.
                    # For now, we'll proceed, and if modified_code_content is None, no file changes occur.
            else:
                logging.warning("LLM response content is empty after tool calls (or no tool calls).")
            break # Exit loop

    if tool_call_iterations >= MAX_TOOL_CALL_ITERATIONS:
        logging.warning(f"Reached max tool call iterations ({MAX_TOOL_CALL_ITERATIONS}). Proceeding with current state.")
        # If the last message from LLM was a tool call request, we don't have a final answer.
        # We might need to grab the last non-tool-requesting message or handle this state.
        # For now, if modified_code_content is still None, it means no code was produced.

    (OUTPUT_DIR / "final_messages_history.json").write_text(json.dumps(messages, indent=2), encoding='utf-8')

    if modified_code_content:
        logging.info(f"Final modified code for {target_file_rel_path} obtained.")
        (OUTPUT_DIR / "llm_response.txt").write_text(modified_code_content, encoding='utf-8')

        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            modified_code_content.splitlines(keepends=True),
            fromfile=f"a/{target_file_rel_path}",
            tofile=f"b/{target_file_rel_path}",
            lineterm=""
        )
        diff_output = "".join(diff)
        (OUTPUT_DIR / "diff.patch").write_text(diff_output, encoding='utf-8')
        logging.info(f"Diff patch saved to {OUTPUT_DIR / 'diff.patch'}")

        with open(target_file_abs_path, "w", encoding='utf-8') as f:
            f.write(modified_code_content)
        logging.info(f"Original file {target_file_abs_path} updated with modified code.")

        if is_github_repo and GITHUB_TOKEN:
            create_github_pr(CODE_DIR, target_file_rel_path, task_description)
        else:
            logging.info("Skipping PR creation (not a GitHub repo or no GITHUB_TOKEN).")
    else:
        logging.warning(f"No definitive modified code was generated for {target_file_rel_path}. Check llm_answer.txt or logs.")
        if not (OUTPUT_DIR / "llm_answer.txt").exists(): # If no textual answer was saved either
             (OUTPUT_DIR / "error.txt").write_text("No modified code or textual answer produced by LLM.", encoding='utf-8')


    logging.info("Agent script finished.")

if __name__ == "__main__":
    main()
```

### 解释与关键点：

1.  **`TOOLS_SCHEMA` 和 `AVAILABLE_TOOLS`:**
    *   `TOOLS_SCHEMA` 是一个列表，遵循 OpenAI 函数调用的格式，描述了每个工具的名称、用途和参数。这会发送给 LLM。
    *   `AVAILABLE_TOOLS` 是一个字典，将工具名称映射到实际的 Python 执行函数 (e.g., `tool_run_shell_command`, `tool_read_file_content`)。

2.  **工具函数 (`tool_run_shell_command`, `tool_read_file_content`):**
    *   这些函数接收 LLM 提供的参数，执行相应的操作。
    *   **重要：** 它们都返回 JSON 字符串作为结果，这是 OpenAI 函数调用期望的格式。
    *   **安全性：**
        *   `tool_run_shell_command`: 使用 `shell=True` 时要特别小心，因为它直接执行字符串命令。这里增加了一个基础的路径检查，确保 `target_directory` 在 `CODE_DIR` 内。对于生产系统，您可能需要一个命令白名单或更复杂的解析和参数化来避免命令注入。
        *   `tool_read_file`: 同样检查路径是否在 `CODE_DIR` 内。
        *   输出限制：工具的输出（stdout, stderr, file content）被截断，以防止向 LLM 发送过多数据，消耗过多 Token。

3.  **主循环 (`while tool_call_iterations < MAX_TOOL_CALL_ITERATIONS`):**
    *   Agent 现在在一个循环中与 LLM 交互。
    *   **LLM 调用：** `get_llm_response_with_tool_calls` 发送当前的消息历史和 `TOOLS_SCHEMA` 给 LLM。
    *   **检查工具调用：** 如果 `response_message.tool_calls` 存在，说明 LLM 想使用一个或多个工具。
    *   **执行工具：** Agent 遍历 `tool_calls`，查找对应的本地函数，执行它，并将结果（JSON 字符串）以 `role: "tool"` 的形式追加到 `messages` 历史中。
    *   **反馈给 LLM：** 循环继续，LLM 接收到工具的输出，并可以基于这些信息决定下一步（调用另一个工具，或最终回答/生成代码）。
    *   **无工具调用/最终输出：** 如果 LLM 不再调用工具，`response_message.content` 应该包含最终的答案或修改后的代码。
        *   **代码识别启发式：** 我添加了一个简单的启发式方法来判断 LLM 的输出是否为代码。在实际应用中，您可能需要更明确地指示 LLM 如何格式化其最终代码输出（例如，使用特定的标记或总是将其包装在 markdown 代码块中）。

4.  **Prompt 更新：**
    *   系统 Prompt 现在明确告知 LLM 它拥有工具，并鼓励它在需要时使用。

5.  **`MAX_TOOL_CALL_ITERATIONS`:** 防止 Agent 和 LLM 陷入无限的工具调用循环。

### 如何测试：

1.  **确保 `OPENAI_API_KEY` 有效且模型支持函数调用** (e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo-0125` 或更新版本)。
2.  **构建新的 Agent Docker 镜像：**
    ```bash
    cd dockerfiles/agent
    docker build -t codex-agent:latest .
    cd ../..
    ```
3.  **启动后端。**
4.  **通过前端提交任务，例如：**
    *   **任务描述:** "Run pylint on the file 'main.py' in the root directory and tell me the issues. Then, refactor the 'calculate_sum' function in 'main.py' to be more efficient if pylint reports any performance concerns related to it."
    *   **目标文件:** `main.py` (假设你的仓库中有一个 `main.py` 文件，并且 `pylint` 安装在 Agent 容器中或通过 `setup.sh` 安装)。
    *   你也可以尝试："Read the first 10 lines of 'utils/helper.py' and then summarize its purpose." (假设有 `utils/helper.py`)

### 预期行为：

*   Agent 会首先调用 LLM。
*   LLM 可能会决定调用 `run_shell_command` (例如，`pylint main.py`)。
*   Agent 执行 `pylint`，并将输出返回给 LLM。
*   LLM 接收 `pylint` 的结果，然后可能会再次调用工具（如果需要更多信息）或直接生成修改后的 `main.py` 代码（如果任务是修改代码）或提供文本答案。
*   你可以在 `storage/logs/<task_id>/` 目录下查看 `prompt_iter_*.json` 和 `final_messages_history.json` 来跟踪对话和工具调用过程。

