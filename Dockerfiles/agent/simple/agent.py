import ast
import difflib
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, TypedDict, Optional

import openai
from dotenv import load_dotenv
from langgraph.graph import Graph

# todo upsonic mem0
# --- Configuration ---
# Load .env file from /app (if it exists, for local testing) or rely on Docker env vars
dotenv_path = Path('/app/.env')
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Path configuration
CODE_DIR = Path("/app/code").resolve()
OUTPUT_DIR = Path("/app/output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINE_TUNE_LOG_DIR = OUTPUT_DIR / "finetuning_data"
FINE_TUNE_LOG_DIR.mkdir(exist_ok=True)

# API and execution configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MAX_MCP_ITERATIONS = int(os.getenv("MAX_MCP_ITERATIONS", "7"))
CMD_TIMEOUT = int(os.getenv("CMD_TIMEOUT", "20"))
CMD_MAX_OUTPUT = int(os.getenv("CMD_MAX_OUTPUT", "20000"))

# --- Logging setup ---
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
    """Execute a subprocess command with logging and error handling.
    
    Args:
        command: List of command arguments or command string
        cwd: Working directory for the command
        env: Environment variables dictionary
        
    Returns:
        CompletedProcess object or None if exception occurred
    """
    cmd_str = ' '.join(command) if isinstance(command, list) else command
    logging.info(f"Running command: {cmd_str} in {cwd or '.'}")
    
    try:
        # Handle command as list or string
        cmd_args = command if isinstance(command, list) else shlex.split(command)
        
        # Check for dangerous commands if it's a string
        if not isinstance(command, list):
            dangerous = [";", "&&", "|", ">", "<", "`", "$(", "&", "sudo", "yum", "apt", "pip", "curl", "wget"]
            if any(tok in command for tok in dangerous):
                logging.error("Command blocked by security policy")
                return None
        
        process = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit, we'll check returncode
            cwd=cwd,
            env=env,
            timeout=CMD_TIMEOUT
        )
        
        # Limit output size
        stdout = process.stdout[:CMD_MAX_OUTPUT] if process.stdout else ""
        stderr = process.stderr[:CMD_MAX_OUTPUT] if process.stderr else ""
        
        if stdout:
            logging.info(f"Stdout: {stdout.strip()}")
        if stderr:
            logging.error(f"Stderr: {stderr.strip()}")
        if process.returncode != 0:
            logging.error(f"Command failed with exit code {process.returncode}")
            
        return process
    except subprocess.TimeoutExpired:
        logging.error(f"Command timed out after {CMD_TIMEOUT}s: {cmd_str}")
        return None
    except Exception as e:
        logging.error(f"Exception running command {cmd_str}: {e}")
        return None


def get_llm_response(prompt_text, max_retries=3):
    """Send a prompt to the OpenAI API and get the response with retry logic.
    
    Args:
        prompt_text: The text prompt to send to the API
        max_retries: Maximum number of retry attempts
        
    Returns:
        Response content or None if error occurred after all retries
    """
    logging.info("Sending request to OpenAI API...")
    
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_text}]
            )
            content = response.choices[0].message.content
            
            if content.startswith("```") and content.endswith("```"):
                content = "\n".join(content.splitlines()[1:-1])
                
            logging.info("Received response from OpenAI API.")
            return content
        except Exception as e:
            logging.error(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Exponential backoff
                backoff = 2 ** attempt
                logging.info(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                logging.error("All retry attempts failed.")
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

# --- Workflow Functions ---

# Define state type
class AgentState(TypedDict):
    task_description: str
    target_file_rel_path: str 
    is_github_repo: bool
    original_code: str
    modified_code: str
    diff_output: str
    pr_created: bool
    modification_failed: bool

# --- Node Functions ---
async def load_code_node(state: AgentState) -> AgentState:
    """Load code from target file"""
    target_file_abs_path = CODE_DIR / state["target_file_rel_path"]
    with open(target_file_abs_path, "r", encoding='utf-8') as f:
        state["original_code"] = f.read()
    return state

async def generate_code_node(state: AgentState) -> AgentState:
    """Generate modified code using LLM with syntax validation"""
    # Determine language from file extension
    language = get_language_from_file(state["target_file_rel_path"])
    language_name = language.capitalize()
    
    # Create a language-specific prompt
    prompt = f"""Modify the following {language_name} code according to the task requirements: {state['task_description']}

Original code:
```{language}
{state['original_code']}
```

Only return the complete modified code, no explanation needed. Ensure the code has valid {language_name} syntax."""
    
    # Get modified code from LLM
    modified_code = get_llm_response(prompt)
    
    # Use MCP to validate syntax
    if not validate_syntax(modified_code, state["target_file_rel_path"]):
        # First retry - explicitly ask for fixed syntax
        retry_prompt = f"""The previous modification has syntax errors. Please fix and provide a valid {language_name} implementation.

Task: {state['task_description']}

Original code:
```{language}
{state['original_code']}
```

Your previous attempt (with syntax errors):
```{language}
{modified_code}
```

Provide corrected code with valid {language_name} syntax:"""
        
        log.warning(f"First attempt had {language_name} syntax errors. Retrying with explicit syntax correction prompt.")
        modified_code = get_llm_response(retry_prompt)
        
        # Check syntax again
        if not validate_syntax(modified_code, state["target_file_rel_path"]):
            # Try MCP-based correction
            task_id = str(uuid.uuid4())
            mcp_payload = {
                "event": "syntax_fix",
                "language": language,
                "code": modified_code,
                "task": state["task_description"]
            }
            
            log.info(f"Requesting MCP syntax correction for {language_name} code")
            mcp_response = mcp_step(task_id, mcp_payload)
            
            if mcp_response.get("event") == "fixed_code" and mcp_response.get("code"):
                modified_code = mcp_response["code"]
                if validate_syntax(modified_code, state["target_file_rel_path"]):
                    log.info(f"MCP successfully fixed {language_name} syntax")
                    state["modified_code"] = modified_code
                    state["modification_failed"] = False
                    return state
            
            # Final fallback - keep original code
            log.error(f"All attempts to fix {language_name} syntax failed. Keeping original code.")
            state["modification_failed"] = True
            state["modified_code"] = state["original_code"]
            return state
    
    # Success path
    state["modified_code"] = modified_code
    state["modification_failed"] = False
    return state

async def create_diff_node(state: AgentState) -> AgentState:
    """Create diff between original and modified code"""
    # Check if modification failed
    if state.get("modification_failed", False):
        log.warning("Skipping diff generation due to failed modification")
        state["diff_output"] = ""  # Empty diff since we're using original code
        return state
        
    # Create diff only if the code was actually modified
    diff = difflib.unified_diff(
        state["original_code"].splitlines(keepends=True),
        state["modified_code"].splitlines(keepends=True),
        fromfile=f"a/{state['target_file_rel_path']}",
        tofile=f"b/{state['target_file_rel_path']}"
    )
    diff_output = "".join(diff)
    
    # Check if we actually have changes
    if not diff_output.strip():
        log.warning("No actual changes made to the code")
        state["modification_failed"] = True
    else:
        state["modification_failed"] = False
        
    state["diff_output"] = diff_output
    return state

async def create_pr_node(state: AgentState) -> AgentState:
    """Create PR if in a GitHub repo and modification was successful"""
    # Skip PR creation if modification failed or if there's no diff
    if state.get("modification_failed", False) or not state["diff_output"].strip():
        log.warning("Skipping PR creation due to failed modification or no changes")
        state["pr_created"] = False
        return state
        
    # Create PR only if in a GitHub repo with token
    if state["is_github_repo"] and GITHUB_TOKEN:
        try:
            # Write the modified code to the file
            target_file_abs_path = CODE_DIR / state["target_file_rel_path"]
            with open(target_file_abs_path, "w", encoding='utf-8') as f:
                f.write(state["modified_code"])
                
            # Create the PR
            state["pr_created"] = create_github_pr(CODE_DIR, state["target_file_rel_path"], state["task_description"])
        except Exception as e:
            log.error(f"Error creating PR: {e}")
            state["pr_created"] = False
    else:
        state["pr_created"] = False
        
    return state

# --- Build Workflow ---
workflow = Graph()

# Add nodes
workflow.add_node("load_code", load_code_node)
workflow.add_node("generate_code", generate_code_node) 
workflow.add_node("create_diff", create_diff_node)
workflow.add_node("create_pr", create_pr_node)

# Define edges
workflow.add_edge("load_code", "generate_code")
workflow.add_edge("generate_code", "create_diff")
workflow.add_edge("create_diff", "create_pr")

# Set entry and exit points
workflow.set_entry_point("load_code")
workflow.set_finish_point("create_pr")

# Main function using workflow
async def main():
    """Main entry point for the workflow-based agent"""
    # Check if we have enough command line arguments
    if len(sys.argv) < 4:
        print("Usage: python agent.py <task_description> <target_file_rel_path> <is_github_repo>")
        sys.exit(1)
        
    # Initialize state from command line arguments
    initial_state = {
        "task_description": sys.argv[1],
        "target_file_rel_path": sys.argv[2],
        "is_github_repo": sys.argv[3].lower() == 'true',
        "original_code": "",
        "modified_code": "",
        "diff_output": "",
        "pr_created": False,
        "modification_failed": False
    }
    
    app = workflow.compile()
    try:
        result = await app.ainvoke(initial_state)
    except Exception as e:
        log.error(f"Error running workflow: {e}")
        return None
    
    return result
# --- Run the script if executed directly ---
if __name__ == "__main__":
    # Check if we're running in MCP mode
    if len(sys.argv) > 1 and sys.argv[1] == "--mcp-mode":
        # MCP service mode - listen for MCP requests
        task_id = str(uuid.uuid4()) if len(sys.argv) <= 2 else sys.argv[2]
        log.info(f"Starting in MCP mode with task ID: {task_id}")
        
        # Simple MCP loop
        payload = {"event": "start", "agent": "codex-agent"}
        try:
            mcp_response = mcp_step(task_id, payload)
            log.info(f"MCP response: {mcp_response}")
        except Exception as e:
            log.error(f"Error in MCP mode: {e}")
    else:
        # Standard workflow mode
        import asyncio
        asyncio.run(main())