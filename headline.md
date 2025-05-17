Okay, this is an excellent and well-defined project! I'll generate a complete project template for you, covering the core functionalities you've outlined.

This template will include:

1.  **Backend (Go):**
    *   API endpoints for task creation (`/task`) and status/log retrieval.
    *   Logic for Git cloning and ZIP extraction.
    *   Docker container spawning and management.
    *   Task state management (in-memory for simplicity, can be extended to a DB).
    *   Serving log files.
2.  **Agent Container (Python):**
    *   `Dockerfile` to build the agent image.
    *   `agent.py` script that:
        *   Takes task instructions and code.
        *   Calls OpenAI LLM.
        *   Generates a diff.
        *   Attempts to create a GitHub PR (if GitHub repo and token are provided).
    *   `requirements.txt` for Python dependencies.
    *   Example `AGENTS.md` and `setup.sh`.
3.  **Basic Frontend (HTML/JS):**
    *   A simple interface to submit tasks and view results/logs.
4.  **README and Configuration:**
    *   Instructions on how to set up, build, and run the system.
    *   `.env.example` for configuration.

Let's get started!

---

## 🚀 Generated Project Template: Codex-like SYS 🚀

Here's the directory structure and file contents:

```
codex-sys/
├── backend/
│   ├── main.go
│   ├── api/
│   │   └── handlers.go
│   ├── docker_utils/
│   │   └── spawner.go
│   ├── tasks/
│   │   └── manager.go
│   ├── utils/
│   │   └── utils.go
│   ├── go.mod
│   └── go.sum
├── dockerfiles/
│   └── agent/
│       ├── Dockerfile
│       ├── agent.py
│       ├── requirements.txt
│       ├── AGENTS.md.example  # Example instructions for the agent
│       └── setup.sh.example   # Example setup script
├── frontend/
│   ├── index.html
│   └── script.js
├── storage/                  # Will be created by the backend
│   ├── repos/                # For cloned/uploaded code
│   └── logs/                 # For task logs and outputs
├── .env.example
└── README.md
```

---

### 1. `README.md`

```markdown
# Codex-like SYS: AI Code Workstation Platform

This project implements a system where users can submit code (via Git URL or ZIP upload) to be processed by an AI agent running in an isolated Docker container. The agent can analyze, modify code, generate diffs, and even attempt to create GitHub Pull Requests.

## Features

*   **Code Input:** Supports Git URLs and ZIP file uploads.
*   **Isolated Execution:** Each task runs in its own Docker container.
*   **LLM Integration:** Agent uses OpenAI (configurable for others) for code tasks.
*   **Output:** Generates diffs, logs, and can create GitHub PRs.
*   **Web API:** Backend API for task management.
*   **Simple Frontend:** Basic UI for interaction.

## Project Structure

```
codex-sys/
├── backend/                  # Go backend service
├── dockerfiles/agent/        # Docker setup for the Python agent
├── frontend/                 # Simple HTML/JS frontend
├── storage/                  # Runtime data (code, logs) - gitignored
├── .env.example              # Environment variable template
└── README.md                 # This file
```

## Prerequisites

*   **Go:** Version 1.20+
*   **Docker:** Docker daemon running
*   **Python 3:** For the agent (will be containerized)
*   **Git:** For cloning repositories
*   **OpenAI API Key:** For the LLM agent
*   **GitHub Personal Access Token (Optional):** If you want the agent to create PRs. Needs `repo` scope.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> codex-sys
    cd codex-sys
    ```

2.  **Configure Environment Variables:**
    Copy `.env.example` to `.env` and fill in your details:
    ```bash
    cp .env.example .env
    nano .env
    ```
    Update `OPENAI_API_KEY` and optionally `GITHUB_TOKEN`.

3.  **Build the Agent Docker Image:**
    ```bash
    cd dockerfiles/agent
    docker build -t codex-agent:latest .
    cd ../..
    ```
    *Note: If you modify `agent.py` or its dependencies, you'll need to rebuild this image.*

4.  **Prepare Backend Dependencies:**
    ```bash
    cd backend
    go mod tidy
    cd ..
    ```

## Running the System

1.  **Start the Backend Service:**
    ```bash
    cd backend
    go run main.go
    ```
    The backend will start (default: `http://localhost:8080`). It will also create `storage/repos` and `storage/logs` directories if they don't exist.

2.  **Access the Frontend:**
    Open `frontend/index.html` in your web browser.
    *Note: For simplicity, this frontend directly accesses the local backend. For a deployed version, you'd serve the frontend via the backend or a web server and handle CORS.*

## How it Works

1.  **User Interaction (Frontend/API):**
    *   User provides a Git URL or uploads a ZIP file, along with a task description (e.g., "Refactor this function for clarity") and a target file within the repo.
    *   The frontend sends this to the backend API (`/api/task`).

2.  **Backend Processing:**
    *   The Go backend receives the request.
    *   It generates a unique Task ID.
    *   If Git URL: Clones the repo into `storage/repos/<task_id>`.
    *   If ZIP: Extracts the ZIP into `storage/repos/<task_id>`.
    *   It creates a log directory `storage/logs/<task_id>`.
    *   It spawns a `codex-agent` Docker container.
        *   The user's code directory (`storage/repos/<task_id>`) is mounted to `/app/code` in the container.
        *   The log directory (`storage/logs/<task_id>`) is mounted to `/app/output` in the container.
        *   `OPENAI_API_KEY` and `GITHUB_TOKEN` are passed as environment variables.
        *   The agent is invoked with the task description and target file.

3.  **Agent Execution (Inside Container):**
    *   The `agent.py` script runs.
    *   It reads `AGENTS.md.example` (if present in the repo root) for custom instructions.
    *   It reads the target code file.
    *   It constructs a prompt for the LLM.
    *   It calls the OpenAI API.
    *   It processes the LLM's response (e.g., modified code).
    *   It generates a `diff.patch` file.
    *   If a `GITHUB_TOKEN` is provided and the input was a GitHub repo, it attempts to:
        *   Create a new branch.
        *   Commit the changes.
        *   Push the branch.
        *   Create a Pull Request.
    *   It writes `prompt.txt`, `llm_response.txt`, `diff.patch`, and `agent.log` to `/app/output`.
    *   It executes `setup.sh.example` if present in the repo root.

4.  **Results & Logging:**
    *   The backend monitors the container. Once it finishes, the task status is updated.
    *   Output files from `/app/output` (now in `storage/logs/<task_id>`) are available via the API (`/api/logs/<task_id>/<filename>`).
    *   The frontend can poll for task status and display links to the logs/diff.

## API Endpoints

*   `POST /api/task`: Create a new task.
    *   Body (form-data):
        *   `git_url` (string, optional): URL of the Git repository.
        *   `zip_file` (file, optional): Uploaded ZIP file of the code.
        *   `task_description` (string): What the agent should do.
        *   `target_file` (string): Relative path to the target file in the repo (e.g., `src/main.py`).
        *   `github_token` (string, optional, for PRs if not set in backend .env): User-provided GitHub token.
*   `GET /api/task/{task_id}/status`: Get the status of a task.
*   `GET /api/logs/{task_id}/{filename}`: Get a specific log file (e.g., `diff.patch`, `agent.log`).

## Security Considerations (Reiteration)

*   **Container Sandboxing:** Each task runs in an ephemeral Docker container.
*   **Resource Limits:** Consider adding CPU/memory limits to container creation.
*   **Network Isolation:** The current agent `Dockerfile` allows internet access for `pip install` and LLM calls. For stricter security, you could have a multi-stage Docker build or disable network during agent execution (if using local LLMs).
*   **Input Sanitization:** Ensure proper handling of user inputs (URLs, file names).
*   **Secrets Management:** `OPENAI_API_KEY` and `GITHUB_TOKEN` are sensitive. Use a proper secrets management solution for production.
*   **Code Execution:** The `setup.sh` script from the user's repo is executed. This is a potential security risk. Implement whitelisting or sandboxing for `setup.sh` commands if untrusted code is processed.

## Future Enhancements

*   Persistent task storage (e.g., PostgreSQL, SQLite).
*   User authentication and authorization.
*   Support for more LLMs (Claude, local models).
*   More sophisticated agent capabilities.
*   WebSockets for real-time log streaming.
*   Job queue (e.g., Redis + Celery/RabbitMQ) for better task management.
*   More robust error handling and retry mechanisms.
*   Configurable container resource limits.
```

---

### 2. `.env.example`

```ini
# Backend Configuration
BACKEND_PORT=8080
STORAGE_PATH=./storage # Relative to backend executable or absolute

# Agent Configuration
OPENAI_API_KEY="sk-your-openai-api-key"

# Optional: GitHub Token for creating Pull Requests by the agent
# The agent will use this token to authenticate with GitHub
# Ensure it has 'repo' scope for private repos, or public_repo for public ones.
GITHUB_TOKEN=""
# If a user provides a GITHUB_TOKEN via the API, that will override this one for that specific task.
```

---

### 3. `dockerfiles/agent/Dockerfile`

```Dockerfile
FROM python:3.10-slim

# Install git and curl (git for cloning/PRs, curl for potential setup.sh needs)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    # GitHub CLI for PR creation
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .
# AGENTS.md.example and setup.sh.example are not copied here
# They are expected to be in the user's code mounted at /app/code

# Ensure output directory exists for the agent
RUN mkdir -p /app/output

# Default command, can be overridden by Docker run command
# CMD ["python3", "agent.py"]
ENTRYPOINT ["python3", "agent.py"]
```

---

### 4. `dockerfiles/agent/requirements.txt`

```txt
openai
python-dotenv
requests
```

---

### 5. `dockerfiles/agent/AGENTS.md.example`

```markdown
## Agent Instructions

You are an AI coding assistant. Your goal is to help the user with their coding task.

**Task Context:**
The user wants to modify a specific file in their codebase. You will be provided with the content of this file.

**Your Role:**
1.  Understand the user's request (provided as "task_description").
2.  Analyze the provided code.
3.  Generate the modified code based on the request.
4.  **IMPORTANT:** Output ONLY the complete, modified code for the specified file. Do not include any explanations, apologies, or conversational text before or after the code block. Just the raw code.

**Example User Request:** "Add a try-except block around the file reading operation to catch FileNotFoundError."

**Example Code Input:**
```python
def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content
```

**Example Expected Output (Your response to the LLM query):**
```python
def read_file(path):
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
```

**Current Task:**
The user will provide the specific task description when you are invoked.
```

---

### 6. `dockerfiles/agent/setup.sh.example`

```bash
#!/bin/bash
echo "--- Running setup.sh from user repository ---"
date
echo "Current directory: $(pwd)"
echo "Listing files in /app/code:"
ls -la /app/code
echo "--- Finished setup.sh ---"

# Example: Install a specific dependency if needed for the code analysis
# (though it's better to have common ones in the Dockerfile)
# if [ -f "/app/code/requirements-dev.txt" ]; then
#   pip install -r /app/code/requirements-dev.txt
# fi
```

---

### 7. `dockerfiles/agent/agent.py`

```python
import openai
import os
import sys
import difflib
import logging
import subprocess
import shutil
from pathlib import Path
from dotenv import load_dotenv

# --- Configuration ---
# Load .env file from /app (if it exists, for local testing) or rely on Docker env vars
dotenv_path = Path('/app/.env')
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # Can be overridden by user via API
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o") # or "gpt-3.5-turbo"

CODE_DIR = Path("/app/code")
OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
log_file_path = OUTPUT_DIR / "agent.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Also print to stdout for Docker logs
    ]
)

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
    run_command(["git", "config", "--global", "user.email", "codex-agent@example.com"], cwd=repo_path)
    run_command(["git", "config", "--global", "user.name", "Codex Agent"], cwd=repo_path)

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

if __name__ == "__main__":
    main()
```

---

### 8. `backend/go.mod`

```mod
module codex-sys/backend

go 1.21 // Or your Go version

require (
        github.com/docker/docker v26.1.3+incompatible
        github.com/gin-contrib/cors v1.7.2
        github.com/gin-gonic/gin v1.10.0
        github.com/google/uuid v1.6.0
        github.com/joho/godotenv v1.5.1
)

// ... other indirect dependencies will be added by `go mod tidy`
// Example indirects (versions might vary):
require (
        github.com/Microsoft/go-winio v0.4.14 // indirect
        github.com/bytedance/sonic v1.11.6 // indirect
        github.com/bytedance/sonic/loader v0.1.1 // indirect
        github.com/cloudwego/base64x v0.1.4 // indirect
        github.com/cloudwego/iasm v0.2.0 // indirect
        github.com/containerd/log v0.1.0 // indirect
        github.com/distribution/reference v0.6.0 // indirect
        github.com/docker/go-connections v0.5.0 // indirect
        github.com/docker/go-units v0.5.0 // indirect
        github.com/felixge/httpsnoop v1.0.4 // indirect
        github.com/gabriel-vasile/mimetype v1.4.3 // indirect
        github.com/gin-contrib/sse v0.1.0 // indirect
        github.com/go-logr/logr v1.4.1 // indirect
        github.com/go-logr/stdr v1.2.2 // indirect
        github.com/go-playground/locales v0.14.1 // indirect
        github.com/go-playground/universal-translator v0.18.1 // indirect
        github.com/go-playground/validator/v10 v10.20.0 // indirect
        github.com/goccy/go-json v0.10.2 // indirect
        github.com/gogo/protobuf v1.3.2 // indirect
        github.com/json-iterator/go v1.1.12 // indirect
        github.com/klauspost/cpuid/v2 v2.2.7 // indirect
        github.com/leodido/go-urn v1.4.0 // indirect
        github.com/mattn/go-isatty v0.0.20 // indirect
        github.com/moby/docker-image-spec v1.3.1 // indirect
        github.com/moby/term v0.5.0 // indirect
        github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
        github.com/modern-go/reflect2 v1.0.2 // indirect
        github.com/morikuni/aec v1.0.0 // indirect
        github.com/opencontainers/go-digest v1.0.0 // indirect
        github.com/opencontainers/image-spec v1.1.0 // indirect
        github.com/pelletier/go-toml/v2 v2.2.2 // indirect
        github.com/pkg/errors v0.9.1 // indirect
        github.com/twitchyliquid64/golang-asm v0.15.1 // indirect
        github.com/ugorji/go/codec v1.2.12 // indirect
        golang.org/x/arch v0.8.0 // indirect
        golang.org/x/crypto v0.23.0 // indirect
        golang.org/x/net v0.25.0 // indirect
        golang.org/x/sys v0.20.0 // indirect
        golang.org/x/text v0.15.0 // indirect
        google.golang.org/genproto/googleapis/rpc v0.0.0-20231002182017-d307bd883b97 // indirect
        google.golang.org/grpc v1.58.3 // indirect
        google.golang.org/protobuf v1.34.1 // indirect
        gopkg.in/yaml.v3 v3.0.1 // indirect
)
```
*(Run `go mod tidy` in the `backend` directory after creating the Go files to populate `go.sum` and ensure dependencies are correct).*

---

### 9. `backend/main.go`

```go
package main

import (
        "codex-sys/backend/api"
        "codex-sys/backend/tasks"
        "codex-sys/backend/utils"
        "fmt"
        "log"
        "os"
        "path/filepath"

        "github.com/gin-contrib/cors"
        "github.com/gin-gonic/gin"
        "github.com/joho/godotenv"
)

func main() {
        err := godotenv.Load("../.env") // Load .env from project root
        if err != nil {
                log.Println("No .env file found or error loading, relying on environment variables")
        }

        port := os.Getenv("BACKEND_PORT")
        if port == "" {
                port = "8080"
        }

        storagePath := os.Getenv("STORAGE_PATH")
        if storagePath == "" {
                storagePath = "../storage" // Relative to backend executable
        }
        // Ensure storagePath is absolute
        absStoragePath, err := filepath.Abs(storagePath)
        if err != nil {
                log.Fatalf("Error getting absolute path for storage: %v", err)
        }
        utils.GlobalStoragePath = absStoragePath // Set global storage path

        // Create necessary storage directories
        if err := os.MkdirAll(filepath.Join(utils.GlobalStoragePath, "repos"), 0755); err != nil {
                log.Fatalf("Failed to create repos directory: %v", err)
        }
        if err := os.MkdirAll(filepath.Join(utils.GlobalStoragePath, "logs"), 0755); err != nil {
                log.Fatalf("Failed to create logs directory: %v", err)
        }

        // Initialize Task Manager
        tasks.InitManager()

        router := gin.Default()

        // CORS configuration
        config := cors.DefaultConfig()
        config.AllowOrigins = []string{"*"} // Allow all origins for simplicity. For production, restrict this.
        config.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
        config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
        router.Use(cors.New(config))

        // API routes
        apiGroup := router.Group("/api")
        {
                apiGroup.POST("/task", api.HandleCreateTask)
                apiGroup.GET("/task/:task_id/status", api.HandleGetTaskStatus)
                apiGroup.GET("/logs/:task_id/:filename", api.HandleGetLogFile)
        }

        // Serve frontend static files (optional, for simple demo)
    // For production, use a dedicated web server or CDN for frontend
    // Note: This path is relative to where the 'backend' executable runs.
    // If you run 'go run main.go' from 'codex-sys/backend/', then '../frontend' is correct.
        router.Static("/ui", "../frontend")
        router.GET("/", func(c *gin.Context) {
                c.Redirect(302, "/ui/index.html")
        })


        log.Printf("Server starting on port %s", port)
        log.Printf("Storage path: %s", utils.GlobalStoragePath)
        log.Printf("Access UI at http://localhost:%s/ui/index.html or http://localhost:%s/", port, port)

        if err := router.Run(":" + port); err != nil {
                log.Fatalf("Failed to run server: %v", err)
        }
}
```

---

### 10. `backend/utils/utils.go`

```go
package utils

import (
        "archive/zip"
        "fmt"
        "io"
        "log"
        "os"
        "path/filepath"
        "strings"

        "github.com/google/uuid"
)

var GlobalStoragePath string // Set this at startup

func GenerateTaskID() string {
        return uuid.New().String()
}

func GetRepoPath(taskID string) string {
        return filepath.Join(GlobalStoragePath, "repos", taskID)
}

func GetLogPath(taskID string) string {
        return filepath.Join(GlobalStoragePath, "logs", taskID)
}

// Unzip will decompress a zip archive, moving all files and folders
// within the zip file (parameter 1) to an output directory (parameter 2).
func Unzip(src string, dest string) ([]string, error) {
        var filenames []string

        r, err := zip.OpenReader(src)
        if err != nil {
                return filenames, err
        }
        defer r.Close()

        for _, f := range r.File {
                // Store filename/path for returning and using later on
                fpath := filepath.Join(dest, f.Name)

                // Check for ZipSlip. More Info: http://bit.ly/2MsjAWE
                if !strings.HasPrefix(fpath, filepath.Clean(dest)+string(os.PathSeparator)) {
                        return filenames, fmt.Errorf("%s: illegal file path", fpath)
                }

                filenames = append(filenames, f.Name) // Use relative path within zip

                if f.FileInfo().IsDir() {
                        // Make Folder
                        os.MkdirAll(fpath, os.ModePerm)
                        continue
                }

                // Make File
                if err = os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
                        return filenames, err
                }

                outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
                if err != nil {
                        return filenames, err
                }

                rc, err := f.Open()
                if err != nil {
                        return filenames, err
                }

                _, err = io.Copy(outFile, rc)

                // Close the file handles explicitly
                outFile.Close()
                rc.Close()

                if err != nil {
                        return filenames, err
                }
        }
        return filenames, nil
}

// SanitizePath prevents directory traversal.
// It cleans the path and ensures it's relative.
func SanitizePath(path string) (string, error) {
    cleaned := filepath.Clean(path)
    if strings.Contains(cleaned, "..") {
        return "", fmt.Errorf("invalid path: contains '..'")
    }
    // Ensure it's a relative path segment
    if filepath.IsAbs(cleaned) {
        return "", fmt.Errorf("invalid path: must be relative")
    }
    return cleaned, nil
}
```

---

### 11. `backend/tasks/manager.go`

```go
package tasks

import (
        "sync"
        "time"
)

type TaskStatus string

const (
        StatusPending   TaskStatus = "pending"
        StatusCloning   TaskStatus = "cloning"
        StatusPreparing TaskStatus = "preparing"
        StatusRunning   TaskStatus = "running"
        StatusCompleted TaskStatus = "completed"
        StatusFailed    TaskStatus = "failed"
)

type Task struct {
        ID              string
        GitURL          string
        ZipFileName     string // Original name of uploaded zip
        TargetFile      string // Relative path to file in repo
        TaskDescription string
        Status          TaskStatus
        Message         string    // e.g., error message or PR URL
        CreatedAt       time.Time
        UpdatedAt       time.Time
        RepoPath        string // Absolute path to the code on disk
        LogPath         string // Absolute path to the logs on disk
        IsGitHubRepo    bool   // True if cloned from github.com
        GitHubToken     string // User-provided token for this task (optional)
}

var (
        taskStore = make(map[string]*Task)
        mu        sync.RWMutex
)

func InitManager() {
        // Could load tasks from a persistent store here in the future
}

func AddTask(task *Task) {
        mu.Lock()
        defer mu.Unlock()
        task.CreatedAt = time.Now()
        task.UpdatedAt = time.Now()
        taskStore[task.ID] = task
}

func GetTask(taskID string) (*Task, bool) {
        mu.RLock()
        defer mu.RUnlock()
        task, exists := taskStore[taskID]
        return task, exists
}

func UpdateTaskStatus(taskID string, status TaskStatus, message string) {
        mu.Lock()
        defer mu.Unlock()
        if task, exists := taskStore[taskID]; exists {
                task.Status = status
                task.Message = message
                task.UpdatedAt = time.Now()
        }
}
```

---

### 12. `backend/docker_utils/spawner.go`

```go
package docker_utils

import (
        "codex-sys/backend/tasks"
        "codex-sys/backend/utils"
        "context"
        "fmt"
        "io"
        "log"
        "os"
        "path/filepath"
        "strings"
        "time"

        "github.com/docker/docker/api/types"
        "github.com/docker/docker/api/types/container"
        "github.com/docker/docker/api/types/mount"
        "github.com/docker/docker/client"
)

const AgentImageName = "codex-agent:latest" // Must match the image built by Dockerfile

func RunAgentContainer(task *tasks.Task) {
        ctx := context.Background()
        cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
        if err != nil {
                log.Printf("Error creating Docker client for task %s: %v", task.ID, err)
                tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Failed to create Docker client")
                return
        }
        defer cli.Close()

        // Ensure the agent image exists locally
        _, _, err = cli.ImageInspectWithRaw(ctx, AgentImageName)
    if err != nil {
        if client.IsErrNotFound(err) {
            log.Printf("Agent image %s not found locally. Attempting to pull...", AgentImageName)
            reader, pullErr := cli.ImagePull(ctx, AgentImageName, types.ImagePullOptions{})
            if pullErr != nil {
                log.Printf("Error pulling image %s for task %s: %v", AgentImageName, task.ID, pullErr)
                tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, fmt.Sprintf("Failed to pull agent image: %s", AgentImageName))
                return
            }
            defer reader.Close()
            io.Copy(os.Stdout, reader) // Show pull progress
            log.Printf("Image %s pulled successfully.", AgentImageName)
        } else {
            log.Printf("Error inspecting image %s for task %s: %v", AgentImageName, task.ID, err)
            tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Failed to inspect agent image")
            return
        }
    }


        containerName := fmt.Sprintf("codex-agent-%s", task.ID)
        repoPath := utils.GetRepoPath(task.ID)
        logPath := utils.GetLogPath(task.ID)

        // Ensure logPath directory exists for mounting
        if err := os.MkdirAll(logPath, 0755); err != nil {
                log.Printf("Error creating log directory %s for task %s: %v", logPath, task.ID, err)
                tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Failed to create log directory")
                return
        }

        envVars := []string{
                fmt.Sprintf("OPENAI_API_KEY=%s", os.Getenv("OPENAI_API_KEY")),
                fmt.Sprintf("OPENAI_MODEL=%s", os.Getenv("OPENAI_MODEL")), // Pass model if set
        }
        // Use task-specific GitHub token if provided, otherwise use the one from .env
        githubToken := task.GitHubToken
        if githubToken == "" {
                githubToken = os.Getenv("GITHUB_TOKEN")
        }
        if githubToken != "" {
                envVars = append(envVars, fmt.Sprintf("GITHUB_TOKEN=%s", githubToken))
        }


        containerConfig := &container.Config{
                Image: AgentImageName,
                Cmd: []string{
                        task.TaskDescription,
                        task.TargetFile, // Relative path for the agent
                        fmt.Sprintf("%t", task.IsGitHubRepo),
                },
                Env:        envVars,
                Tty:        false, // Important for non-interactive log collection
                WorkingDir: "/app",
        }

        hostConfig := &container.HostConfig{
                Mounts: []mount.Mount{
                        {
                                Type:   mount.TypeBind,
                                Source: repoPath, // Absolute path on host
                                Target: "/app/code",
                        },
                        {
                                Type:   mount.TypeBind,
                                Source: logPath, // Absolute path on host
                                Target: "/app/output",
                        },
                },
                AutoRemove: true, // Remove container once it stops
        }

        tasks.UpdateTaskStatus(task.ID, tasks.StatusPreparing, "Creating container")
        resp, err := cli.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, containerName)
        if err != nil {
                log.Printf("Error creating container for task %s: %v", task.ID, err)
                tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Failed to create container")
                return
        }
        log.Printf("Container %s created for task %s", resp.ID, task.ID)

        tasks.UpdateTaskStatus(task.ID, tasks.StatusRunning, "Starting container")
        if err := cli.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
                log.Printf("Error starting container %s for task %s: %v", resp.ID, task.ID, err)
                tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Failed to start container")
                return
        }
        log.Printf("Container %s started for task %s", resp.ID, task.ID)

        // Wait for container to finish
        statusCh, errCh := cli.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)
        select {
        case err := <-errCh:
                if err != nil {
                        log.Printf("Error waiting for container %s for task %s: %v", resp.ID, task.ID, err)
                        tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Error during container execution")
                        // Attempt to get logs even on error
                        logContainerOutput(ctx, cli, resp.ID, task.ID, logPath)
                        return
                }
        case status := <-statusCh:
                log.Printf("Container %s for task %s finished with status code %d", resp.ID, task.ID, status.StatusCode)
                // Log container output regardless of status code
                logContainerOutput(ctx, cli, resp.ID, task.ID, logPath)

                if status.StatusCode == 0 {
                        // Check for pr_url.txt to update message
                        prURLFile := filepath.Join(logPath, "pr_url.txt")
                        if _, err := os.Stat(prURLFile); err == nil {
                                prURLBytes, _ := os.ReadFile(prURLFile)
                                prURL := strings.TrimSpace(string(prURLBytes))
                                if prURL != "" {
                                        tasks.UpdateTaskStatus(task.ID, tasks.StatusCompleted, "Completed. PR URL: "+prURL)
                                        return
                                }
                        }
                        tasks.UpdateTaskStatus(task.ID, tasks.StatusCompleted, "Container completed successfully")
                } else {
                        errMsg := fmt.Sprintf("Container exited with code %d. Check agent.log in output.", status.StatusCode)
                        // Check for error.txt
                        errorFile := filepath.Join(logPath, "error.txt")
                        if _, err := os.Stat(errorFile); err == nil {
                                errorBytes, _ := os.ReadFile(errorFile)
                                agentError := strings.TrimSpace(string(errorBytes))
                                if agentError != "" {
                                        errMsg = fmt.Sprintf("Container exited with code %d. Agent error: %s", status.StatusCode, agentError)
                                }
                        }
                        tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, errMsg)
                }
        case <-time.After(15 * time.Minute): // Timeout for container execution
                log.Printf("Container %s for task %s timed out.", resp.ID, task.ID)
                tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Container execution timed out")
                // Attempt to stop and log
                if err := cli.ContainerStop(context.Background(), resp.ID, container.StopOptions{}); err != nil {
                        log.Printf("Failed to stop timed-out container %s: %v", resp.ID, err)
                }
                logContainerOutput(ctx, cli, resp.ID, task.ID, logPath)
        }
}

func logContainerOutput(ctx context.Context, cli *client.Client, containerID, taskID, logPath string) {
        // Get container logs (stdout/stderr of the agent.py itself)
        // The agent.py also writes its own logs to /app/output/agent.log, this is supplementary.
        options := container.LogsOptions{ShowStdout: true, ShowStderr: true, Timestamps: true}
        out, err := cli.ContainerLogs(ctx, containerID, options)
        if err != nil {
                log.Printf("Error getting logs for container %s (task %s): %v", containerID, taskID, err)
                return
        }
        defer out.Close()

        dockerLogFilePath := filepath.Join(logPath, "docker_container.log")
        file, err := os.Create(dockerLogFilePath)
        if err != nil {
                log.Printf("Error creating docker_container.log for task %s: %v", taskID, err)
                return
        }
        defer file.Close()

        _, err = io.Copy(file, out)
        if err != nil {
                log.Printf("Error writing Docker logs to file for task %s: %v", taskID, err)
        }
        log.Printf("Docker container logs saved to %s for task %s", dockerLogFilePath, taskID)
}
```

---

### 13. `backend/api/handlers.go`

```go
package api

import (
        "codex-sys/backend/docker_utils"
        "codex-sys/backend/tasks"
        "codex-sys/backend/utils"
        "fmt"
        "io"
        "log"
        "mime/multipart"
        "net/http"
        "os"
        "os/exec"
        "path/filepath"
        "strings"

        "github.com/gin-gonic/gin"
)

func HandleCreateTask(c *gin.Context) {
        taskID := utils.GenerateTaskID()
        repoPath := utils.GetRepoPath(taskID)
        logPath := utils.GetLogPath(taskID)

        if err := os.MkdirAll(repoPath, 0755); err != nil {
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create repository directory", "details": err.Error()})
                return
        }
        if err := os.MkdirAll(logPath, 0755); err != nil {
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create log directory", "details": err.Error()})
                return
        }

        // Parse form data
        gitURL := c.PostForm("git_url")
        taskDescription := c.PostForm("task_description")
        targetFile := c.PostForm("target_file") // e.g., "src/main.py"
        userGitHubToken := c.PostForm("github_token") // Optional, task-specific token

        if taskDescription == "" {
                c.JSON(http.StatusBadRequest, gin.H{"error": "task_description is required"})
                return
        }
        if targetFile == "" {
                c.JSON(http.StatusBadRequest, gin.H{"error": "target_file is required"})
                return
        }
    // Sanitize target_file to prevent path traversal within the repo
    cleanTargetFile, err := utils.SanitizePath(targetFile)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid target_file path", "details": err.Error()})
        return
    }
    targetFile = cleanTargetFile


        var zipFileHeader *multipart.FileHeader
        var zipFile multipart.File
        var errUpload error

        if gitURL == "" {
                zipFile, zipFileHeader, errUpload = c.Request.FormFile("zip_file")
                if errUpload != nil {
                        c.JSON(http.StatusBadRequest, gin.H{"error": "git_url or zip_file is required", "details": errUpload.Error()})
                        return
                }
                defer zipFile.Close()
        }

        task := &tasks.Task{
                ID:              taskID,
                GitURL:          gitURL,
                TargetFile:      targetFile,
                TaskDescription: taskDescription,
                Status:          tasks.StatusPending,
                RepoPath:        repoPath,
                LogPath:         logPath,
                GitHubToken:     userGitHubToken, // Store user-provided token
        }

        tasks.AddTask(task)

        if gitURL != "" {
                tasks.UpdateTaskStatus(taskID, tasks.StatusCloning, "Cloning repository...")
                log.Printf("Cloning %s into %s for task %s", gitURL, repoPath, taskID)
                cmd := exec.Command("git", "clone", "--depth=1", gitURL, repoPath)
                output, err := cmd.CombinedOutput()
                if err != nil {
                        log.Printf("Git clone error for task %s: %s\nOutput: %s", taskID, err, string(output))
                        tasks.UpdateTaskStatus(taskID, tasks.StatusFailed, "Failed to clone Git repository: "+err.Error())
                        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to clone Git repository", "details": err.Error(), "output": string(output)})
                        return
                }
                log.Printf("Git clone successful for task %s", taskID)
                if strings.Contains(gitURL, "github.com") {
                        task.IsGitHubRepo = true // Mark if it's likely a GitHub repo for PR creation
                }

        } else if zipFileHeader != nil {
                tasks.UpdateTaskStatus(taskID, tasks.StatusPreparing, "Processing ZIP file...")
                task.ZipFileName = zipFileHeader.Filename
                tempZipPath := filepath.Join(repoPath, "_temp_"+zipFileHeader.Filename) // Store zip temporarily inside repoPath for simplicity
                
                out, err := os.Create(tempZipPath)
                if err != nil {
                        tasks.UpdateTaskStatus(taskID, tasks.StatusFailed, "Failed to save ZIP file")
                        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save ZIP file", "details": err.Error()})
                        return
                }
                defer out.Close()
                defer os.Remove(tempZipPath) // Clean up temp zip

                _, err = io.Copy(out, zipFile)
                if err != nil {
                        tasks.UpdateTaskStatus(taskID, tasks.StatusFailed, "Failed to copy ZIP file content")
                        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to copy ZIP file content", "details": err.Error()})
                        return
                }
                // Important: Close the file before unzipping
                out.Close()

                log.Printf("Unzipping %s into %s for task %s", tempZipPath, repoPath, taskID)
                _, err = utils.Unzip(tempZipPath, repoPath)
                if err != nil {
                        log.Printf("Unzip error for task %s: %v", taskID, err)
                        tasks.UpdateTaskStatus(taskID, tasks.StatusFailed, "Failed to unzip archive")
                        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to unzip archive", "details": err.Error()})
                        return
                }
                log.Printf("Unzip successful for task %s", taskID)
                task.IsGitHubRepo = false // ZIP uploads are not GitHub repos for PR purposes
        }

        // Start the Docker agent in a goroutine
        go docker_utils.RunAgentContainer(task)

        c.JSON(http.StatusOK, gin.H{
                "message":  "Task created successfully",
                "task_id":  taskID,
                "status":   tasks.StatusPending, // Initial status will be updated by async process
                "logs_url": fmt.Sprintf("/api/logs/%s/", taskID),
        })
}

func HandleGetTaskStatus(c *gin.Context) {
        taskID := c.Param("task_id")
        task, exists := tasks.GetTask(taskID)
        if !exists {
                c.JSON(http.StatusNotFound, gin.H{"error": "Task not found"})
                return
        }
        c.JSON(http.StatusOK, task)
}

func HandleGetLogFile(c *gin.Context) {
        taskID := c.Param("task_id")
        filename := c.Param("filename")

        // Sanitize filename to prevent path traversal
        cleanFilename, err := utils.SanitizePath(filename)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid filename", "details": err.Error()})
        return
    }
    filename = cleanFilename


        task, exists := tasks.GetTask(taskID)
        if !exists {
                c.JSON(http.StatusNotFound, gin.H{"error": "Task not found"})
                return
        }

        logFilePath := filepath.Join(task.LogPath, filename)

        // Security check: ensure the path is still within the intended log directory
        absLogPath, _ := filepath.Abs(task.LogPath)
        absRequestedPath, _ := filepath.Abs(logFilePath)
        if !strings.HasPrefix(absRequestedPath, absLogPath) {
                c.JSON(http.StatusForbidden, gin.H{"error": "Access to this file is forbidden"})
                return
        }

        if _, err := os.Stat(logFilePath); os.IsNotExist(err) {
                c.JSON(http.StatusNotFound, gin.H{"error": "Log file not found", "filename": filename})
                return
        }

        // Set appropriate content type for patches
        if strings.HasSuffix(filename, ".patch") || strings.HasSuffix(filename, ".diff") {
                c.Header("Content-Type", "text/x-diff")
        } else if strings.HasSuffix(filename, ".txt") || strings.HasSuffix(filename, ".log") {
                c.Header("Content-Type", "text/plain; charset=utf-8")
        }
        // Default to octet-stream if unsure, or let browser guess
        // c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=%s", filename)) // Optional: force download

        c.File(logFilePath)
}
```

---

### 14. `frontend/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codex-like SYS</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        label { display: block; margin-top: 10px; margin-bottom: 5px; }
        input[type="text"], input[type="file"], textarea, button {
            width: calc(100% - 22px); padding: 10px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #ddd;
        }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        .output, .logs { margin-top: 20px; padding: 10px; border: 1px solid #eee; border-radius: 4px; background-color: #f9f9f9; }
        .log-link { display: block; margin: 5px 0; }
        pre { background-color: #eee; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }
        .error { color: red; }
        .success { color: green; }
        #status { font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Codex-like System</h1>

        <form id="taskForm">
            <label for="inputType">Input Type:</label>
            <select id="inputType" name="inputType">
                <option value="git">Git URL</option>
                <option value="zip">ZIP File</option>
            </select>

            <div id="gitUrlGroup">
                <label for="gitUrl">Git Repository URL:</label>
                <input type="text" id="gitUrl" name="git_url" placeholder="https://github.com/user/repo.git">
            </div>

            <div id="zipFileGroup" style="display:none;">
                <label for="zipFile">Upload Code (ZIP):</label>
                <input type="file" id="zipFile" name="zip_file" accept=".zip">
            </div>

            <label for="taskDescription">Task Description:</label>
            <textarea id="taskDescription" name="task_description" rows="3" placeholder="e.g., Refactor function 'foo' in main.py to improve readability."></textarea>

            <label for="targetFile">Target File Path (relative to repo root):</label>
            <input type="text" id="targetFile" name="target_file" placeholder="e.g., src/main.py or README.md">
            
            <label for="githubToken">GitHub Token (Optional, for PR creation):</label>
            <input type="text" id="githubToken" name="github_token" placeholder="ghp_...">


            <button type="submit">Submit Task</button>
        </form>

        <div class="output">
            <h2>Task Status</h2>
            <p id="statusMessage">Submit a task to see its status.</p>
            <div id="taskIdDisplay" style="display:none;"><strong>Task ID:</strong> <span id="taskId"></span></div>
            <div id="currentStatusDisplay" style="display:none;"><strong>Status:</strong> <span id="status"></span></div>
            <div id="statusDetailDisplay" style="display:none;"><strong>Details:</strong> <span id="statusDetail"></span></div>
            <div id="logLinks" class="logs"></div>
        </div>

        <div class="output" id="diffPreviewContainer" style="display:none;">
            <h2>Diff Preview (<span id="diffFileName"></span>)</h2>
            <pre id="diffPreview"></pre>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>
```

---

### 15. `frontend/script.js`

```javascript
const API_BASE_URL = "http://localhost:8080/api"; // Adjust if your backend runs elsewhere

const taskForm = document.getElementById('taskForm');
const inputTypeSelect = document.getElementById('inputType');
const gitUrlGroup = document.getElementById('gitUrlGroup');
const zipFileGroup = document.getElementById('zipFileGroup');

const statusMessageEl = document.getElementById('statusMessage');
const taskIdDisplay = document.getElementById('taskIdDisplay');
const taskIdEl = document.getElementById('taskId');
const currentStatusDisplay = document.getElementById('currentStatusDisplay');
const statusEl = document.getElementById('status');
const statusDetailDisplay = document.getElementById('statusDetailDisplay');
const statusDetailEl = document.getElementById('statusDetail');

const logLinksEl = document.getElementById('logLinks');
const diffPreviewContainer = document.getElementById('diffPreviewContainer');
const diffFileNameEl = document.getElementById('diffFileName');
const diffPreviewEl = document.getElementById('diffPreview');

let pollingInterval;

inputTypeSelect.addEventListener('change', function() {
    if (this.value === 'git') {
        gitUrlGroup.style.display = 'block';
        zipFileGroup.style.display = 'none';
    } else {
        gitUrlGroup.style.display = 'none';
        zipFileGroup.style.display = 'block';
    }
});

taskForm.addEventListener('submit', async function(event) {
    event.preventDefault();
    clearPreviousResults();

    const formData = new FormData();
    const inputType = inputTypeSelect.value;

    if (inputType === 'git') {
        const gitUrl = document.getElementById('gitUrl').value;
        if (!gitUrl) {
            updateStatusDisplay("Please provide a Git URL.", "error");
            return;
        }
        formData.append('git_url', gitUrl);
    } else {
        const zipFile = document.getElementById('zipFile').files[0];
        if (!zipFile) {
            updateStatusDisplay("Please select a ZIP file.", "error");
            return;
        }
        formData.append('zip_file', zipFile);
    }

    const taskDescription = document.getElementById('taskDescription').value;
    const targetFile = document.getElementById('targetFile').value;
    const githubToken = document.getElementById('githubToken').value;


    if (!taskDescription) {
        updateStatusDisplay("Task description is required.", "error");
        return;
    }
    if (!targetFile) {
        updateStatusDisplay("Target file path is required.", "error");
        return;
    }

    formData.append('task_description', taskDescription);
    formData.append('target_file', targetFile);
    if (githubToken) {
        formData.append('github_token', githubToken);
    }


    statusMessageEl.textContent = "Submitting task...";
    statusMessageEl.className = '';

    try {
        const response = await fetch(`${API_BASE_URL}/task`, {
            method: 'POST',
            body: formData // FormData sets Content-Type to multipart/form-data automatically
        });

        const data = await response.json();

        if (!response.ok) {
            updateStatusDisplay(`Error: ${data.error || response.statusText} ${data.details ? '('+data.details+')': ''}`, "error");
            return;
        }

        statusMessageEl.textContent = "Task submitted successfully.";
        statusMessageEl.className = 'success';
        taskIdEl.textContent = data.task_id;
        taskIdDisplay.style.display = 'block';
        currentStatusDisplay.style.display = 'block';

        pollTaskStatus(data.task_id);

    } catch (error) {
        console.error("Submission error:", error);
        updateStatusDisplay(`Submission failed: ${error.message}`, "error");
    }
});

function updateStatusDisplay(message, type = "info") {
    statusMessageEl.textContent = message;
    statusMessageEl.className = type; // 'info', 'success', 'error'
    if (type === "error") {
        taskIdDisplay.style.display = 'none';
        currentStatusDisplay.style.display = 'none';
        statusDetailDisplay.style.display = 'none';
    }
}


function pollTaskStatus(taskId) {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/task/${taskId}/status`);
            if (!response.ok) {
                if (response.status === 404) {
                    updateStatusDisplay(`Task ${taskId} not found. Stopping polling.`, "error");
                    clearInterval(pollingInterval);
                } else {
                    statusEl.textContent = `Error fetching status: ${response.statusText}`;
                    statusEl.className = 'error';
                }
                return;
            }

            const data = await response.json();
            statusEl.textContent = data.Status;
            statusEl.className = data.Status === 'failed' ? 'error' : (data.Status === 'completed' ? 'success' : '');
            
            if (data.Message) {
                statusDetailEl.textContent = data.Message;
                statusDetailDisplay.style.display = 'block';
            } else {
                statusDetailDisplay.style.display = 'none';
            }


            if (data.Status === "completed" || data.Status === "failed") {
                clearInterval(pollingInterval);
                displayLogLinks(taskId, data.Status === "completed");
            }

        } catch (error) {
            console.error("Polling error:", error);
            statusEl.textContent = `Polling error: ${error.message}`;
            statusEl.className = 'error';
            // Optionally stop polling on certain errors
            // clearInterval(pollingInterval);
        }
    }, 2000); // Poll every 2 seconds
}

function displayLogLinks(taskId, isCompleted) {
    logLinksEl.innerHTML = '<h3>Task Outputs:</h3>';
    const commonLogs = ["prompt.txt", "llm_response.txt", "diff.patch", "agent.log", "docker_container.log", "setup.log"];
    if (isCompleted) {
        commonLogs.push("pr_url.txt"); // If completed, PR URL might exist
    } else {
         commonLogs.push("error.txt"); // If failed, error.txt might exist
    }


    commonLogs.forEach(logFile => {
        const link = document.createElement('a');
        link.href = `${API_BASE_URL}/logs/${taskId}/${logFile}`;
        link.textContent = logFile;
        link.target = "_blank";
        link.className = "log-link";
        logLinksEl.appendChild(link);

        if (logFile === "diff.patch" && isCompleted) {
            link.addEventListener('click', async (e) => {
                e.preventDefault(); // Prevent default navigation
                try {
                    const response = await fetch(link.href);
                    if (response.ok) {
                        const diffText = await response.text();
                        diffPreviewEl.textContent = diffText;
                        diffFileNameEl.textContent = logFile;
                        diffPreviewContainer.style.display = 'block';
                        window.open(link.href, '_blank'); // Still open in new tab
                    } else {
                        diffPreviewEl.textContent = `Could not load ${logFile}. Status: ${response.status}`;
                        diffPreviewContainer.style.display = 'block';
                    }
                } catch (err) {
                    diffPreviewEl.textContent = `Error loading ${logFile}: ${err.message}`;
                    diffPreviewContainer.style.display = 'block';
                }
            });
        }
    });
}

function clearPreviousResults() {
    if (pollingInterval) clearInterval(pollingInterval);
    statusMessageEl.textContent = "Submit a task to see its status.";
    statusMessageEl.className = '';
    taskIdDisplay.style.display = 'none';
    currentStatusDisplay.style.display = 'none';
    statusDetailDisplay.style.display = 'none';
    logLinksEl.innerHTML = '';
    diffPreviewContainer.style.display = 'none';
    diffPreviewEl.textContent = '';
    diffFileNameEl.textContent = '';
    document.getElementById('githubToken').value = ''; // Clear token field for safety
}

// Initial setup for input type visibility
inputTypeSelect.dispatchEvent(new Event('change'));
```

---

This comprehensive template should give you a very solid starting point.
Remember to:
1.  Fill in `.env` with your `OPENAI_API_KEY` (and optionally `GITHUB_TOKEN`).
2.  Build the agent Docker image: `cd dockerfiles/agent && docker build -t codex-agent:latest . && cd ../..`
3.  Run `go mod tidy` in the `backend` directory.
4.  Start the backend: `cd backend && go run main.go`
5.  Open `frontend/index.html` in your browser.

Good luck with your AI Code Workstation Platform!
