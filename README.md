

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
*   Support for more LLMs (Claude3.7thinkingMax, local models).
*   More sophisticated agent capabilities.
*   WebSockets for real-time log streaming.
*   Job queue (e.g., Pulsar) for better task management.
*   More robust error handling and retry mechanisms.
*   Configurable container resource limits.

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


1.  Fill in `.env` with your `OPENAI_API_KEY` (and optionally `GITHUB_TOKEN`).
2.  Build the agent Docker image: `cd dockerfiles/agent && docker build -t codex-agent:latest . && cd ../..`
3.  Run `go mod tidy` in the `backend` directory.
4.  Start the backend: `cd backend && go run main.go`
5.  Open `frontend/index.html` in your browser.



