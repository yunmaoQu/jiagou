# 🧠 Codex System

A modular AI-powered system to analyze and modify codebases in a secure, isolated environment. Users can submit code (via Git URL or ZIP upload) and describe tasks for an AI agent to perform—like refactoring, bug fixing, or documentation. The agent can generate diffs, logs, and even create **GitHub Pull Requests** automatically.

---

## ✨ Features
![Go](https://img.shields.io/badge/Go-1.20+-00ADD8?logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Dockerized](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/github/license/yunmaoQu/Codex-like-Sys?color=green)
![Issues](https://img.shields.io/github/issues/yunmaoQu/Codex-like-Sys)
![Stars](https://img.shields.io/github/stars/yunmaoQu/Codex-like-Sys?style=social)
- ✅ **Code Input**: Supports Git URLs and ZIP file uploads.
- 🔐 **Isolated Execution**: Each task runs in its own Docker container.
- 🤖 **LLM Integration**: Uses OpenAI (configurable for others) to understand and modify code.
- 📦 **Output Artifacts**: Generates diffs, logs, and optionally creates GitHub PRs.
- 🌐 **Web API**: RESTful API for task management.
- 🖼️ **Simple Frontend**: Basic UI to interact with the system.

---

## 📁 Project Structure

```
codex-sys/
├── backend/                  # Go backend service
├── dockerfiles/agent/        # Docker setup for the Python agent
├── frontend/                 # Simple HTML/JS frontend
├── storage/                  # Runtime data (code, logs) - gitignored
├── .env.example              # Environment variable template
└── README.md                 # This file
```

---

## ⚙️ Prerequisites

- [Go](https://golang.org/doc/install) `v1.20+`
- [Docker](https://docs.docker.com/get-docker/)
- Python 3.x (used inside Docker)
- [Git](https://git-scm.com/)
- OpenAI API Key ([Get yours here](https://platform.openai.com/account/api-keys))
- **Optional**: GitHub Personal Access Token (for PRs) with `repo` scope

---

## 🚀 Setup Guide

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url> codex-sys
cd codex-sys
```

### 2️⃣ Configure Environment Variables

```bash
cp .env.example .env
nano .env  # or use your favorite editor
```

Update:
- `OPENAI_API_KEY`
- (Optional) `GITHUB_TOKEN`

### 3️⃣ Build the Agent Docker Image

```bash
cd dockerfiles/swe
docker build -t codex-swe:latest .
cd ../..
```

> ⚠️ If you change `agent.py` or its dependencies, rebuild the image.

### 4️⃣ Prepare Backend Dependencies

```bash
cd backend
go mod tidy
cd ..
```

---

## 🏃 Running the System

### ✅ Start Backend

```bash
cd backend
go run main.go
```

- Runs at: `http://localhost:8080`
- Creates:
  - `storage/repos/`
  - `storage/logs/`

### 🌐 Access Frontend

Open `/localhost:8080/ui` path in your browser.

> ⚠️ For production, serve the frontend via backend or a web server with CORS support.

---

## 🧬 How It Works

### 1. User Interaction

- Submit a Git URL or ZIP file
- Enter a **task description** (e.g., _"Refactor this function"_)
- Provide the **target file path** (e.g., `src/main.py`)
- (Optionally) Supply a GitHub token

### 2. Backend Processing

- Receives task via `/api/task`
- Creates a unique `task_id`
- Clones or extracts repo into `storage/repos/<task_id>`
- Spawns a Docker container:
  - Mounts code to `/app/code`
  - Mounts logs to `/app/output`
  - Passes `OPENAI_API_KEY`, `GITHUB_TOKEN` as env vars
  - Invokes Python agent with task + target file

### 3. Agent Execution (Inside Docker)

- Runs `agent.py`
- Reads `AGENTS.md.example` (optional task hints)
- Loads target file, builds prompt
- Calls OpenAI API
- Writes outputs:
  - `prompt.txt`
  - `llm_response.txt`
  - `diff.patch`
  - `agent.log`
- If GitHub token and valid Git repo:
  - Creates a branch
  - Commits + pushes changes
  - Opens a Pull Request 🎉
- Runs `setup.sh.example` if present

### 4. Results & Logging

- Backend monitors job status
- Logs and diffs saved in `storage/logs/<task_id>`
- Available via API: `/api/logs/<task_id>/<filename>`

---

## 📡 API Endpoints

### `POST /api/task`

Creates a new code task.

**Body (form-data):**

| Field           | Type     | Required | Description                                  |
|----------------|----------|----------|----------------------------------------------|
| `git_url`       | string   | optional | Git repository URL                           |
| `zip_file`      | file     | optional | ZIP archive of the codebase                  |
| `task_description` | string | ✅       | Task for the AI agent                        |
| `target_file`   | string   | ✅       | Relative path to the file (e.g. `src/app.py`)|
| `github_token`  | string   | optional | GitHub token (overrides `.env` if provided)  |

---

### `GET /api/task/{task_id}/status`

Returns task status (`pending`, `running`, `completed`, `failed`).

---

### `GET /api/logs/{task_id}/{filename}`

Download logs or result files:
- `prompt.txt`
- `llm_response.txt`
- `diff.patch`
- `agent.log`

---

## 🔐 Security Considerations

- 🧱 **Container Isolation**: Each task runs in a clean Docker container.
- 🧮 **Resource Limits**: Add CPU/memory limits to prevent abuse.
- 🌐 **Network Isolation**: Disable internet in agent if not needed.
- 🧼 **Input Sanitization**: Validate repo URLs, file paths, etc.
- 🔐 **Secrets Management**: Use a secure store for API keys and tokens.
- ⚠️ **`setup.sh` Warning**: Currently executed *as-is*. Sandbox or restrict in production.

---

## 🛠️ Future Enhancements

- Persistent task storage (e.g. PostgreSQL, SQLite)
- Authentication & authorization
- Multi-LLM support (Claude, local models)
- Real-time log streaming via WebSockets
- Job queue system (e.g. Pulsar, Redis)
- Configurable Docker resource limits
- Advanced error handling and retries

---

## 🧪 Example `.env` File

```ini
# Backend Config
BACKEND_PORT=8080
STORAGE_PATH=./storage

# LLM API Key
OPENAI_API_KEY="sk-your-openai-api-key"

# Optional GitHub Token (used by swe for PRs)
GITHUB_TOKEN=""
```

---

## 🏁 Quick Start Checklist

1. ✅ Fill in `.env` with your OpenAI key (and GitHub token)
2. 🐳 Build Docker image:  
   ```bash
   cd dockerfiles/swe && docker build -t codex-swe:latest . && cd ../..
   ```
3. 🧹 Tidy Go modules:  
   ```bash
   cd backend && go mod tidy && cd ..
   ```
4. 🚀 Start backend:  
   ```bash
   cd backend && go run main.go
   ```
5. 🌍 Open `frontend/index.html` in your browser

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📄 License

[MIT](LICENSE)


## 📈 Star Trending

[![Star History Chart](https://api.star-history.com/svg?repos=yunmaoQu/Codex-like-Sys&type=Date)](https://star-history.com/#yunmaoQu/Codex-like-Sys&Date)
