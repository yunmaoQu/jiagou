## Codex-like SYS dev guide

1.  **Backend (Go):**
    *   API endpoints for task creation (`/task`) and status/log retrieval.
    *   Logic for Git cloning and ZIP extraction.
    *   Docker container spawning and management.
    *   Task state management .
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

---

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