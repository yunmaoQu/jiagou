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

    const submitButton = document.querySelector('#taskForm button[type="submit"]');
    const originalButtonText = submitButton.textContent;
    submitButton.innerHTML = '<span class="spinner"></span> Processing...';
    submitButton.disabled = true;

    try {
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
    } catch (error) {
        console.error("General error in form submission:", error);
        updateStatusDisplay(`An error occurred during form submission: ${error.message}`, "error");
    } finally {
        const submitButton = document.querySelector('#taskForm button[type="submit"]');
        submitButton.textContent = originalButtonText;
        submitButton.disabled = false;
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
