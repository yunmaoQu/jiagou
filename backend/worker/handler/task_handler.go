package handler

import (
	"codex-sys/worker/internal/config"
	"codex-sys/worker/internal/platform/database"
	"codex-sys/worker/internal/platform/k8s"
	"codex-sys/worker/internal/platform/objectstorage"
	"codex-sys/worker/internal/task"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type TaskHandler struct {
	db  *database.DBClientWrapper // Wrapper for sqlx.DB
	cos *objectstorage.ClientWrapper
	k8s *k8s.Client
	cfg config.WorkerConfig
}

func NewTaskHandler(db *database.DBClientWrapper, cos *objectstorage.ClientWrapper, k8s *k8s.Client, cfg config.WorkerConfig) *TaskHandler {
	return &TaskHandler{db: db, cos: cos, k8s: k8s, cfg: cfg}
}

// Handle is called by the Kafka consumer with a new message
func (h *TaskHandler) Handle(ctx context.Context, key []byte, value []byte) error {
	var msg task.KafkaTaskMessage // Use the shared struct
	if err := json.Unmarshal(value, &msg); err != nil {
		log.Printf("Worker: Failed to unmarshal Kafka message: %v. Message: %s", err, string(value))
		return nil // Acknowledge and drop malformed message
	}

	log.Printf("Worker: Received task %s for processing. Type: %s, Code: %s", msg.TaskID, msg.InputType, msg.CodeLocation)

	// 1. Update task status in DB
	if err := h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusProcessing, "Worker received task"); err != nil {
		log.Printf("Worker: Failed to update task %s status to Processing: %v", msg.TaskID, err)
		return err // Nack and retry
	}

	// --- Local temp directory for this task's operations ---
	// This should be on a fast, ephemeral storage if possible, or cleaned up diligently.
	// In K8s, this could be an emptyDir volume for the worker pod itself, or just /tmp.
	localTaskWorkspace := filepath.Join(os.TempDir(), "codex_worker_space", msg.TaskID)
	if err := os.MkdirAll(localTaskWorkspace, 0755); err != nil {
		h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, fmt.Sprintf("Failed to create local workspace: %v", err))
		return err
	}
	defer os.RemoveAll(localTaskWorkspace) // Cleanup

	// 2. Download/Prepare code
	if err := h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusDownloadingCode, "Preparing code"); err != nil { /* ... */
	}

	codeSourceDir := filepath.Join(localTaskWorkspace, "code_source") // Where raw code goes
	os.MkdirAll(codeSourceDir, 0755)

	if msg.InputType == string(taskdef.InputGit) {
		log.Printf("Worker: Cloning Git repo %s for task %s", msg.CodeLocation, msg.TaskID)
		cmd := exec.Command("git", "clone", "--depth=1", msg.CodeLocation, codeSourceDir)
		if output, err := cmd.CombinedOutput(); err != nil {
			errMsg := fmt.Sprintf("Failed to clone Git repo: %v. Output: %s", err, string(output))
			log.Printf("Worker: Task %s: %s", msg.TaskID, errMsg)
			h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, errMsg)
			return nil // Acknowledge, don't retry clone errors usually
		}
	} else if msg.InputType == string(taskdef.InputZip) {
		// msg.CodeLocation is S3 key like "tmp_zips/<task_id>/code.zip"
		// Bucket name comes from config
		zipFileName := filepath.Base(msg.CodeLocation) // e.g. code.zip
		localZipPath := filepath.Join(localTaskWorkspace, zipFileName)

		log.Printf("Worker: Downloading ZIP %s from COS for task %s", msg.CodeLocation, msg.TaskID)
		zipFile, err := os.Create(localZipPath)
		if err != nil { /* ... error handling & DB update ... */
			return nil
		}

		err = h.cos.DownloadFile(ctx, h.cfg.COSCodeBucket, msg.CodeLocation, zipFile)
		zipFile.Close() // Close before potential error check or unzip
		if err != nil {
			errMsg := fmt.Sprintf("Failed to download ZIP from COS: %v", err)
			log.Printf("Worker: Task %s: %s", msg.TaskID, errMsg)
			h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, errMsg)
			os.Remove(localZipPath)
			return nil
		}

		log.Printf("Worker: Unzipping %s for task %s", localZipPath, msg.TaskID)
		// Use a robust unzip utility. For simplicity, assume utils.Unzip exists.
		// _, err = utils.Unzip(localZipPath, codeSourceDir)
		// A placeholder:
		cmd := exec.Command("unzip", "-o", localZipPath, "-d", codeSourceDir)
		if output, err := cmd.CombinedOutput(); err != nil {
			errMsg := fmt.Sprintf("Failed to unzip archive: %v. Output: %s", err, string(output))
			log.Printf("Worker: Task %s: %s", msg.TaskID, errMsg)
			h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, errMsg)
			os.Remove(localZipPath)
			return nil
		}
		os.Remove(localZipPath) // Clean up downloaded zip
	} else {
		errMsg := fmt.Sprintf("Unknown input type: %s", msg.InputType)
		log.Printf("Worker: Task %s: %s", msg.TaskID, errMsg)
		h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, errMsg)
		return nil
	}
	log.Printf("Worker: Code prepared for task %s in %s", msg.TaskID, codeSourceDir)

	// --- At this point, codeSourceDir contains the user's code ---
	// Option A: Upload codeSourceDir to a new COS location for the Agent Pod to mount via CSI.
	// Option B: Create a K8s PVC, copy codeSourceDir to it, then Agent Pod mounts the PVC.
	// Option C (Simpler for demo, but less scalable): Build a tarball, pass it to Agent Pod via initContainer or configmap (only for small repos).

	// Let's assume Option A: Upload to a new COS path for this specific task's execution
	agentCodeCOSPath := fmt.Sprintf("agent_code_input/%s/", msg.TaskID) // Prefix in COS
	log.Printf("Worker: Uploading processed code for task %s to COS path %s", msg.TaskID, agentCodeCOSPath)

	// Walk codeSourceDir and upload files to COS under agentCodeCOSPath
	// This needs a recursive upload function in h.cos
	err := h.cos.UploadDirectory(ctx, h.cfg.COSCodeBucket, agentCodeCOSPath, codeSourceDir)
	if err != nil {
		errMsg := fmt.Sprintf("Failed to upload agent input code to COS: %v", err)
		log.Printf("Worker: Task %s: %s", msg.TaskID, errMsg)
		h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, errMsg)
		return nil
	}
	log.Printf("Worker: Agent input code uploaded to COS for task %s", msg.TaskID)

	// 3. Define and Run K8s Job for the Agent
	if err := h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusRunningAgent, "Launching agent"); err != nil { /* ... */
	}

	agentJobName := fmt.Sprintf("codex-agent-%s", strings.ReplaceAll(msg.TaskID, "_", "-")) // K8s names are restrictive

	// Prepare Agent environment variables
	// Fetch OpenAI API Key from a K8s secret or environment variable available to the Worker pod
	openaiAPIKey := os.Getenv("WORKER_OPENAI_API_KEY") // Or from cfg
	if openaiAPIKey == "" {
		log.Printf("Worker: Task %s: WORKER_OPENAI_API_KEY not set for worker, cannot pass to agent.", msg.TaskID)
		// ... handle error ...
		h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, "OpenAI API Key not configured for agent")
		return nil
	}
	agentEnvVars := map[string]string{
		"OPENAI_API_KEY": openaiAPIKey,
		"GITHUB_TOKEN":   msg.UserGitHubToken, // If provided
		// Other env vars for agent
	}

	// Define how agent gets code and writes output
	// This is where COS CSI or other volume strategies come in.
	// For simplicity, let's imagine a CSI driver that mounts a COS path.
	// Or, an initContainer in the Agent Pod could download from agentCodeCOSPath.
	// And another sidecar/post-run step to upload /app/output.

	// Simplified: Assuming K8s client can create a Job that somehow gets code from `agentCodeCOSPath`
	// and makes `/app/output` available for retrieval after completion.
	agentCmd := []string{msg.TaskDescription, msg.TargetFile, "true"} // Assuming is_github_repo is true for now

	jobConfig := k8s.AgentJobConfig{
		Name:           agentJobName,
		Namespace:      h.cfg.K8sAgentNamespace,
		Image:          h.cfg.AgentDockerImage, // e.g., "your-registry/codex-agent:latest"
		Command:        agentCmd,
		EnvVars:        agentEnvVars,
		CodeCOSPath:    fmt.Sprintf("s3://%s/%s", h.cfg.COSCodeBucket, agentCodeCOSPath), // For CSI or initContainer
		OutputCOSPath:  fmt.Sprintf("s3://%s/logs/%s/", h.cfg.COSLogsBucket, msg.TaskID), // For sidecar or post-run upload
		CPULimit:       "1",
		MemoryLimit:    "2Gi",
		ServiceAccount: h.cfg.K8sAgentServiceAccount, // If agent needs K8s permissions or cloud permissions via IRSA/Workload Identity
	}

	err = h.k8s.RunAgentJob(ctx, jobConfig)
	if err != nil {
		errMsg := fmt.Sprintf("Failed to launch K8s agent job: %v", err)
		log.Printf("Worker: Task %s: %s", msg.TaskID, errMsg)
		h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusFailed, errMsg)
		return nil
	}
	log.Printf("Worker: K8s agent job %s launched for task %s", agentJobName, msg.TaskID)

	// 4. Monitor K8s Job (this should be asynchronous or have a timeout)
	// The k8s.RunAgentJob might be synchronous for completion or return immediately.
	// If it returns immediately, worker needs a separate mechanism to monitor jobs.
	// For this example, let's assume RunAgentJob waits or the K8s client has a watch.
	// In a real system, you might update DB to "AGENT_RUNNING" and have another process poll K8s.

	// After job completion (success or failure):
	// Assume job status (success/failure) is determined by k8s.RunAgentJob or a subsequent check.
	// Assume outputs from /app/output are automatically uploaded to `jobConfig.OutputCOSPath`
	// by a sidecar in the agent pod or by the k8s.RunAgentJob logic upon completion.

	// For now, let's simulate a wait and assume success/failure is known
	// This part needs robust K8s job monitoring.
	time.Sleep(30 * time.Second) // Placeholder for actual job monitoring

	// Example: Check job status
	jobStatus, err := h.k8s.GetJobStatus(ctx, jobConfig.Namespace, jobConfig.Name)
	if err != nil {
		log.Printf("Worker: Failed to get K8s job status for %s: %v", jobConfig.Name, err)
		// Decide if this is a retryable error or task failure
	}

	finalStatus := task.StatusFailed
	finalMessage := "Agent job finished with unknown status."

	if jobStatus == k8s.JobSucceeded {
		finalStatus = task.StatusCompleted
		finalMessage = "Agent processing completed successfully."
		// Check for PR URL if applicable (e.g., by reading a specific file from COS output)
		// prURL, _ := h.cos.ReadFileContent(ctx, h.cfg.COSLogsBucket, fmt.Sprintf("logs/%s/pr_url.txt", msg.TaskID))
		// if prURL != "" { h.db.UpdateTaskPRURL(ctx, msg.TaskID, prURL) }

	} else if jobStatus == k8s.JobFailed {
		finalMessage = "Agent processing failed. Check logs on COS."
	} else { // Still running or other state
		finalMessage = fmt.Sprintf("Agent job status: %s. Monitoring may have timed out.", jobStatus)
	}

	// 5. Update final task status in DB
	if err := h.db.UpdateTaskStatus(ctx, msg.TaskID, finalStatus, finalMessage); err != nil {
		log.Printf("Worker: Failed to update final task %s status: %v", msg.TaskID, err)
		// This is tricky, task might be done but DB update fails. Retry logic for DB important.
	}

	log.Printf("Worker: Finished handling task %s. Final status: %s", msg.TaskID, finalStatus)
	return nil // Acknowledge message from Kafka
}
