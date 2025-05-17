package utils

//docker_utils
import (
	"codex-sys/backend/tasks"
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
