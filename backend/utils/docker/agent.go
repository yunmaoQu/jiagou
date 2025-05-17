package docker

import (
	"codex-sys/backend/tasks"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
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

	// Create container with the following configuration
	resp, err := cli.ContainerCreate(ctx, &container.Config{
		Image: AgentImageName,
		Cmd:   []string{task.TaskDescription}, // Pass task description as command
		Env: []string{
			fmt.Sprintf("TASK_ID=%s", task.ID),
			fmt.Sprintf("TARGET_FILE=%s", task.TargetFile),
			fmt.Sprintf("GITHUB_TOKEN=%s", task.GitHubToken),
		},
	}, &container.HostConfig{
		AutoRemove: true, // Clean up container after it exits
		Mounts: []mount.Mount{
			{
				Type:   mount.TypeBind,
				Source: task.RepoPath,
				Target: "/app/workspace",
			},
			{
				Type:   mount.TypeBind,
				Source: task.LogPath,
				Target: "/app/output",
			},
		},
	}, nil, nil, containerName)

	if err != nil {
		log.Printf("Error creating container for task %s: %v", task.ID, err)
		tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Failed to create container")
		return
	}

	// Start container
	log.Printf("Starting container %s for task %s", containerName, task.ID)
	tasks.UpdateTaskStatus(task.ID, tasks.StatusRunning, "Starting container")

	if err := cli.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		log.Printf("Error starting container for task %s: %v", task.ID, err)
		tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Failed to start container")
		return
	}

	// Stream logs in background
	go func() {
		logOptions := types.ContainerLogsOptions{
			ShowStdout: true,
			ShowStderr: true,
			Follow:     true,
			Timestamps: true,
		}

		// Get container logs
		reader, err := cli.ContainerLogs(ctx, resp.ID, logOptions)
		if err != nil {
			log.Printf("Error getting container logs for task %s: %v", task.ID, err)
			return
		}
		defer reader.Close()

		logFile, err := os.Create(filepath.Join(task.LogPath, "container.log"))
		if err != nil {
			log.Printf("Error creating log file for task %s: %v", task.ID, err)
			return
		}
		defer logFile.Close()

		// Copy logs to file
		_, err = io.Copy(logFile, reader)
		if err != nil && err != io.EOF {
			log.Printf("Error copying container logs for task %s: %v", task.ID, err)
		}
	}()

	// Wait for container to finish in background
	statusCh, errCh := cli.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			log.Printf("Error waiting for container for task %s: %v", task.ID, err)
			tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Container wait error")
		}
	case status := <-statusCh:
		log.Printf("Container for task %s finished with status code %d", task.ID, status.StatusCode)
		if status.StatusCode == 0 {
			tasks.UpdateTaskStatus(task.ID, tasks.StatusCompleted, "Task completed successfully")
		} else {
			tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, fmt.Sprintf("Container exited with status %d", status.StatusCode))
		}
	case <-time.After(30 * time.Minute): // Timeout after 30 minutes
		log.Printf("Container timeout for task %s", task.ID)
		if err := cli.ContainerStop(ctx, resp.ID, container.StopOptions{}); err != nil {
			log.Printf("Error stopping container for task %s: %v", task.ID, err)
		}
		tasks.UpdateTaskStatus(task.ID, tasks.StatusFailed, "Task timeout after 30 minutes")
	}
}
