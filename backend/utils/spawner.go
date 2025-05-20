package utils

import (
	"context"
	"fmt"
	"github.com/jmoiron/sqlx"
	"github.com/yunmaoQu/codex-sys/internal/task"
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

var DB *sqlx.DB

// InitDB initializes the database connection for the utils package
func InitDB(db *sqlx.DB) {
	DB = db
}

func RunAgentContainer(taskDef *task.Definition) {
	ctx := context.Background()
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		log.Printf("Error creating Docker client for task %s: %v", taskDef.ID, err)
		// Update task status in database
		if DB != nil {
			err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, "Failed to create Docker client")
			if err != nil {
				log.Printf("Error updating task status: %v", err)
			}
		} else {
			log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, "Failed to create Docker client")
		}
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
				log.Printf("Error pulling image %s for task %s: %v", AgentImageName, taskDef.ID, pullErr)
				// Update task status in database
				if DB != nil {
					err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, fmt.Sprintf("Failed to pull agent image: %s", AgentImageName))
					if err != nil {
						log.Printf("Error updating task status: %v", err)
					}
				} else {
					log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, fmt.Sprintf("Failed to pull agent image: %s", AgentImageName))
				}
				return
			}
			defer reader.Close()
			io.Copy(os.Stdout, reader) // Show pull progress
			log.Printf("Image %s pulled successfully.", AgentImageName)
		} else {
			log.Printf("Error inspecting image %s for task %s: %v", AgentImageName, taskDef.ID, err)
			// Update task status in database
			if DB != nil {
				err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, "Failed to inspect agent image")
				if err != nil {
					log.Printf("Error updating task status: %v", err)
				}
			} else {
				log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, "Failed to inspect agent image")
			}
			return
		}
	}

	containerName := fmt.Sprintf("codex-agent-%s", taskDef.ID)
	repoPath := GetRepoPath(taskDef.ID)
	logPath := GetLogPath(taskDef.ID)

	// Ensure logPath directory exists for mounting
	if err := os.MkdirAll(logPath, 0755); err != nil {
		log.Printf("Error creating log directory %s for task %s: %v", logPath, taskDef.ID, err)
		// Update task status in database
		if DB != nil {
			err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, "Failed to create log directory")
			if err != nil {
				log.Printf("Error updating task status: %v", err)
			}
		} else {
			log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, "Failed to create log directory")
		}
		return
	}

	envVars := []string{
		fmt.Sprintf("OPENAI_API_KEY=%s", os.Getenv("OPENAI_API_KEY")),
		fmt.Sprintf("OPENAI_MODEL=%s", os.Getenv("OPENAI_MODEL")), // Pass model if set
	}
	// Use task-specific GitHub token if provided, otherwise use the one from .env
	githubToken := taskDef.GitHubToken
	if githubToken == "" {
		githubToken = os.Getenv("GITHUB_TOKEN")
	}
	if githubToken != "" {
		envVars = append(envVars, fmt.Sprintf("GITHUB_TOKEN=%s", githubToken))
	}

	// Determine if it's a GitHub repo based on the GitURL
	isGitHubRepo := strings.Contains(taskDef.GitURL, "github.com")

	containerConfig := &container.Config{
		Image: AgentImageName,
		Cmd: []string{
			taskDef.TaskDescription,
			taskDef.TargetFile, // Relative path for the agent
			fmt.Sprintf("%t", isGitHubRepo),
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

	// Update task status in database
	if DB != nil {
		err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusPending, "Creating container")
		if err != nil {
			log.Printf("Error updating task status: %v", err)
		}
	} else {
		log.Printf("Task %s status: PENDING - %s (DB not initialized)", taskDef.ID, "Creating container")
	}
	resp, err := cli.ContainerCreate(ctx, containerConfig, hostConfig, nil, nil, containerName)
	if err != nil {
		log.Printf("Error creating container for task %s: %v", taskDef.ID, err)
		// Update task status in database
		if DB != nil {
			err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, "Failed to create container")
			if err != nil {
				log.Printf("Error updating task status: %v", err)
			}
		} else {
			log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, "Failed to create container")
		}
		return
	}
	log.Printf("Container %s created for task %s", resp.ID, taskDef.ID)

	// Update task status in database
	if DB != nil {
		err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusRunningAgent, "Starting container")
		if err != nil {
			log.Printf("Error updating task status: %v", err)
		}
	} else {
		log.Printf("Task %s status: RUNNING_AGENT - %s (DB not initialized)", taskDef.ID, "Starting container")
	}
	if err := cli.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		log.Printf("Error starting container %s for task %s: %v", resp.ID, taskDef.ID, err)
		// Update task status in database
		if DB != nil {
			err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, "Failed to start container")
			if err != nil {
				log.Printf("Error updating task status: %v", err)
			}
		} else {
			log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, "Failed to start container")
		}
		return
	}
	log.Printf("Container %s started for task %s", resp.ID, taskDef.ID)

	// Wait for container to finish
	statusCh, errCh := cli.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			log.Printf("Error waiting for container %s for task %s: %v", resp.ID, taskDef.ID, err)
			// Update task status in database
			if DB != nil {
				err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, "Error during container execution")
				if err != nil {
					log.Printf("Error updating task status: %v", err)
				}
			} else {
				log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, "Error during container execution")
			}
			// Attempt to get logs even on error
			logContainerOutput(ctx, cli, resp.ID, taskDef.ID, logPath)
			return
		}
	case status := <-statusCh:
		log.Printf("Container %s for task %s finished with status code %d", resp.ID, taskDef.ID, status.StatusCode)
		// Log container output regardless of status code
		logContainerOutput(ctx, cli, resp.ID, taskDef.ID, logPath)

		if status.StatusCode == 0 {
			// Check for pr_url.txt to update message
			prURLFile := filepath.Join(logPath, "pr_url.txt")
			if _, err := os.Stat(prURLFile); err == nil {
				prURLBytes, _ := os.ReadFile(prURLFile)
				prURL := strings.TrimSpace(string(prURLBytes))
				if prURL != "" {
					// Update task status in database
					if DB != nil {
						err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusCompleted, "Completed. PR URL: "+prURL)
						if err != nil {
							log.Printf("Error updating task status: %v", err)
						}
					} else {
						log.Printf("Task %s status: COMPLETED - %s (DB not initialized)", taskDef.ID, "Completed. PR URL: "+prURL)
					}
					return
				}
			}
			// Update task status in database
			if DB != nil {
				err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusCompleted, "Container completed successfully")
				if err != nil {
					log.Printf("Error updating task status: %v", err)
				}
			} else {
				log.Printf("Task %s status: COMPLETED - %s (DB not initialized)", taskDef.ID, "Container completed successfully")
			}
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
			// Update task status in database
			if DB != nil {
				err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, errMsg)
				if err != nil {
					log.Printf("Error updating task status: %v", err)
				}
			} else {
				log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, errMsg)
			}
		}
	case <-time.After(15 * time.Minute): // Timeout for container execution
		log.Printf("Container %s for task %s timed out.", resp.ID, taskDef.ID)
		// Update task status in database
		if DB != nil {
			err := task.UpdateStatus(context.Background(), DB, taskDef.ID, task.StatusFailed, "Container execution timed out")
			if err != nil {
				log.Printf("Error updating task status: %v", err)
			}
		} else {
			log.Printf("Task %s status: FAILED - %s (DB not initialized)", taskDef.ID, "Container execution timed out")
		}
		// Attempt to stop and log
		// For Docker API v24.0.7, we need to use container.StopOptions{}
		if err := cli.ContainerStop(context.Background(), resp.ID, container.StopOptions{}); err != nil {
			log.Printf("Failed to stop timed-out container %s: %v", resp.ID, err)
		}
		logContainerOutput(ctx, cli, resp.ID, taskDef.ID, logPath)
	}
}

func logContainerOutput(ctx context.Context, cli *client.Client, containerID, taskID, logPath string) {
	// Get container logs (stdout/stderr of the agent.py itself)
	// The agent.py also writes its own logs to /app/output/agent.log, this is supplementary.
	logs, err := cli.ContainerLogs(ctx, containerID, types.ContainerLogsOptions{ShowStdout: true, ShowStderr: true, Timestamps: true})
	if err != nil {
		log.Printf("Error getting logs for container %s (task %s): %v", containerID, taskID, err)
		return
	}
	defer logs.Close()

	dockerLogFilePath := filepath.Join(logPath, "docker_container.log")
	file, err := os.Create(dockerLogFilePath)
	if err != nil {
		log.Printf("Error creating docker_container.log for task %s: %v", taskID, err)
		return
	}
	defer file.Close()

	_, err = io.Copy(file, logs)
	if err != nil {
		log.Printf("Error writing Docker logs to file for task %s: %v", taskID, err)
	}
	log.Printf("Docker container logs saved to %s for task %s", dockerLogFilePath, taskID)
}
