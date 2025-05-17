package api

import (
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
	targetFile := c.PostForm("target_file")       // e.g., "src/main.py"
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
	go utils.RunAgentContainer(task)

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
