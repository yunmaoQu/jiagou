package api

import (
	"context"
	"fmt"
	"github.com/yunmaoQu/codex-sys/internal/config"
	"log"
	"net/http"
	"time"

	"github.com/yunmaoQu/codex-sys/internal/platform/kafka"
	"github.com/yunmaoQu/codex-sys/internal/platform/objectstorage"
	"github.com/yunmaoQu/codex-sys/internal/task"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
)

type TaskAPI struct {
	db            *sqlx.DB
	kafkaProducer *kafka.Producer
	cosClient     *objectstorage.ClientWrapper
	cfg           *config.Config
}

func RegisterRoutes(router *gin.Engine, db *sqlx.DB, kp *kafka.Producer, cosClient *objectstorage.ClientWrapper) {
	api := &TaskAPI{db: db, kafkaProducer: kp, cosClient: cosClient}
	router.POST("/api/task", api.HandleCreateTask)
	router.GET("/api/task/:task_id/status", api.HandleGetTaskStatus)
	router.GET("/logs/:task_id/:filename", api.HandleGetLogFile)
}

func (api *TaskAPI) HandleCreateTask(c *gin.Context) {
	gitURL := c.PostForm("git_url")
	taskDescription := c.PostForm("task_description")
	targetFile := c.PostForm("target_file")

	taskID := uuid.NewString()
	newTask := task.Definition{
		ID:              taskID,
		TaskDescription: taskDescription,
		TargetFile:      targetFile,
		Status:          task.StatusQueued,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	var codeLocationS3Path string

	if gitURL != "" {
		newTask.InputType = task.InputGit
		newTask.GitURL = gitURL
		codeLocationS3Path = gitURL // Worker will handle cloning
	} else {
		zipFile, zipFileHeader, errUpload := c.Request.FormFile("zip_file")
		if errUpload != nil { /* ... error handling ... */
			return
		}
		defer zipFile.Close()

		newTask.InputType = task.InputZip
		newTask.ZipFileName = zipFileHeader.Filename

		// Upload ZIP to COS
		cosKey := fmt.Sprintf("tmp_zips/%s/%s", taskID, zipFileHeader.Filename)
		err := api.cosClient.UploadFile(context.Background(), api.cfg.COS.Buckets.Code, cosKey, zipFile, zipFileHeader.Size)
		if err != nil {
			log.Printf("Failed to upload ZIP to COS for task %s: %v", taskID, err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to upload code to storage"})
			return
		}
		newTask.CodePathCOS = fmt.Sprintf("s3://%s/%s", api.cfg.COS.Buckets.Code, cosKey)
		log.Printf("Uploaded ZIP to COS: %s", codeLocationS3Path)
	}

	// Save task to MySQL
	if err := task.Create(c.Request.Context(), api.db, &newTask); err != nil {
		log.Printf("Failed to save task to DB for task %s: %v", taskID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create task record"})
		return
	}

	// Publish task to Kafka
	kafkaMsg := task.KafkaTaskMessage{
		TaskID:             taskID,
		InputType:          string(newTask.InputType),
		CodeLocation:       codeLocationS3Path, // Git URL or S3 path to ZIP
		TargetFile:         targetFile,
		TaskDescription:    taskDescription,
		UserGitHubToken:    c.PostForm("github_token"), // Pass if provided
		OpenAIAPIKeySource: "env_or_secret_manager",    // Indicate where worker should get it
	}
	if err := api.kafkaProducer.Publish(c.Request.Context(), taskID, kafkaMsg); err != nil {
		log.Printf("Failed to publish task to Kafka for task %s: %v", taskID, err)
		// Potentially rollback DB transaction or mark task as failed to publish
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to queue task"})
		return
	}

	log.Printf("Task %s created and published to Kafka.", taskID)
	c.JSON(http.StatusOK, gin.H{"task_id": taskID, "status": task.StatusQueued})
}

func (api *TaskAPI) HandleGetTaskStatus(c *gin.Context) {
	taskID := c.Param("task_id")
	//todo redis
	t, err := task.GetByID(c.Request.Context(), api.db, taskID)
	if err != nil { /* ... error handling ... */
		return
	}

	//If task is completed and logs are on COS, generate pre-signed URLs for log files
	if t.Status == task.StatusCompleted {
		logLinks := make(map[string]string)
		for _, logFile := range []string{"diff.patch", "agent.log"} {
			s3Key := fmt.Sprintf("logs/%s/%s", taskID, logFile)
			url, err := api.cosClient.GetPresignedURL(context.Background(), "your-logs-bucket", s3Key, time.Minute*15)
			if err == nil {
				logLinks[logFile] = url
			}
		}
		t.LogFileURLs = logLinks // Add this field to task.Definition
	}
	c.JSON(http.StatusOK, t)
}

func (api *TaskAPI) HandleGetLogFile(c *gin.Context) {
	taskID := c.Param("task_id")
	filename := c.Param("filename")

	// 日志文件存储在 COS，可以生成预签名URL返回，或直接代理下载
	// 生成 COS 预签名URL
	s3Key := fmt.Sprintf("logs/%s/%s", taskID, filename)
	logBucket := "codex-logs" // 与 config.yaml 保持一致，或通过依赖注入传入
	url, err := api.cosClient.GetPresignedURL(c.Request.Context(), logBucket, s3Key, time.Minute*15)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get log file URL", "details": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"url": url})
}
