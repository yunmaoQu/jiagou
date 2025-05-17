
```python
# Mapping of tool names to functions
AVAILABLE_TOOLS = {
    "run_shell_command": tool_run_shell_command,
    "read_file_content": tool_read_file,
    "edit_file_content": ,
    "web_search_doc_and_reference":,
    "call_api":,
    "",
}

# OpenAI function schema for the tools
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Executes a shell command in the specified directory (relative to /app/code) and returns its stdout, stderr, and exit code. Use for linters, tests, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command_string": {
                        "type": "string",
                        "description": "The shell command to execute. E.g., 'pylint main.py' or 'npm test'.",
                    },
                    "target_directory": {
                        "type": "string",
                        "description": "The directory (relative to /app/code) in which to run the command. Defaults to '.' (code root). E.g., 'src' or 'tests'.",
                        "default": "."
                    }
                },
                "required": ["command_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_content",
            "description": "Reads the content of a specified file (relative to /app/code).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file from the root of the codebase (/app/code). E.g., 'src/utils.py'.",
                    }
                },
                "required": ["file_path"],
            },
        },
    }
]
# --- End Tool Definitions ---


def get_llm_response_with_tool_calls(messages_history):
    logging.info(f"Sending request to OpenAI API with {len(messages_history)} messages (tools enabled)...")
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_history,
            tools=TOOLS_SCHEMA,
            tool_choice="auto" # Let the model decide if it wants to use a tool
        )
        return response.choices[0] # Return the full choice object
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None
```
好的，这是一个非常宏大的目标，涉及到将我们当前的单体应用（尽管有 goroutine 模拟队列）重构成一个**完全分布式、可伸缩、生产级的云原生系统**。

实现所有这些组件（Kafka, 独立 Worker 服务, MySQL+Redis, Kubernetes, COS）的完整代码会非常庞大，并且会依赖具体的云提供商（如腾讯云 COS）或自建基础设施。

**我将为您提供一个架构蓝图和关键代码片段的示例，重点展示如何将这些组件集成进来，以及代码结构会如何演变。** 这将是一个高级别的设计和部分实现，而不是一个可以直接运行的完整项目。

---

## 🚀 架构蓝图：分布式 Codex-like 系统 🚀

```mermaid
graph LR
    subgraph "用户侧"
        UserInterface[前端 UI / API 客户端]
    end

    subgraph "API 层 (可水平扩展)"
        APIGateway[API 网关 (Nginx)]
        APIService1[Codex API 服务 1 (Go)]
        APIService2[Codex API 服务 2 (Go)]
        APIServiceN[Codex API 服务 N (Go)]
    end

    subgraph "数据持久化层"
        MySQLDB[(MySQL - 任务元数据)]
        RedisCache[(Redis - 任务状态缓存/锁)]
    end

    subgraph "消息队列"
        KafkaCluster[Kafka 集群]
        TaskTopic[任务主题 (e.g., codex-tasks)]
        ResultTopic[结果主题 (e.g., codex-results)]
    end

    subgraph "Worker 层 (Kubernetes 管理)"
        K8sCluster[Kubernetes 集群]
        WorkerDeployment[Worker 服务 Deployment (Go)]
        WorkerPod1[Worker Pod 1] --> AgentContainer1[Agent Docker 容器]
        WorkerPod2[Worker Pod 2] --> AgentContainer2[Agent Docker 容器]
        WorkerPodN[Worker Pod N] --> AgentContainerN[Agent Docker 容器]
    end

    subgraph "分布式存储"
        COS[腾讯云 COS]
        COSCodeBucket[代码存储桶 (repos)]
        COSLogsBucket[日志存储桶 (logs)]
    end

    UserInterface --> APIGateway
    APIGateway --> APIService1
    APIGateway --> APIService2
    APIGateway --> APIServiceN

    APIService1 --> MySQLDB
    APIService1 --> RedisCache
    APIService1 -- 发布任务 --> TaskTopic

    APIService2 --> MySQLDB
    APIService2 --> RedisCache
    APIService2 -- 发布任务 --> TaskTopic

    APIServiceN --> MySQLDB
    APIServiceN --> RedisCache
    APIServiceN -- 发布任务 --> TaskTopic

    TaskTopic -- 消费任务 --> WorkerPod1
    TaskTopic -- 消费任务 --> WorkerPod2
    TaskTopic -- 消费任务 --> WorkerPodN

    WorkerPod1 -- 操作 --> COS
    WorkerPod2 -- 操作 --> COS
    WorkerPodN -- 操作 --> COS

    WorkerPod1 -- 更新状态 --> MySQLDB
    WorkerPod1 -- 更新状态 --> RedisCache
    WorkerPod1 -- 发布结果 --> ResultTopic

    ResultTopic -- (可选) API 服务订阅 --> APIService1

    %% K8s 管理 Agent 容器的细节
    %% WorkerPod1 -.-> DockerSpawnerK8s[K8s Docker Spawner]
    %% DockerSpawnerK8s -.-> AgentContainer1
```

**核心流程变化：**

1.  **任务创建 (API 服务):**
    *   用户通过 API 网关向任一 `APIService` 实例提交任务。
    *   `APIService`:
        *   验证请求。
        *   生成任务 ID。
        *   将任务元数据（不包括代码本身）存入 `MySQLDB` (状态：`PENDING` 或 `QUEUED`)。
        *   如果用户上传的是 ZIP 文件，`APIService` 将其**直接上传到 `COSCodeBucket`** 中的一个临时位置 (e.g., `tmp_zips/<task_id>/code.zip`)。
        *   将包含任务 ID、COS 上的代码位置（Git URL 或 ZIP 文件在 COS 上的路径）、任务描述等信息的**消息发布到 Kafka 的 `TaskTopic`**。
        *   （可选）在 `RedisCache` 中设置任务的初始状态。
        *   向用户返回任务 ID 和一个轮询状态的端点。

2.  **任务处理 (Worker 服务 - K8s Pods):**
    *   `WorkerDeployment` 在 Kubernetes 中运行多个 `WorkerPod` 实例。
    *   每个 `WorkerPod` (Go 程序) 都是 Kafka `TaskTopic` 的消费者。
    *   当 `WorkerPod` 收到一个任务消息：
        *   更新 `MySQLDB` 和 `RedisCache` 中的任务状态为 `PROCESSING` (或更细化的状态，如 `DOWNLOADING_CODE`)。
        *   **代码获取：**
            *   如果任务是 Git URL，Worker 直接在 Pod 内或临时 Volume 中 `git clone`。
            *   如果任务是 ZIP，Worker 从 `COSCodeBucket` (e.g., `tmp_zips/<task_id>/code.zip`) **下载 ZIP 文件**到 Pod 内或临时 Volume，然后解压。
        *   **准备 Agent 容器的输入：**
            *   将处理后的代码（克隆或解压后）**上传到 `COSCodeBucket`** 的一个任务专属路径 (e.g., `processed_code/<task_id>/`)。Worker 需要确保 Agent 容器能够访问这些代码。
        *   **启动 Agent 容器 (通过 Kubernetes API):**
            *   Worker 不再直接调用 Docker SDK。它会**创建一个 Kubernetes `Job` 或 `Pod`**。
            *   这个 K8s `Job/Pod` 的定义会包含：
                *   Agent Docker 镜像 (`codex-agent:latest`)。
                *   环境变量 (API Keys, 任务参数)。
                *   **Volume 挂载：**
                    *   **代码输入：** 使用 CSI (Container Storage Interface)驱动程序直接从 `COSCodeBucket` 挂载代码到容器的 `/app/code` (例如，使用 `goofys` 或腾讯云的 `cosfs` CSI 插件)，或者 Worker 先下载代码到 K8s `emptyDir` 或 `PersistentVolumeClaim`，再挂载给 Agent Pod。后者更常见，因为 Agent 可能需要写权限。
                    *   **日志输出：** 挂载一个 `emptyDir` 或 `PVC` 到容器的 `/app/output`。
        *   Worker 监控 K8s `Job/Pod` 的状态。

3.  **Agent 执行 (K8s 内的容器):**
    *   Agent 容器如常运行，读取 `/app/code` 中的代码，执行任务。
    *   所有输出 (logs, diff, prompt.txt 等) 写入到 `/app/output`。

4.  **结果收集与状态更新 (Worker 服务):**
    *   当 Agent K8s `Job/Pod` 完成后：
        *   Worker 从 `/app/output` 挂载的 Volume 中收集所有日志和结果文件。
        *   将这些文件**上传到 `COSLogsBucket`** (e.g., `logs/<task_id>/diff.patch`)。
        *   更新 `MySQLDB` 和 `RedisCache` 中的任务状态为 `COMPLETED` 或 `FAILED`，并存储 COS 上日志文件的链接或PR链接。
        *   （可选）将任务完成的简要信息（如任务 ID、状态、结果摘要的 COS 路径）发布到 Kafka 的 `ResultTopic`。API 服务或其他下游服务可以订阅此主题。

5.  **用户获取结果 (API 服务):**
    *   用户轮询 API 服务的状态接口。
    *   `APIService` 从 `RedisCache` (快速路径) 或 `MySQLDB` (持久路径) 获取任务状态。
    *   如果任务完成，API 服务返回指向 `COSLogsBucket` 中结果文件的**预签名 URL** 或通过 API 代理下载这些文件。

---

## 关键代码结构与示例片段

### 1. `backend/` (API 服务 - Go)

#### `backend/cmd/api/main.go` (简化)

```go
package main

import (
        "codex-sys/backend/internal/api"
        "codex-sys/backend/internal/config"
        "codex-sys/backend/internal/platform/database"
        "codex-sys/backend/internal/platform/kafka"
        "codex-sys/backend/internal/platform/objectstorage"
        "log"

        "github.com/gin-gonic/gin"
        "github.com/jmoiron/sqlx" // For MySQL
        // "github.com/go-redis/redis/v8" // For Redis
    // "github.com/segmentio/kafka-go" // For Kafka
)

func main() {
        cfg := config.Load() // Load from .env or config file

        // --- Initialize Platforms ---
        db, err := database.NewMySQLConnection(cfg.MySQLDSN)
        if err != nil {
                log.Fatalf("Failed to connect to MySQL: %v", err)
        }
        defer db.Close()

        // redisClient := database.NewRedisClient(cfg.RedisAddr)
        // defer redisClient.Close()

        kafkaProducer, err := kafka.NewProducer(cfg.KafkaBrokers, cfg.KafkaTaskTopic)
        if err != nil {
                log.Fatalf("Failed to create Kafka producer: %v", err)
        }
        defer kafkaProducer.Close()

        cosClient, err := objectstorage.NewCOSClient(cfg.COSConfig) // COS/S3 client
        if err != nil {
                log.Fatalf("Failed to create COS client: %v", err)
        }

        // --- Setup Router & Handlers ---
        router := gin.Default()
        // Pass DB, Kafka Producer, COS Client to handlers
        api.RegisterRoutes(router, db, kafkaProducer, cosClient /*, redisClient */)

        log.Printf("API Server starting on port %s", cfg.APIPort)
        if err := router.Run(":" + cfg.APIPort); err != nil {
                log.Fatalf("Failed to run API server: %v", err)
        }
}
```

#### `backend/internal/api/task_handlers.go` (简化)

```go
package api

import (
        "codex-sys/backend/internal/task" // Task domain logic
        "codex-sys/backend/internal/platform/kafka"
        "codex-sys/backend/internal/platform/objectstorage"
        "context"
        "fmt"
        "log"
        "net/http"
        "path/filepath"
        "time"

        "github.com/gin-gonic/gin"
        "github.com/google/uuid"
        "github.com/jmoiron/sqlx"
)

type TaskAPI struct {
        db            *sqlx.DB
        kafkaProducer *kafka.Producer // Assuming a wrapper for kafka-go
        cosClient     *objectstorage.ClientWrapper // Assuming a wrapper for COS SDK
        // redisClient *redis.Client
}

func RegisterRoutes(router *gin.Engine, db *sqlx.DB, kp *kafka.Producer, cosClient *objectstorage.ClientWrapper) {
        api := &TaskAPI{db: db, kafkaProducer: kp, cosClient: cosClient}
        // ... routes ...
        router.POST("/api/task", api.HandleCreateTask)
        router.GET("/api/task/:task_id/status", api.HandleGetTaskStatus)
}


func (api *TaskAPI) HandleCreateTask(c *gin.Context) {
        // ... (parse form data as before: git_url, task_description, target_file, etc.) ...
        gitURL := c.PostForm("git_url")
        taskDescription := c.PostForm("task_description")
        targetFile := c.PostForm("target_file")
    // ... (validation) ...

        taskID := uuid.NewString()
        newTask := task.Definition{ // task.Definition is your struct for task metadata
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
                if errUpload != nil { /* ... error handling ... */ return }
                defer zipFile.Close()

                newTask.InputType = task.InputZip
                newTask.ZipFileName = zipFileHeader.Filename
                
                // Upload ZIP to COS
                s3Key := fmt.Sprintf("tmp_zips/%s/%s", taskID, zipFileHeader.Filename)
                err := api.cosClient.UploadFile(context.Background(), "your-code-bucket", s3Key, zipFile, zipFileHeader.Size)
                if err != nil {
                        log.Printf("Failed to upload ZIP to COS for task %s: %v", taskID, err)
                        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to upload code to storage"})
                        return
                }
                codeLocationS3Path = fmt.Sprintf("s3://your-code-bucket/%s", s3Key) // Or just the key
                newTask.CodePathCOS = codeLocationS3Path
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
                OpenAIAPIKeySource: "env_or_secret_manager", // Indicate where worker should get it
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
    // Fetch from Redis first (cache)
    // If not in Redis, fetch from MySQL
    // For simplicity, directly from MySQL:
    t, err := task.GetByID(c.Request.Context(), api.db, taskID)
    if err != nil { /* ... error handling ... */ return }

    // If task is completed and logs are on COS, generate pre-signed URLs for log files
    // if t.Status == task.StatusCompleted {
    //    logLinks := make(map[string]string)
    //    for _, logFile := range []string{"diff.patch", "agent.log"} {
    //        s3Key := fmt.Sprintf("logs/%s/%s", taskID, logFile)
    //        url, err := api.cosClient.GetPresignedURL(context.Background(), "your-logs-bucket", s3Key, time.Minute*15)
    //        if err == nil { logLinks[logFile] = url }
    //    }
    //    t.LogFileURLs = logLinks // Add this field to task.Definition
    // }
    c.JSON(http.StatusOK, t)
}
```

#### `backend/internal/task/task_model.go` (MySQL interaction)

```go
package task

import (
        "context"
        "time"
        "github.com/jmoiron/sqlx"
        _ "github.com/go-sql-driver/mysql" // MySQL driver
)

type Status string
type InputType string

const (
        StatusPending   Status = "PENDING"
        StatusQueued    Status = "QUEUED"
        StatusProcessing Status = "PROCESSING"
        StatusDownloadingCode Status = "DOWNLOADING_CODE"
        StatusRunningAgent Status = "RUNNING_AGENT"
        StatusUploadingResults Status = "UPLOADING_RESULTS"
        StatusCompleted Status = "COMPLETED"
        StatusFailed    Status = "FAILED"

        InputGit InputType = "GIT"
        InputZip InputType = "ZIP"
)

type Definition struct {
        ID              string    `db:"id" json:"id"`
        InputType       InputType `db:"input_type" json:"input_type"`
        GitURL          string    `db:"git_url,omitempty" json:"git_url,omitempty"`
        ZipFileName     string    `db:"zip_file_name,omitempty" json:"zip_file_name,omitempty"`
        CodePathCOS     string    `db:"code_path_cos,omitempty" json:"code_path_cos,omitempty"` // Path to code on COS
        TargetFile      string    `db:"target_file" json:"target_file"`
        TaskDescription string    `db:"task_description" json:"task_description"`
        Status          Status    `db:"status" json:"status"`
        Message         string    `db:"message,omitempty" json:"message,omitempty"`
        GitHubToken     string    `db:"github_token,omitempty" json:"-"` // Don't store in DB directly, manage via secrets
        PRURL           string    `db:"pr_url,omitempty" json:"pr_url,omitempty"`
        CreatedAt       time.Time `db:"created_at" json:"created_at"`
        UpdatedAt       time.Time `db:"updated_at" json:"updated_at"`
        LogFileURLs     map[string]string `json:"log_file_urls,omitempty"` // For API response, not DB field
}

type KafkaTaskMessage struct { // Message sent to Kafka
        TaskID             string `json:"task_id"`
        InputType          string `json:"input_type"`      // "GIT" or "ZIP"
        CodeLocation       string `json:"code_location"`   // Git URL or S3 Key for ZIP
        TargetFile         string `json:"target_file"`
        TaskDescription    string `json:"task_description"`
        UserGitHubToken    string `json:"user_github_token,omitempty"`
        OpenAIAPIKeySource string `json:"openai_api_key_source"` // e.g., "k8s_secret_ref_name"
}


func Create(ctx context.Context, db *sqlx.DB, t *Definition) error {
        query := `INSERT INTO tasks (id, input_type, git_url, zip_file_name, code_path_cos, target_file, task_description, status, created_at, updated_at)
                  VALUES (:id, :input_type, :git_url, :zip_file_name, :code_path_cos, :target_file, :task_description, :status, :created_at, :updated_at)`
        _, err := db.NamedExecContext(ctx, query, t)
        return err
}

func GetByID(ctx context.Context, db *sqlx.DB, id string) (*Definition, error) {
        var t Definition
        query := `SELECT * FROM tasks WHERE id = ?`
        err := db.GetContext(ctx, &t, query, id)
        return &t, err
}

func UpdateStatus(ctx context.Context, db *sqlx.DB, id string, status Status, message string) error {
        query := `UPDATE tasks SET status = ?, message = ?, updated_at = ? WHERE id = ?`
        _, err := db.ExecContext(ctx, query, status, message, time.Now(), id)
        return err
}
// ... other CRUD operations for tasks ...
```
**MySQL Table Schema (tasks):**
```sql
CREATE TABLE tasks (
    id VARCHAR(36) PRIMARY KEY,
    input_type VARCHAR(10) NOT NULL,
    git_url VARCHAR(512),
    zip_file_name VARCHAR(255),
    code_path_cos VARCHAR(1024), -- S3 key or full path for the initial code
    target_file VARCHAR(512) NOT NULL,
    task_description TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,
    message TEXT,
    pr_url VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

#### `backend/internal/platform/kafka/producer.go` (Kafka Producer Wrapper)
```go
package kafka

import (
        "context"
        "encoding/json"
        "log"

        "github.com/segmentio/kafka-go"
)

type Producer struct {
        writer *kafka.Writer
        topic  string
}

func NewProducer(brokers []string, topic string) (*Producer, error) {
        writer := &kafka.Writer{
                Addr:     kafka.TCP(brokers...),
                Topic:    topic,
                Balancer: &kafka.LeastBytes{},
                // RequiredAcks: kafka.RequireAll, // For higher durability
                // Async: true, // For higher throughput
        }
        return &Producer{writer: writer, topic: topic}, nil
}

func (p *Producer) Publish(ctx context.Context, key string, value interface{}) error {
        msgBytes, err := json.Marshal(value)
        if err != nil {
                return fmt.Errorf("failed to marshal kafka message: %w", err)
        }

        err = p.writer.WriteMessages(ctx, kafka.Message{
                Key:   []byte(key),
                Value: msgBytes,
        })
        if err != nil {
                return fmt.Errorf("failed to write kafka message: %w", err)
        }
        log.Printf("Published message to Kafka topic %s with key %s", p.topic, key)
        return nil
}

func (p *Producer) Close() error {
        return p.writer.Close()
}
```

#### `backend/internal/platform/objectstorage/cos.go` (COS Wrapper)
```go
package objectstorage

import (
        "context"
        "fmt"
        "io"
        "net/http"
        "net/url"
        "time"

        "github.com/tencentyun/cos-go-sdk-v5" // Tencent Cloud COS SDK
)
// This is a simplified wrapper. You'd need to configure credentials etc.
// For S3, you'd use aws-sdk-go. For MinIO, also aws-sdk-go compatible.

type ClientWrapper struct {
        client *cos.Client
}

type COSConfig struct { // Example config structure
        SecretID  string
        SecretKey string
        Region    string
        BucketURL string // e.g., https://mybucket-1250000000.cos.ap-guangzhou.myqcloud.com
}

func NewCOSClient(cfg COSConfig) (*ClientWrapper, error) {
        u, _ := url.Parse(cfg.BucketURL)
        b := &cos.BaseURL{BucketURL: u}
        client := cos.NewClient(b, &http.Client{
                Transport: &cos.AuthorizationTransport{
                        SecretID:  cfg.SecretID,
                        SecretKey: cfg.SecretKey,
                },
        })
        return &ClientWrapper{client: client}, nil
}

func (cw *ClientWrapper) UploadFile(ctx context.Context, bucketName string, key string, reader io.Reader, size int64) error {
        // bucketName might be part of the BaseURL or specified here depending on SDK usage
        opt := &cos.ObjectPutOptions{
                ObjectPutHeaderOptions: &cos.ObjectPutHeaderOptions{
                        ContentLength: size,
                },
        }
        _, err := cw.client.Object.Put(ctx, key, reader, opt)
        if err != nil {
                return fmt.Errorf("COS Put failed for key %s: %w", key, err)
        }
        return nil
}

func (cw *ClientWrapper) DownloadFile(ctx context.Context, bucketName string, key string, writer io.Writer) error {
        resp, err := cw.client.Object.Get(ctx, key, nil)
        if err != nil {
                return fmt.Errorf("COS Get failed for key %s: %w", key, err)
        }
        defer resp.Body.Close()
        if resp.StatusCode >= 300 {
                return fmt.Errorf("COS Get failed for key %s with status %d", key, resp.StatusCode)
        }
        _, err = io.Copy(writer, resp.Body)
        return err
}

func (cw *ClientWrapper) GetPresignedURL(ctx context.Context, bucketName string, key string, duration time.Duration) (string, error) {
        presignedURL, err := cw.client.Object.GetPresignedURL(ctx, http.MethodGet, key, cw.client.Conf.SecretID, cw.client.Conf.SecretKey, duration, nil)
        if err != nil {
                return "", err
        }
        return presignedURL.String(), nil
}
```

---

### 2. `worker/` (Worker 服务 - Go)

This would be a separate Go application, built into a Docker image, and deployed on Kubernetes.

#### `worker/cmd/main.go`

```go
package main

import (
        "codex-sys/worker/internal/config"
        "codex-sys/worker/internal/consumer"
        "codex-sys/worker/internal/handler"
        "codex-sys/worker/internal/platform/database"
        "codex-sys/worker/internal/platform/k8s"
        "codex-sys/worker/internal/platform/objectstorage"
        "log"
        "os"
        "os/signal"
        "syscall"
        "context"
)

func main() {
        cfg := config.LoadWorkerConfig()

        db, err := database.NewMySQLConnection(cfg.MySQLDSN)
        if err != nil { log.Fatalf("Worker: Failed to connect to MySQL: %v", err) }
        defer db.Close()

        cosClient, err := objectstorage.NewCOSClient(cfg.COSConfig)
        if err != nil { log.Fatalf("Worker: Failed to create COS client: %v", err) }

        k8sClient, err := k8s.NewK8sClient() // In-cluster or from kubeconfig
        if err != nil { log.Fatalf("Worker: Failed to create Kubernetes client: %v", err) }

        taskHandler := handler.NewTaskHandler(db, cosClient, k8sClient, cfg)

        // Setup Kafka consumer
        kafkaConsumer, err := consumer.NewConsumer(cfg.KafkaBrokers, cfg.KafkaTaskTopic, "codex-worker-group", taskHandler.Handle)
        if err != nil {
                log.Fatalf("Worker: Failed to create Kafka consumer: %v", err)
        }
        defer kafkaConsumer.Close()

        log.Println("Worker service started. Waiting for tasks...")

        // Graceful shutdown
        sigterm := make(chan os.Signal, 1)
        signal.Notify(sigterm, syscall.SIGINT, syscall.SIGTERM)
        
        ctx, cancel := context.WithCancel(context.Background())
        go func() {
                <-sigterm
                log.Println("Worker: Termination signal received. Shutting down...")
                cancel() // Signal consumer to stop
        }()

        if err := kafkaConsumer.Run(ctx); err != nil { // Run blocks until context is cancelled or error
        log.Printf("Worker: Kafka consumer exited with error: %v", err)
    }

        log.Println("Worker service stopped.")
}
```

#### `worker/internal/handler/task_handler.go`

```go
package handler

import (
        "codex-sys/worker/internal/config"
        "codex-sys/worker/internal/platform/database" // Your DB interaction logic for worker
        "codex-sys/worker/internal/platform/k8s"
        "codex-sys/worker/internal/platform/objectstorage"
        "codex-sys/worker/internal/taskdef" // Shared task definition struct
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
        db        *database.DBClientWrapper // Wrapper for sqlx.DB
        cos       *objectstorage.ClientWrapper
        k8s       *k8s.Client
        cfg       config.WorkerConfig
}

func NewTaskHandler(db *database.DBClientWrapper, cos *objectstorage.ClientWrapper, k8s *k8s.Client, cfg config.WorkerConfig) *TaskHandler {
        return &TaskHandler{db: db, cos: cos, k8s: k8s, cfg: cfg}
}

// Handle is called by the Kafka consumer with a new message
func (h *TaskHandler) Handle(ctx context.Context, key []byte, value []byte) error {
        var msg taskdef.KafkaTaskMessage // Use the shared struct
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
        if err := h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusDownloadingCode, "Preparing code"); err != nil { /* ... */ }
        
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
                if err != nil { /* ... error handling & DB update ... */ return nil }
                
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
        if err := h.db.UpdateTaskStatus(ctx, msg.TaskID, taskdef.StatusRunningAgent, "Launching agent"); err != nil { /* ... */ }
        
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
        "GITHUB_TOKEN": msg.UserGitHubToken, // If provided
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
        Name:            agentJobName,
        Namespace:       h.cfg.K8sAgentNamespace,
        Image:           h.cfg.AgentDockerImage, // e.g., "your-registry/codex-agent:latest"
        Command:         agentCmd,
        EnvVars:         agentEnvVars,
        CodeCOSPath:     fmt.Sprintf("s3://%s/%s", h.cfg.COSCodeBucket, agentCodeCOSPath), // For CSI or initContainer
        OutputCOSPath:   fmt.Sprintf("s3://%s/logs/%s/", h.cfg.COSLogsBucket, msg.TaskID), // For sidecar or post-run upload
        CPULimit:        "1",
        MemoryLimit:     "2Gi",
        ServiceAccount:  h.cfg.K8sAgentServiceAccount, // If agent needs K8s permissions or cloud permissions via IRSA/Workload Identity
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

    finalStatus := taskdef.StatusFailed
    finalMessage := "Agent job finished with unknown status."

    if jobStatus == k8s.JobSucceeded {
        finalStatus = taskdef.StatusCompleted
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
```

#### `worker/internal/platform/k8s/client.go` (Kubernetes Client Wrapper - Highly Simplified)

```go
package k8s

import (
        "context"
        "fmt"
        "log"
        "time"

        batchv1 "k8s.io/api/batch/v1"
        corev1 "k8s.io/api/core/v1"
        "k8s.io/apimachinery/pkg/api/resource"
        metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
        "k8s.io/client-go/kubernetes"
        "k8s.io/client-go/rest"
        // "k8s.io/client-go/tools/clientcmd" // For out-of-cluster config
)

type Client struct {
        clientset *kubernetes.Clientset
}

type AgentJobConfig struct {
    Name            string
    Namespace       string
    Image           string
    Command         []string
    EnvVars         map[string]string
    CodeCOSPath     string // s3://bucket/path/to/code/
    OutputCOSPath   string // s3://bucket/path/to/logs/
    CPULimit        string
    MemoryLimit     string
    ServiceAccount  string
}

const (
    JobSucceeded JobStatus = "Succeeded"
    JobFailed    JobStatus = "Failed"
    JobRunning   JobStatus = "Running"
    JobUnknown   JobStatus = "Unknown"
)
type JobStatus string


func NewK8sClient() (*Client, error) {
        config, err := rest.InClusterConfig()
        if err != nil {
                // Fallback to kubeconfig for local development if needed
                // kubeconfig := clientcmd.RecommendedHomeFile
                // config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
                // if err != nil {
                return nil, fmt.Errorf("failed to get k8s config: %w", err)
                // }
        }
        clientset, err := kubernetes.NewForConfig(config)
        if err != nil {
                return nil, fmt.Errorf("failed to create k8s clientset: %w", err)
        }
        return &Client{clientset: clientset}, nil
}

// RunAgentJob is highly simplified. Real implementation needs robust error handling,
// volume mounting strategies (CSI for COS, or initContainers + emptyDir/PVC),
// and potentially sidecars for log uploading.
func (c *Client) RunAgentJob(ctx context.Context, cfg AgentJobConfig) error {
        log.Printf("K8s: Creating job %s in namespace %s", cfg.Name, cfg.Namespace)

        env := []corev1.EnvVar{}
        for k, v := range cfg.EnvVars {
                env = append(env, corev1.EnvVar{Name: k, Value: v})
        }

    // This is a placeholder for how COS paths would be used.
    // Actual mounting depends on CSI drivers or initContainer logic.
    // Example: an initContainer could use 'aws s3 sync' or 'coscmd'
    // to download from cfg.CodeCOSPath to an emptyDir volume.
    // A sidecar or post-stop lifecycle hook could upload /app/output to cfg.OutputCOSPath.

    tempCodeVolumeName := "agent-code"
    tempOutputVolumeName := "agent-output"

        job := &batchv1.Job{
                ObjectMeta: metav1.ObjectMeta{
                        Name:      cfg.Name,
                        Namespace: cfg.Namespace,
                },
                Spec: batchv1.JobSpec{
                        Template: corev1.PodTemplateSpec{
                                Spec: corev1.PodSpec{
                                        RestartPolicy: corev1.RestartPolicyNever, // Or OnFailure
                    ServiceAccountName: cfg.ServiceAccount, // If specified
                                        Containers: []corev1.Container{
                                                {
                                                        Name:    "codex-agent",
                                                        Image:   cfg.Image,
                                                        Command: []string{"python3", "agent.py"}, // Entrypoint is in Dockerfile
                                                        Args:    cfg.Command,
                                                        Env:     env,
                                                        Resources: corev1.ResourceRequirements{
                                                                Limits: corev1.ResourceList{
                                                                        corev1.ResourceCPU:    resource.MustParse(cfg.CPULimit),
                                                                        corev1.ResourceMemory: resource.MustParse(cfg.MemoryLimit),
                                                                },
                                // Requests should also be set
                                                        },
                            VolumeMounts: []corev1.VolumeMount{
                                {Name: tempCodeVolumeName, MountPath: "/app/code"},
                                {Name: tempOutputVolumeName, MountPath: "/app/output"},
                            },
                                                },
                                        },
                    // This is where you'd define initContainers for downloading code
                    // and potentially sidecars for uploading logs if not handled by the agent itself
                    // or post-job cleanup hooks.
                    InitContainers: []corev1.Container{
                        {
                            Name: "setup-code-volume",
                            Image: "amazon/aws-cli:latest", // Or appropriate COS tool image
                            Command: []string{"sh", "-c"},
                            // Example: aws s3 sync s3://bucket/code/task123 /mnt/code --delete
                            Args: []string{fmt.Sprintf("aws s3 sync %s /mnt/code --delete && echo 'Code downloaded'", cfg.CodeCOSPath)},
                            VolumeMounts: []corev1.VolumeMount{
                                {Name: tempCodeVolumeName, MountPath: "/mnt/code"},
                            },
                            // Ensure this initContainer has IAM permissions for S3/COS
                        },
                    },
                    Volumes: []corev1.Volume{
                        {Name: tempCodeVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
                        {Name: tempOutputVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
                    },
                                },
                        },
                        BackoffLimit:           int32Ptr(1), // Retry once on failure
                        TTLSecondsAfterFinished: int32Ptr(3600), // Auto-cleanup job after 1 hour
                },
        }

        _, err := c.clientset.BatchV1().Jobs(cfg.Namespace).Create(ctx, job, metav1.CreateOptions{})
        if err != nil {
                return fmt.Errorf("failed to create K8s job: %w", err)
        }

    // Simplified: In reality, you'd watch the job or poll its status.
    // This example doesn't wait for completion. The worker would need a separate monitoring loop.
        log.Printf("K8s: Job %s submitted.", cfg.Name)
        return nil
}


func (c *Client) GetJobStatus(ctx context.Context, namespace, jobName string) (JobStatus, error) {
    job, err := c.clientset.BatchV1().Jobs(namespace).Get(ctx, jobName, metav1.GetOptions{})
    if err != nil {
        return JobUnknown, fmt.Errorf("failed to get K8s job %s: %w", jobName, err)
    }

    if job.Status.Succeeded > 0 {
        return JobSucceeded, nil
    }
    if job.Status.Failed > 0 {
        // Could inspect pod logs for more details on failure
        return JobFailed, nil
    }
    if job.Status.Active > 0 {
        return JobRunning, nil
    }
    // Could be pending, or other conditions
    return JobUnknown, nil
}


func int32Ptr(i int32) *int32 { return &i }
```

---

### 3. `agent/` (Agent - Python)

No major changes are strictly *required* in the agent for this backend refactor, as long as:
*   It still receives code in `/app/code`.
*   It can still write logs/outputs to `/app/output`.
*   Environment variables (like `OPENAI_API_KEY`, `GITHUB_TOKEN`) are correctly passed.

**However, to integrate with COS for output uploading from the Agent Pod (if not using a sidecar):**

The Agent Pod's main container (or a post-stop lifecycle hook) would need:
1.  COS credentials (e.g., via K8s secrets mounted as env vars or files, or using Workload Identity/IRSA).
2.  A COS SDK or CLI tool (like `aws s3 sync` or `coscmd`) installed in the agent image.
3.  Logic at the end of `agent.py` (or in a wrapper script) to upload the contents of `/app/output` to the designated `OutputCOSPath` (which would need to be passed as an env var to the agent).

**Example snippet for agent.py to upload output (conceptual):**
```python
# At the end of agent.py's main() function

def upload_output_to_cos(output_dir_path: Path, cos_bucket: str, cos_prefix: str):
    logging.info(f"Attempting to upload contents of {output_dir_path} to COS s3://{cos_bucket}/{cos_prefix}")
    # This requires awscli or coscmd to be in the container and configured
    # For awscli (works with S3 and MinIO, and often Tencent COS with S3 compatibility)
    # Ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION are set as env vars
    # or the pod has an IAM role attached (IRSA/Workload Identity).
    # For Tencent COS native: ensure coscmd is installed and configured.

    # Example using aws s3 sync:
    # target_s3_path = f"s3://{cos_bucket}/{cos_prefix}"
    # command = ["aws", "s3", "sync", str(output_dir_path), target_s3_path, "--delete"]
    
    # Example using coscmd (Tencent Cloud):
    # Needs prior `coscmd config -a key -s secret -b default_bucket -r region` or env vars
    # Or pass credentials directly if supported by coscmd version
    target_cos_path = f"cos://{cos_bucket}/{cos_prefix}" # coscmd uses cos:// schema
    # coscmd upload -r /local/path/ cos_path/
    command = ["coscmd", "upload", "-r", str(output_dir_path) + "/", target_cos_path]


    logging.info(f"Executing COS upload command: {' '.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        logging.info(f"COS Upload STDOUT: {process.stdout}")
        if process.stderr:
            logging.warning(f"COS Upload STDERR: {process.stderr}")
        logging.info("Output successfully uploaded to COS.")
    except subprocess.CalledProcessError as e:
        logging.error(f"COS Upload failed. Exit code: {e.returncode}")
        logging.error(f"COS Upload STDOUT: {e.stdout}")
        logging.error(f"COS Upload STDERR: {e.stderr}")
    except FileNotFoundError:
        logging.error(f"COS upload command (e.g., aws or coscmd) not found. Ensure it's in the Docker image PATH.")


# In agent.py main() before exiting:
# ... (rest of agent logic)
# output_cos_bucket = os.getenv("OUTPUT_COS_BUCKET")
# output_cos_prefix = os.getenv("OUTPUT_COS_PREFIX") # e.g., "logs/task_id/"
# if output_cos_bucket and output_cos_prefix:
#    upload_output_to_cos(OUTPUT_DIR, output_cos_bucket, output_cos_prefix)
# else:
#    logging.warning("COS output bucket/prefix not configured. Skipping upload from agent.")
```
This upload logic is often better handled by a K8s sidecar container or a post-run script defined in the K8s Job spec to keep the agent focused on its core task.

---

### Deployment Considerations:

*   **Configuration:** Each service (API, Worker) will need its own configuration management (env vars, config files, K8s ConfigMaps/Secrets).
*   **Docker Images:** You'll build separate Docker images for the API service, the Worker service, and the Agent.
*   **Kubernetes Manifests:** You'll need K8s `Deployment` YAMLs for the API and Worker services, `Service` YAMLs to expose them, and a way to define the Agent `Job` (perhaps the Worker generates the Job spec dynamically).
*   **Database Setup:** MySQL schema needs to be applied. Redis setup.
*   **Kafka Setup:** Topics (`codex-tasks`, `codex-results`) need to be created.
*   **COS Buckets:** `your-code-bucket` and `your-logs-bucket` need to be created with appropriate permissions.
*   **IAM/Permissions:**
    *   API service might need permission to write to COS (for initial ZIP uploads).
    *   Worker service needs permission to read/write to COS, interact with Kubernetes API (create Jobs, get Job status), and read/write to MySQL/Redis.
    *   Agent Pods (if they handle their own COS interactions via CSI or direct SDK calls) need COS read (for code) and write (for logs) permissions. This is often managed via K8s Service Accounts + IRSA (AWS), Workload Identity (GCP/Azure), or OIDC federation with Tencent Cloud CAM.

---

This is a significant architectural shift. Each component introduced (Kafka, K8s, distributed DBs, object storage) has its own complexities and learning curve. The provided code snippets are illustrative and would need substantial fleshing out, error handling, and testing to be production-ready.

Start by tackling one piece at a time, e.g., introducing Kafka between the API and a simplified worker first, then moving the worker to K8s, then integrating COS, etc. This iterative approach will be more manageable.
