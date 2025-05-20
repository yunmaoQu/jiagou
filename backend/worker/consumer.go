package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/Shopify/sarama"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
	"github.com/tencentyun/cos-go-sdk-v5"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// TaskConsumer 任务消费者
type TaskConsumer struct {
	kafkaConsumer sarama.Consumer
	cosClient     *cos.Client
	ctx           context.Context
	cancel        context.CancelFunc
}

// NewTaskConsumer 创建新的任务消费者
func NewTaskConsumer(kafkaBrokers []string, cosClient *cos.Client) (*TaskConsumer, error) {
	config := sarama.NewConfig()
	config.Consumer.Return.Errors = true
	config.Consumer.Offsets.Initial = sarama.OffsetOldest

	consumer, err := sarama.NewConsumer(kafkaBrokers, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kafka consumer: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	return &TaskConsumer{
		kafkaConsumer: consumer,
		cosClient:     cosClient,
		ctx:           ctx,
		cancel:        cancel,
	}, nil
}

// Start 开始消费任务
func (c *TaskConsumer) Start() error {
	// 订阅任务主题
	partitionConsumer, err := c.kafkaConsumer.ConsumePartition("codex-tasks", 0, sarama.OffsetNewest)
	if err != nil {
		return fmt.Errorf("failed to start partition consumer: %w", err)
	}

	// 处理消息
	go func() {
		for {
			select {
			case msg := <-partitionConsumer.Messages():
				if err := c.handleMessage(msg); err != nil {
					log.Printf("Error processing message: %v", err)
				}
			case err := <-partitionConsumer.Errors():
				log.Printf("Consumer error: %v", err)
			case <-c.ctx.Done():
				return
			}
		}
	}()

	return nil
}

// handleMessage 处理消息
func (c *TaskConsumer) handleMessage(msg *sarama.ConsumerMessage) error {
	var task struct {
		TaskID       string `json:"task_id"`
		CodeLocation string `json:"code_location"`
		TargetFile   string `json:"target_file"`
		Description  string `json:"description"`
		GitHubToken  string `json:"github_token,omitempty"`
		APIKeySource string `json:"api_key_source"`
	}

	if err := json.Unmarshal(msg.Value, &task); err != nil {
		return fmt.Errorf("failed to unmarshal task message: %w", err)
	}

	// 1. 下载代码到本地
	if err := c.downloadCode(task.CodeLocation); err != nil {
		return fmt.Errorf("failed to download code: %w", err)
	}

	// 2. 创建容器运行 Agent
	if err := c.runAgent(task); err != nil {
		return fmt.Errorf("failed to run agent: %w", err)
	}

	return nil
}

// downloadCode 下载代码
func (c *TaskConsumer) downloadCode(codeLocation string) error {
	// 创建临时目录用于存放代码
	tmpDir, err := os.MkdirTemp("", "codex-code-*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}

	// 判断代码位置类型
	if strings.HasPrefix(codeLocation, "git://") || strings.HasPrefix(codeLocation, "https://") {
		// Git 仓库克隆
		cmd := exec.Command("git", "clone", "--depth=1", codeLocation, tmpDir)
		output, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("git clone failed: %w, output: %s", err, string(output))
		}
	} else if strings.HasPrefix(codeLocation, "cos://") {
		// 从 COS 下载 ZIP 文件
		// 解析 COS 路径
		cosPath := strings.TrimPrefix(codeLocation, "cos://")

		// 下载到临时文件
		zipFile, err := os.CreateTemp("", "codex-*.zip")
		if err != nil {
			return fmt.Errorf("failed to create temp file: %w", err)
		}
		defer os.Remove(zipFile.Name())
		defer zipFile.Close()

		// todo 使用 COS 客户端下载文件,注意：这里简化了实现，实际需要根据 COS SDK 调整
		opt := &cos.ObjectGetOptions{}
		resp, err := c.cosClient.Object.Get(context.Background(), cosPath, opt)
		if err != nil {
			return fmt.Errorf("failed to download from COS: %w", err)
		}
		defer resp.Body.Close()

		// 写入临时文件
		if _, err := io.Copy(zipFile, resp.Body); err != nil {
			return fmt.Errorf("failed to write zip file: %w", err)
		}

		// 解压文件
		cmd := exec.Command("unzip", "-o", zipFile.Name(), "-d", tmpDir)
		output, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("unzip failed: %w, output: %s", err, string(output))
		}
	} else {
		return fmt.Errorf("unsupported code location format: %s", codeLocation)
	}

	return nil
}

// runAgent 运行 Agent
func (c *TaskConsumer) runAgent(task struct {
	TaskID       string `json:"task_id"`
	CodeLocation string `json:"code_location"`
	TargetFile   string `json:"target_file"`
	Description  string `json:"description"`
	GitHubToken  string `json:"github_token,omitempty"`
	APIKeySource string `json:"api_key_source"`
}) error {
	// 获取 Docker 客户端
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return fmt.Errorf("failed to create docker client: %w", err)
	}

	// 创建容器配置
	containerConfig := &container.Config{
		Image: "codex-agent:latest", // 使用预先构建的 Agent 镜像
		Cmd:   []string{"./agent", "--task", task.Description, "--target", task.TargetFile},
		Env: []string{
			"OPENAI_API_KEY=" + os.Getenv("OPENAI_API_KEY"),
			"GITHUB_TOKEN=" + task.GitHubToken,
			"TASK_ID=" + task.TaskID,
		},
	}

	// 获取代码目录的绝对路径
	codeDir, err := filepath.Abs(filepath.Join(os.TempDir(), "codex-code-"+task.TaskID))
	if err != nil {
		return fmt.Errorf("failed to get absolute path: %w", err)
	}

	// 创建输出目录
	outputDir := filepath.Join(os.TempDir(), "codex-output-"+task.TaskID)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// 主机配置（挂载卷）
	hostConfig := &container.HostConfig{
		Binds: []string{
			codeDir + ":/app/code",     // 挂载代码目录
			outputDir + ":/app/output", // 挂载输出目录
		},
		// 可选：限制网络访问
		NetworkMode: "none", // 执行阶段断网，安全隔离
	}

	// 创建容器
	resp, err := cli.ContainerCreate(context.Background(), containerConfig, hostConfig, nil, nil, "")
	if err != nil {
		return fmt.Errorf("failed to create container: %w", err)
	}

	// 启动容器
	if err := cli.ContainerStart(context.Background(), resp.ID, types.ContainerStartOptions{}); err != nil {
		return fmt.Errorf("failed to start container: %w", err)
	}

	// 等待容器完成
	statusCh, errCh := cli.ContainerWait(context.Background(), resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			return fmt.Errorf("error waiting for container: %w", err)
		}
	case <-statusCh:
		// 容器执行完成
	}

	// 获取容器日志
	options := types.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
	}
	logs, err := cli.ContainerLogs(context.Background(), resp.ID, options)
	if err != nil {
		return fmt.Errorf("failed to get container logs: %w", err)
	}
	defer logs.Close()

	// 将日志写入文件
	logFile, err := os.Create(filepath.Join(outputDir, "agent.log"))
	if err != nil {
		return fmt.Errorf("failed to create log file: %w", err)
	}
	defer logFile.Close()

	if _, err := io.Copy(logFile, logs); err != nil {
		return fmt.Errorf("failed to write logs: %w", err)
	}

	// 上传结果到 COS 或其他存储
	// TODO: 实现结果上传逻辑

	// 清理容器
	removeOptions := types.ContainerRemoveOptions{
		RemoveVolumes: true,
		Force:         true,
	}
	if err := cli.ContainerRemove(context.Background(), resp.ID, removeOptions); err != nil {
		return fmt.Errorf("failed to remove container: %w", err)
	}

	return nil
}

// Stop 停止消费者
func (c *TaskConsumer) Stop() error {
	c.cancel()
	if err := c.kafkaConsumer.Close(); err != nil {
		return fmt.Errorf("failed to close kafka consumer: %w", err)
	}
	return nil
}
