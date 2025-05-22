package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/Shopify/sarama"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
	"github.com/yunmaoQu/codex-sys/internal/platform/database"
	"github.com/yunmaoQu/codex-sys/internal/platform/objectstorage"
	"github.com/yunmaoQu/codex-sys/internal/task"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// ExecutionMode defines how agent tasks will be executed
type ExecutionMode string

const (
	// DockerMode runs agents in local Docker containers
	DockerMode ExecutionMode = "docker"
	// K8sMode runs agents as Kubernetes jobs
	K8sMode ExecutionMode = "kubernetes"
)

// AgentJobConfig contains configuration for an agent job
type AgentJobConfig struct {
	Name           string
	Namespace      string
	Image          string
	Command        []string
	EnvVars        map[string]string
	CodeCOSPath    string // s3://bucket/path/to/code/
	OutputCOSPath  string // s3://bucket/path/to/logs/
	CPULimit       string
	MemoryLimit    string
	ServiceAccount string
}

// JobStatus describes the status of a running job
type JobStatus string

const (
	JobSucceeded JobStatus = "Succeeded"
	JobFailed    JobStatus = "Failed"
	JobRunning   JobStatus = "Running"
	JobUnknown   JobStatus = "Unknown"
)

// Config contains configuration for the worker
type Config struct {
	ExecutionMode      ExecutionMode
	AgentImage         string
	TempDirBase        string
	TaskTopic          string
	K8sNamespace       string
	K8sServiceAccount  string
	CPULimit           string
	MemoryLimit        string
	CleanupTempDirs    bool
	CodeBucket         string
	LogsBucket         string
	EnableGitHubAccess bool
}

// Worker handles task processing and agent execution
type Worker struct {
	cfg           Config
	kafkaConsumer sarama.Consumer
	cosClient     *objectstorage.ClientWrapper
	dockerClient  *client.Client
	k8sClient     *kubernetes.Clientset
	db            *database.DBClientWrapper
	ctx           context.Context
	cancel        context.CancelFunc
}

// NewWorker creates a new Worker instance
func NewWorker(
	cfg Config,
	kafkaBrokers []string,
	cosClient *objectstorage.ClientWrapper,
	db *database.DBClientWrapper,
) (*Worker, error) {
	// Set up Kafka consumer
	kafkaConfig := sarama.NewConfig()
	kafkaConfig.Consumer.Return.Errors = true
	kafkaConfig.Consumer.Offsets.Initial = sarama.OffsetNewest

	consumer, err := sarama.NewConsumer(kafkaBrokers, kafkaConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kafka consumer: %w", err)
	}

	// Create context for the worker
	ctx, cancel := context.WithCancel(context.Background())

	// Create worker instance
	worker := &Worker{
		cfg:           cfg,
		kafkaConsumer: consumer,
		cosClient:     cosClient,
		db:            db,
		ctx:           ctx,
		cancel:        cancel,
	}

	// Set up execution environment based on mode
	if cfg.ExecutionMode == DockerMode {
		worker.dockerClient, err = client.NewClientWithOpts(client.FromEnv)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create docker client: %w", err)
		}
	} else if cfg.ExecutionMode == K8sMode {
		k8sConfig, err := rest.InClusterConfig()
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to get k8s config: %w", err)
		}

		worker.k8sClient, err = kubernetes.NewForConfig(k8sConfig)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create k8s clientset: %w", err)
		}
	} else {
		cancel()
		return nil, fmt.Errorf("unsupported execution mode: %s", cfg.ExecutionMode)
	}

	return worker, nil
}

// Start begins consuming messages and processing tasks
func (w *Worker) Start() error {
	// Subscribe to the task topic
	partitionConsumer, err := w.kafkaConsumer.ConsumePartition(w.cfg.TaskTopic, 0, sarama.OffsetNewest)
	if err != nil {
		return fmt.Errorf("failed to start partition consumer: %w", err)
	}

	log.Printf("Worker started. Listening for tasks on topic: %s", w.cfg.TaskTopic)

	// Process messages
	go func() {
		for {
			select {
			case msg := <-partitionConsumer.Messages():
				log.Printf("Received message: key=%s", string(msg.Key))
				if err := w.handleMessage(msg); err != nil {
					log.Printf("Error processing message: %v", err)
				}
			case err := <-partitionConsumer.Errors():
				log.Printf("Consumer error: %v", err)
			case <-w.ctx.Done():
				log.Println("Worker context cancelled, stopping consumer")
				return
			}
		}
	}()

	return nil
}

// handleMessage processes a Kafka message
func (w *Worker) handleMessage(msg *sarama.ConsumerMessage) error {
	// Parse the task message
	var taskMsg task.KafkaTaskMessage
	if err := json.Unmarshal(msg.Value, &taskMsg); err != nil {
		log.Printf("Failed to unmarshal task message: %v", err)
		return fmt.Errorf("failed to unmarshal task message: %w", err)
	}

	log.Printf("Processing task: %s, Input: %s, Target: %s",
		taskMsg.TaskID, taskMsg.InputType, taskMsg.TargetFile)

	// Update task status in database
	if w.db != nil {
		if err := w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusProcessing, "Worker processing task"); err != nil {
			log.Printf("Failed to update task status: %v", err)
			// Continue processing despite DB error
		}
	}

	// Set up workspace for task
	workDir, err := w.setupWorkspace(taskMsg.TaskID)
	if err != nil {
		w.updateTaskFailure(taskMsg.TaskID, fmt.Sprintf("Failed to set up workspace: %v", err))
		return err
	}

	// Clean up workspace when done if configured
	if w.cfg.CleanupTempDirs {
		defer os.RemoveAll(workDir)
	}

	// Download and prepare code
	codeDir, err := w.prepareCode(taskMsg, workDir)
	if err != nil {
		w.updateTaskFailure(taskMsg.TaskID, fmt.Sprintf("Failed to prepare code: %v", err))
		return err
	}

	// Execute the agent using the appropriate mode
	if err := w.executeAgent(taskMsg, codeDir); err != nil {
		w.updateTaskFailure(taskMsg.TaskID, fmt.Sprintf("Agent execution failed: %v", err))
		return err
	}

	return nil
}

// setupWorkspace creates a workspace directory for the task
func (w *Worker) setupWorkspace(taskID string) (string, error) {
	// Create task-specific directory
	baseDir := w.cfg.TempDirBase
	if baseDir == "" {
		baseDir = os.TempDir()
	}

	workDir := filepath.Join(baseDir, fmt.Sprintf("codex-task-%s", taskID))
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create workspace directory: %w", err)
	}

	return workDir, nil
}

// prepareCode downloads and prepares code for agent execution
func (w *Worker) prepareCode(taskMsg task.KafkaTaskMessage, workDir string) (string, error) {
	// Update status to downloading code
	if w.db != nil {
		w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusDownloadingCode, "Downloading code")
	}

	codeDir := filepath.Join(workDir, "code")
	if err := os.MkdirAll(codeDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create code directory: %w", err)
	}

	// Handle different input types
	switch taskMsg.InputType {
	case string(task.InputGit):
		// Clone git repo
		log.Printf("Cloning Git repo: %s", taskMsg.CodeLocation)
		cmd := exec.Command("git", "clone", "--depth=1", taskMsg.CodeLocation, codeDir)
		output, err := cmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("git clone failed: %v, output: %s", err, string(output))
		}

	case string(task.InputZip):
		// Download from object storage
		log.Printf("Downloading ZIP from storage: %s", taskMsg.CodeLocation)
		zipPath := filepath.Join(workDir, "code.zip")
		zipFile, err := os.Create(zipPath)
		if err != nil {
			return "", fmt.Errorf("failed to create zip file: %w", err)
		}

		// Use COS client to download the file
		err = w.cosClient.DownloadFile(w.ctx, w.cfg.CodeBucket, taskMsg.CodeLocation, zipFile)
		zipFile.Close()
		if err != nil {
			os.Remove(zipPath)
			return "", fmt.Errorf("failed to download ZIP: %w", err)
		}

		// Unzip the file
		cmd := exec.Command("unzip", "-o", zipPath, "-d", codeDir)
		output, err := cmd.CombinedOutput()
		if err != nil {
			os.Remove(zipPath)
			return "", fmt.Errorf("unzip failed: %v, output: %s", err, string(output))
		}

		// Clean up zip file
		os.Remove(zipPath)

	default:
		return "", fmt.Errorf("unsupported input type: %s", taskMsg.InputType)
	}

	return codeDir, nil
}

// executeAgent runs the agent using the configured execution mode
func (w *Worker) executeAgent(taskMsg task.KafkaTaskMessage, codeDir string) error {
	// Update status to running agent
	if w.db != nil {
		w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusRunningAgent, "Running agent")
	}

	// Determine if it's a GitHub repo
	isGitHubRepo := strings.Contains(taskMsg.CodeLocation, "github.com")

	// Set up environment variables for the agent
	envVars := map[string]string{
		"OPENAI_API_KEY": os.Getenv("OPENAI_API_KEY"),
		"TASK_ID":        taskMsg.TaskID,
	}

	// Add GitHub token if available
	if taskMsg.UserGitHubToken != "" && w.cfg.EnableGitHubAccess {
		envVars["GITHUB_TOKEN"] = taskMsg.UserGitHubToken
	}

	// Execute agent using configured mode
	switch w.cfg.ExecutionMode {
	case DockerMode:
		return w.runDockerAgent(taskMsg, codeDir, envVars, isGitHubRepo)
	case K8sMode:
		return w.runK8sAgent(taskMsg, codeDir, envVars, isGitHubRepo)
	default:
		return fmt.Errorf("unsupported execution mode: %s", w.cfg.ExecutionMode)
	}
}

// runDockerAgent executes the agent in a Docker container
func (w *Worker) runDockerAgent(
	taskMsg task.KafkaTaskMessage,
	codeDir string,
	envVars map[string]string,
	isGitHubRepo bool,
) error {
	// Create output directory
	outputDir := filepath.Join(filepath.Dir(codeDir), "output")
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Convert env vars map to string slice
	env := []string{}
	for k, v := range envVars {
		env = append(env, fmt.Sprintf("%s=%s", k, v))
	}

	// Create container config
	containerConfig := &container.Config{
		Image: w.cfg.AgentImage,
		Cmd: []string{
			taskMsg.TaskDescription,
			taskMsg.TargetFile,
			fmt.Sprintf("%v", isGitHubRepo),
		},
		Env: env,
		Tty: false,
	}

	// Host config with volume mounts
	hostConfig := &container.HostConfig{
		Binds: []string{
			fmt.Sprintf("%s:/app/code:ro", codeDir),
			fmt.Sprintf("%s:/app/output:rw", outputDir),
		},
		Resources: container.Resources{
			Memory:   1024 * 1024 * 1024 * 2, // 2GB
			NanoCPUs: 1000000000,             // 1 CPU
		},
	}

	// Create and start container
	resp, err := w.dockerClient.ContainerCreate(
		w.ctx,
		containerConfig,
		hostConfig,
		nil,
		nil,
		fmt.Sprintf("codex-agent-%s", taskMsg.TaskID),
	)
	if err != nil {
		return fmt.Errorf("failed to create container: %w", err)
	}

	// Start container
	if err := w.dockerClient.ContainerStart(w.ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return fmt.Errorf("failed to start container: %w", err)
	}

	// Wait for container to finish
	statusCh, errCh := w.dockerClient.ContainerWait(w.ctx, resp.ID, container.WaitConditionNotRunning)
	var statusCode int64
	select {
	case err := <-errCh:
		if err != nil {
			return fmt.Errorf("error waiting for container: %w", err)
		}
	case status := <-statusCh:
		statusCode = status.StatusCode
	}

	// Get container logs
	out, err := w.dockerClient.ContainerLogs(w.ctx, resp.ID, types.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     false,
	})
	if err != nil {
		log.Printf("Failed to get container logs: %v", err)
	} else {
		defer out.Close()

		// Write logs to file
		logFile, err := os.Create(filepath.Join(outputDir, "agent.log"))
		if err != nil {
			log.Printf("Failed to create log file: %v", err)
		} else {
			defer logFile.Close()
			io.Copy(logFile, out)
		}
	}

	// Upload results to object storage
	err = w.uploadResults(taskMsg.TaskID, outputDir)
	if err != nil {
		return fmt.Errorf("failed to upload results: %w", err)
	}

	// Update task status based on container exit code
	if statusCode != 0 {
		if w.db != nil {
			w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusFailed, fmt.Sprintf("Agent exited with code %d", statusCode))
		}
		return fmt.Errorf("agent container exited with code %d", statusCode)
	}

	// Mark task as completed
	if w.db != nil {
		w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusCompleted, "Agent completed successfully")
	}

	return nil
}

// runK8sAgent executes the agent as a Kubernetes job
func (w *Worker) runK8sAgent(
	taskMsg task.KafkaTaskMessage,
	codeDir string,
	envVars map[string]string,
	isGitHubRepo bool,
) error {
	// First, upload code to object storage for K8s job to access
	codeCOSPath := fmt.Sprintf("agent_input/%s/", taskMsg.TaskID)
	if err := w.uploadCodeToStorage(taskMsg.TaskID, codeDir, codeCOSPath); err != nil {
		return fmt.Errorf("failed to upload code to storage: %w", err)
	}

	// Convert env vars map to k8s env vars
	env := []corev1.EnvVar{}
	for k, v := range envVars {
		env = append(env, corev1.EnvVar{Name: k, Value: v})
	}

	// Sanitize job name for K8s (must be DNS-1123 compatible)
	jobName := fmt.Sprintf("codex-agent-%s", strings.ReplaceAll(taskMsg.TaskID, "_", "-"))
	jobName = strings.ToLower(jobName)
	if len(jobName) > 63 {
		jobName = jobName[:63]
	}

	// Define volume names
	codeVolumeName := "agent-code"
	outputVolumeName := "agent-output"

	// Create job spec
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: w.cfg.K8sNamespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy:      corev1.RestartPolicyNever,
					ServiceAccountName: w.cfg.K8sServiceAccount,
					Containers: []corev1.Container{
						{
							Name:    "codex-agent",
							Image:   w.cfg.AgentImage,
							Command: []string{"python3", "agent.py"},
							Args: []string{
								taskMsg.TaskDescription,
								taskMsg.TargetFile,
								fmt.Sprintf("%v", isGitHubRepo),
							},
							Env: env,
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse(w.cfg.CPULimit),
									corev1.ResourceMemory: resource.MustParse(w.cfg.MemoryLimit),
								},
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("100m"),
									corev1.ResourceMemory: resource.MustParse("256Mi"),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: codeVolumeName, MountPath: "/app/code"},
								{Name: outputVolumeName, MountPath: "/app/output"},
							},
						},
					},
					// Init container to download code from object storage
					InitContainers: []corev1.Container{
						{
							Name:    "setup-code",
							Image:   "amazon/aws-cli:latest", // Or appropriate COS tool image
							Command: []string{"sh", "-c"},
							Args: []string{
								fmt.Sprintf(
									"aws s3 sync s3://%s/%s /app/code --delete && echo 'Code downloaded'",
									w.cfg.CodeBucket, codeCOSPath,
								),
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: codeVolumeName, MountPath: "/app/code"},
							},
							Env: []corev1.EnvVar{
								{Name: "AWS_ACCESS_KEY_ID", Value: os.Getenv("COS_ACCESS_KEY")},
								{Name: "AWS_SECRET_ACCESS_KEY", Value: os.Getenv("COS_SECRET_KEY")},
								{Name: "AWS_DEFAULT_REGION", Value: "ap-guangzhou"},                        // Adjust for your COS region
								{Name: "AWS_ENDPOINT_URL", Value: "https://cos.ap-guangzhou.myqcloud.com"}, // Adjust for COS
							},
						},
					},
					// Volumes for code and output
					Volumes: []corev1.Volume{
						{
							Name: codeVolumeName,
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
						{
							Name: outputVolumeName,
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			},
			BackoffLimit:            int32Ptr(2),    // Retry twice on failure
			TTLSecondsAfterFinished: int32Ptr(3600), // Auto-delete after 1 hour
		},
	}

	// Create the job
	_, err := w.k8sClient.BatchV1().Jobs(w.cfg.K8sNamespace).Create(w.ctx, job, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create K8s job: %w", err)
	}

	log.Printf("K8s job %s created for task %s", jobName, taskMsg.TaskID)

	// Monitor job status (could be made async with timeouts in production)
	for {
		time.Sleep(5 * time.Second)

		jobStatus, err := w.getK8sJobStatus(jobName)
		if err != nil {
			log.Printf("Failed to get job status: %v", err)
			continue
		}

		if jobStatus == JobSucceeded {
			// Download results from pod
			if err := w.downloadK8sJobResults(taskMsg.TaskID, jobName); err != nil {
				log.Printf("Failed to download job results: %v", err)
				w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusFailed,
					fmt.Sprintf("Failed to retrieve results: %v", err))
				return err
			}

			// Mark task as completed
			if w.db != nil {
				w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusCompleted, "Agent completed successfully")
			}
			return nil
		} else if jobStatus == JobFailed {
			errMsg := "K8s job failed"
			// Try to get logs
			logs, err := w.getK8sJobLogs(jobName)
			if err == nil && logs != "" {
				errMsg = fmt.Sprintf("Agent failed: %s", logs)
			}

			// Mark task as failed
			if w.db != nil {
				w.db.UpdateTaskStatus(w.ctx, taskMsg.TaskID, task.StatusFailed, errMsg)
			}
			return fmt.Errorf(errMsg)
		}
		// Continue waiting for job to complete
	}
}

// uploadCodeToStorage uploads code to object storage for K8s job to access
func (w *Worker) uploadCodeToStorage(taskID, codeDir, codeCOSPath string) error {
	return w.cosClient.UploadDirectory(w.ctx, w.cfg.CodeBucket, codeCOSPath, codeDir)
}

// uploadResults uploads agent output to object storage
func (w *Worker) uploadResults(taskID, outputDir string) error {
	// Upload all files in the output directory to object storage
	outputCOSPath := fmt.Sprintf("logs/%s/", taskID)
	return w.cosClient.UploadDirectory(w.ctx, w.cfg.LogsBucket, outputCOSPath, outputDir)
}

// getK8sJobStatus checks the status of a K8s job
func (w *Worker) getK8sJobStatus(jobName string) (JobStatus, error) {
	job, err := w.k8sClient.BatchV1().Jobs(w.cfg.K8sNamespace).Get(w.ctx, jobName, metav1.GetOptions{})
	if err != nil {
		return JobUnknown, fmt.Errorf("failed to get job status: %w", err)
	}

	if job.Status.Succeeded > 0 {
		return JobSucceeded, nil
	}
	if job.Status.Failed > 0 {
		return JobFailed, nil
	}
	if job.Status.Active > 0 {
		return JobRunning, nil
	}
	return JobUnknown, nil
}

// getK8sJobLogs retrieves logs from a K8s job pod
func (w *Worker) getK8sJobLogs(jobName string) (string, error) {
	// Get pods for the job
	pods, err := w.k8sClient.CoreV1().Pods(w.cfg.K8sNamespace).List(w.ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("job-name=%s", jobName),
	})
	if err != nil || len(pods.Items) == 0 {
		return "", fmt.Errorf("failed to find pods for job: %w", err)
	}

	// Get logs from the first pod
	podName := pods.Items[0].Name
	logs, err := w.k8sClient.CoreV1().Pods(w.cfg.K8sNamespace).GetLogs(podName, &corev1.PodLogOptions{
		Container: "codex-agent",
	}).Do(w.ctx).Raw()

	if err != nil {
		return "", fmt.Errorf("failed to get pod logs: %w", err)
	}

	return string(logs), nil
}

// downloadK8sJobResults downloads results from a completed K8s job
func (w *Worker) downloadK8sJobResults(taskID, jobName string) error {
	// In a real implementation, this would:
	// 1. Either copy files from the pod's output volume before it's terminated
	// 2. Or have the agent upload results directly to object storage
	// 3. Or use a persistent volume that survives pod termination

	// For this example, we assume the agent uploads results directly to object storage
	// So there's nothing additional to do here
	return nil
}

// Stop gracefully shuts down the worker
func (w *Worker) Stop() error {
	w.cancel()
	if err := w.kafkaConsumer.Close(); err != nil {
		return fmt.Errorf("failed to close kafka consumer: %w", err)
	}
	return nil
}

// updateTaskFailure is a helper to update task status on failure
func (w *Worker) updateTaskFailure(taskID, message string) {
	if w.db != nil {
		w.db.UpdateTaskStatus(w.ctx, taskID, task.StatusFailed, message)
	}
}

// int32Ptr converts an int32 to a pointer
func int32Ptr(i int32) *int32 {
	return &i
}
