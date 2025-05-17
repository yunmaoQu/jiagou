package k8s

import (
	"context"
	"fmt"
	"log"

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
					RestartPolicy:      corev1.RestartPolicyNever, // Or OnFailure
					ServiceAccountName: cfg.ServiceAccount,        // If specified
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
							Name:    "setup-code-volume",
							Image:   "amazon/aws-cli:latest", // Or appropriate COS tool image
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
			BackoffLimit:            int32Ptr(1),    // Retry once on failure
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
