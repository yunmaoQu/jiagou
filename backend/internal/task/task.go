package task

import (
	"context"
	"time"

	"github.com/jmoiron/sqlx"
)

type Status string
type InputType string

const (
	StatusPending          Status = "PENDING"
	StatusQueued           Status = "QUEUED"
	StatusProcessing       Status = "PROCESSING"
	StatusDownloadingCode  Status = "DOWNLOADING_CODE"
	StatusRunningAgent     Status = "RUNNING_AGENT"
	StatusUploadingResults Status = "UPLOADING_RESULTS"
	StatusCompleted        Status = "COMPLETED"
	StatusFailed           Status = "FAILED"

	InputGit InputType = "GIT"
	InputZip InputType = "ZIP"
)

type Definition struct {
	ID              string            `db:"id" json:"id"`
	InputType       InputType         `db:"input_type" json:"input_type"`
	GitURL          string            `db:"git_url,omitempty" json:"git_url,omitempty"`
	ZipFileName     string            `db:"zip_file_name,omitempty" json:"zip_file_name,omitempty"`
	CodePathCOS     string            `db:"code_path_cos,omitempty" json:"code_path_cos,omitempty"`
	TargetFile      string            `db:"target_file" json:"target_file"`
	TaskDescription string            `db:"task_description" json:"task_description"`
	Status          Status            `db:"status" json:"status"`
	Message         string            `db:"message,omitempty" json:"message,omitempty"`
	GitHubToken     string            `db:"github_token,omitempty" json:"-"`
	PRURL           string            `db:"pr_url,omitempty" json:"pr_url,omitempty"`
	CreatedAt       time.Time         `db:"created_at" json:"created_at"`
	UpdatedAt       time.Time         `db:"updated_at" json:"updated_at"`
	LogFileURLs     map[string]string `json:"log_file_urls,omitempty"`
}

type KafkaTaskMessage struct {
	TaskID             string `json:"task_id"`
	InputType          string `json:"input_type"`
	CodeLocation       string `json:"code_location"`
	TargetFile         string `json:"target_file"`
	TaskDescription    string `json:"task_description"`
	UserGitHubToken    string `json:"user_github_token,omitempty"`
	OpenAIAPIKeySource string `json:"openai_api_key_source"`
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
