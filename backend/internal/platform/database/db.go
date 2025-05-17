package database

import (
	"context"
	"time"

	"github.com/jmoiron/sqlx"
	"github.com/yunmaoQu/codex-sys/internal/task"
)

// DBClientWrapper wraps a sqlx.DB connection with task-specific operations
type DBClientWrapper struct {
	db *sqlx.DB
}

// NewMySQLConnection creates a new MySQL connection
func NewMySQLConnection(dsn string) (*sqlx.DB, error) {
	db, err := sqlx.Connect("mysql", dsn)
	if err != nil {
		return nil, err
	}

	// Configure connection pool
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	return db, nil
}

// NewDBClientWrapper creates a new DBClientWrapper
func NewDBClientWrapper(db *sqlx.DB) *DBClientWrapper {
	return &DBClientWrapper{db: db}
}

// UpdateTaskStatus updates the status of a task
func (w *DBClientWrapper) UpdateTaskStatus(ctx context.Context, taskID string, status task.Status, message string) error {
	return task.UpdateStatus(ctx, w.db, taskID, status, message)
}

// GetTaskByID retrieves a task by its ID
func (w *DBClientWrapper) GetTaskByID(ctx context.Context, taskID string) (*task.Definition, error) {
	return task.GetByID(ctx, w.db, taskID)
}

// UpdateTaskPRURL updates the PR URL for a task
func (w *DBClientWrapper) UpdateTaskPRURL(ctx context.Context, taskID string, prURL string) error {
	query := `UPDATE tasks SET pr_url = ?, updated_at = ? WHERE id = ?`
	_, err := w.db.ExecContext(ctx, query, prURL, time.Now(), taskID)
	return err
}