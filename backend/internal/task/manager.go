package task

import (
	"context"
	"database/sql"
	"log"
	"sync"

	"github.com/jmoiron/sqlx"
)

// TaskManager 管理任务的创建、更新和查询
type TaskManager struct {
	db *sqlx.DB
	mutex sync.RWMutex
}

// 全局任务管理器实例
var (
	globalManager *TaskManager
	once         sync.Once
)

// InitManager 初始化任务管理器
// 现在使用数据库存储任务状态
func InitManager() {
	log.Println("Task manager initialized (using database storage)")
}

// SetDB 设置任务管理器使用的数据库连接
func SetDB(db *sqlx.DB) {
	once.Do(func() {
		globalManager = &TaskManager{
			db: db,
		}
		log.Println("Task manager initialized with database connection")

		// 确保任务表存在
		ensureTaskTableExists(db)
	})
}

// GetManager 返回全局任务管理器实例
func GetManager() *TaskManager {
	if globalManager == nil {
		log.Println("WARNING: Task manager not initialized with database. Using empty manager.")
		return &TaskManager{}
	}
	return globalManager
}

// ensureTaskTableExists 确保任务表存在
func ensureTaskTableExists(db *sqlx.DB) {
	// 创建任务表的SQL语句
	schema := `
	CREATE TABLE IF NOT EXISTS tasks (
		id VARCHAR(36) PRIMARY KEY,
		input_type VARCHAR(10) NOT NULL,
		git_url TEXT,
		zip_file_name TEXT,
		code_path_cos TEXT,
		target_file TEXT NOT NULL,
		task_description TEXT NOT NULL,
		status VARCHAR(20) NOT NULL,
		message TEXT,
		github_token TEXT,
		pr_url TEXT,
		created_at TIMESTAMP NOT NULL,
		updated_at TIMESTAMP NOT NULL
	);`

	// 执行创建表的SQL语句
	_, err := db.Exec(schema)
	if err != nil {
		log.Printf("Error creating tasks table: %v", err)
	}
}

// CreateTask 创建新任务
func (m *TaskManager) CreateTask(ctx context.Context, task *Definition) error {
	if m.db == nil {
		return sql.ErrConnDone
	}

	m.mutex.Lock()
	defer m.mutex.Unlock()

	return Create(ctx, m.db, task)
}

// GetTaskByID 根据ID获取任务
func (m *TaskManager) GetTaskByID(ctx context.Context, id string) (*Definition, error) {
	if m.db == nil {
		return nil, sql.ErrConnDone
	}

	m.mutex.RLock()
	defer m.mutex.RUnlock()

	return GetByID(ctx, m.db, id)
}

// UpdateTaskStatus 更新任务状态
func (m *TaskManager) UpdateTaskStatus(ctx context.Context, id string, status Status, message string) error {
	if m.db == nil {
		return sql.ErrConnDone
	}

	m.mutex.Lock()
	defer m.mutex.Unlock()

	return UpdateStatus(ctx, m.db, id, status, message)
}
