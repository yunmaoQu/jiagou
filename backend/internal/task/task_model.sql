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