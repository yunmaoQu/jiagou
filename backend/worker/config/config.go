package config

import (
	"gopkg.in/yaml.v2"
	"os"
)

// Config 配置结构体
type Config struct {
	Kafka struct {
		Brokers []string
		Topics  struct {
			Task   string
			Result string
		}
	}
	// COS configuration
	COS struct {
		AccessKey  string
		SecretKey string
		Region    string
		Buckets   struct {
			Code string
			Logs string
		}
	}
	Docker struct {
		AgentImage string
	}

	Database struct {
		Host     string
		Port     string
		User string
		Password string
		Name     string
	}
}

// LoadFromYAML 加载 YAML 配置文件
func LoadFromYAML(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
