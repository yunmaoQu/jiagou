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
			TaskTopic   string
			ResultTopic string
		}
	}
	COS struct {
		Bucket    string
		Region    string
		AccessKey string
		SecretKey string
	}
	Docker struct {
		AgentImage string
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
