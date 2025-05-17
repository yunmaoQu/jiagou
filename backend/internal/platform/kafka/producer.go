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
