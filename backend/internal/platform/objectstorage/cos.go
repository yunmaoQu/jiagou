package objectstorage

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tencentyun/cos-go-sdk-v5" // Tencent Cloud COS SDK
)

// This is a simplified wrapper. You'd need to configure credentials etc.
// For S3, you'd use aws-sdk-go. For MinIO, also aws-sdk-go compatible.

type ClientWrapper struct {
	client    *cos.Client
	secretID  string
	secretKey string
}

type COSConfig struct { // Example config structure
	SecretID  string
	SecretKey string
	Region    string
	BucketURL string // e.g., https://mybucket-1250000000.cos.ap-guangzhou.myqcloud.com
}

func NewCOSClient(cfg COSConfig) (*ClientWrapper, error) {
	u, _ := url.Parse(cfg.BucketURL)
	b := &cos.BaseURL{BucketURL: u}
	client := cos.NewClient(b, &http.Client{
		Transport: &cos.AuthorizationTransport{
			SecretID:  cfg.SecretID,
			SecretKey: cfg.SecretKey,
		},
	})
	return &ClientWrapper{
		client:    client,
		secretID:  cfg.SecretID,
		secretKey: cfg.SecretKey,
	}, nil
}

func (cw *ClientWrapper) UploadFile(ctx context.Context, bucketName string, key string, reader io.Reader, size int64) error {
	// bucketName might be part of the BaseURL or specified here depending on SDK usage
	opt := &cos.ObjectPutOptions{
		ObjectPutHeaderOptions: &cos.ObjectPutHeaderOptions{
			ContentLength: size,
		},
	}
	_, err := cw.client.Object.Put(ctx, key, reader, opt)
	if err != nil {
		return fmt.Errorf("COS Put failed for key %s: %w", key, err)
	}
	return nil
}

func (cw *ClientWrapper) DownloadFile(ctx context.Context, bucketName string, key string, writer io.Writer) error {
	resp, err := cw.client.Object.Get(ctx, key, nil)
	if err != nil {
		return fmt.Errorf("COS Get failed for key %s: %w", key, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("COS Get failed for key %s: status %d", key, resp.StatusCode)
	}
	_, err = io.Copy(writer, resp.Body)
	if err != nil {
		return fmt.Errorf("COS download copy failed for key %s: %w", key, err)
	}
	return nil
}

// GetPresignedURL generates a presigned URL for the given object key.
func (cw *ClientWrapper) GetPresignedURL(ctx context.Context, bucketName string, key string, duration time.Duration) (string, error) {
	presignedURL, err := cw.client.Object.GetPresignedURL(ctx, http.MethodGet, key, cw.secretID, cw.secretKey, duration, nil)
	if err != nil {
		return "", err
	}
	return presignedURL.String(), nil
}

// UploadDirectory uploads all files in a directory to COS recursively.
func (cw *ClientWrapper) UploadDirectory(ctx context.Context, bucketName string, cosPrefix string, localDirPath string) error {
	// This is a simplified implementation that walks through the directory
	// and uploads each file individually
	return filepath.Walk(localDirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		// Skip directories
		if info.IsDir() {
			return nil
		}
		
		// Calculate the relative path from the base directory
		relPath, err := filepath.Rel(localDirPath, path)
		if err != nil {
			return fmt.Errorf("failed to get relative path: %w", err)
		}
		
		// Create the COS key by joining the prefix and relative path
		cosKey := filepath.Join(cosPrefix, relPath)
		// Convert Windows backslashes to forward slashes for COS paths
		cosKey = strings.ReplaceAll(cosKey, "\\", "/")
		
		// Open the file for reading
		file, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("failed to open file %s: %w", path, err)
		}
		defer file.Close()
		
		// Upload the file
		if err := cw.UploadFile(ctx, bucketName, cosKey, file, info.Size()); err != nil {
			return fmt.Errorf("failed to upload file %s to %s: %w", path, cosKey, err)
		}
		
		return nil
	})
}
