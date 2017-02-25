BROKER_HOST = "localhost"
BROKER_PORT = 6379
#BROKER_USER = "guest"
#BROKER_PASSWORD = "guest"
BROKER_VHOST = "/"

# バックエンドを指定。今回はRabbitMQに対してAMQPというプロトコルで接続する
CELERY_RESULT_BACKEND = "redis"

# workerの設定
## 平行度 CPUの数に近づけるといいらしい。省略するとCPU/coreが使われる。
#CELERYD_CONCURRENCY
## ログの出力先。省略すると標準エラー出力が選ばれる
CELERYD_LOG_FILE = "celeryd.log"
## ログのレベル
CELERYD_LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR or CRITICAL

# 起動時に読み込むモジュール
CELERY_IMPORTS = ("app",)