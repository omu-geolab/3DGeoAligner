# ベースイメージ
FROM python:3.9-slim

# 環境変数
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# システムライブラリ (libgl1必須)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# ライブラリインストール（キャッシュ効率が良い順序を維持）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ディレクトリ作成
# 前回決めた構成案（results）に合わせるのがおすすめです
RUN mkdir -p /app/data /app/results /app/src

# デフォルトでBashを起動
CMD ["/bin/bash"]