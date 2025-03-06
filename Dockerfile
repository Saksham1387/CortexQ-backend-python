FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.1

USER root

# Set environment variables for Elasticsearch configuration
ENV ELASTIC_PASSWORD=123456
ENV discovery.type=single-node
ENV xpack.security.enabled=true
ENV xpack.security.enrollment.enabled=true
ENV xpack.security.http.ssl.enabled=false
ENV xpack.security.transport.ssl.enabled=false

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directory for Python application
WORKDIR /app

# Copy requirements.txt first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Python script
COPY ./src /app/src

# Create a startup script to run both Elasticsearch and Python script
RUN echo '#!/bin/bash\n\
    # Start Elasticsearch in the background\n\
    su elasticsearch -c "/usr/local/bin/docker-entrypoint.sh elasticsearch" &\n\
    \n\
    # Wait for Elasticsearch to be ready\n\
    until curl -s -u elastic:123456 http://localhost:9200 > /dev/null; do\n\
    echo "Waiting for Elasticsearch..."\n\
    sleep 5\n\
    done\n\
    \n\
    echo "Elasticsearch is up, running Python script..."\n\
    python3 /app/src/insert.py\n\
    \n\
    # Keep container running\n\
    tail -f /dev/null' > /start.sh && \
    chmod +x /start.sh

# Expose the default Elasticsearch ports
EXPOSE 9200 9300

# Set the entry point to our custom script
ENTRYPOINT ["/start.sh"]
