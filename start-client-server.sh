# Starts a client server. Only use this on a Raspberry Pi
# Otherwise, use make start-server
cd "$(dirname "$0")"

python src/client_server.py
