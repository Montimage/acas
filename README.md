# Network Detection and Response (NDR)
Advanced Cybersecurity Analytics Service (ACAS) for AI-based analysis of network events captured by the Montimage Monitoring Tools (MMT).

## Installation
### Build from Docker
```bash
# Clone the repo
git clone --branch cybersuite --single-branch https://github.com/Montimage/acas.git
cd acas

# Copy the Docker configuration template
cp env.example .env

# Build and run the server
sudo docker-compose -f docker-compose.server-redis.yml up --build

# Access the Swagger UI on http://localhost:31057/docs/
```
