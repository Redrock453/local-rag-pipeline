# Docker Compose basics

Compose describes a multi-container application in one `docker-compose.yml`.

Start everything in the background with `docker compose up -d`. Show service
logs with `docker compose logs -f <service>`. Stop and remove the containers
with `docker compose down`; add `-v` only if you also want to delete volumes.

A service that must survive host reboots needs `restart: unless-stopped`.
Data that must outlive a container belongs in a named volume, not in the
container's writable layer.

## Healthchecks

Add a `healthcheck` to every service other services depend on, and use
`depends_on` with `condition: service_healthy` so startup order is reliable.
