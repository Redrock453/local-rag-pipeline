# nginx as a reverse proxy

Terminate TLS in nginx and forward plain HTTP to the application on a local
port. A minimal server block needs `proxy_pass http://127.0.0.1:8000;` plus the
headers `Host`, `X-Real-IP` and `X-Forwarded-Proto` so the application sees the
original request details.

Always run `nginx -t` before `systemctl reload nginx`. A failed test leaves the
running configuration untouched, a bad reload can take the site down.

For streaming responses (server-sent events, token streaming) disable proxy
buffering for that location with `proxy_buffering off;`.
