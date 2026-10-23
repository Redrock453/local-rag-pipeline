# Running a Python service under systemd

Create `/etc/systemd/system/myapp.service` with `ExecStart` pointing at the
virtualenv interpreter, a dedicated `User=`, and `Restart=on-failure`.

Load the unit with `systemctl daemon-reload`, then `systemctl enable --now
myapp`. Read logs with `journalctl -u myapp -f`.

Keep secrets out of the unit file: put them in an `EnvironmentFile=` with mode
0600 owned by root.
