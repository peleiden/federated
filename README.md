# federated
Federated Learning project for 02460 Advanced Machine Learning

# API Description

`/ping` returns basic device information such as hostname, commit of the repository and commit of the repository on the device.

`/logs` returns the logfile content.

`/telemetry` returns telemetry such as CPU and memory usage.

`/command` is used to issue system commands to the Pi. This effectively over-the-air updating by issuing `git pull` followed by `reboot`.

`/configure` takes an integer `num` as argument and sets the local static IP and hostname accordingly.
