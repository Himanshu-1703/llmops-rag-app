# PromQL Queries for Grafana

### CPU Usage per Core

```
100 - (avg by (instance, cpu) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```
legend --> {{instance}} - core {{cpu}}

### Used RAM

```
avg(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1e9
```
legend --> Used

```
avg(node_memory_MemAvailable_bytes) / 1e9
```
legend --> Available


### Network Speed(KB/sec)

```
rate(node_network_receive_bytes_total{device!="lo"}[5m]) / 1000
```
label --> {{instance}} {{device}} in

```
rate(node_network_transmit_bytes_total{device!="lo"}[5m]) / 1000
```
label --> {{instance}} {{device}} out


### Disk Usage(%)

```
(1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100
``` 

### Disk Usage (GB)

```
(node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"}) / 1e9
```


### Template ID
1860