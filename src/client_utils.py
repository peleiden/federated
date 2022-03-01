import socket


def set_hostname(hostname: str):
    """ Changes the hostname of the device. Requires sudo priviledges to run. """
    with open("/etc/hostname", "w") as hostname_file:
        hostname_file.write(hostname+"\n")

    # To change name in the hosts file, first find the correct line to change and the change it
    with open("/etc/hosts") as hosts_file:
        lines = hosts_file.readlines()
    index_of_line_to_change = next(i for i, line in enumerate(lines) if line.startswith("127.0.1.1"))
    lines[index_of_line_to_change] = "127.0.1.1\t" + hostname + "\n"
    with open("/etc/hosts", "w") as hosts_file:
        hosts_file.writelines(lines)

def set_static_ip(ip: str):
    with open("/etc/dhcpcd.conf", "a") as conffile:
        conffile.writelines([
            "interface wlan0\n",
            "static ip_address=%s/24\n" % ip,
            "static routers=192.168.0.1\n",
            "static domain_name_servers=192.168.0.1\n",
        ])

def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip
