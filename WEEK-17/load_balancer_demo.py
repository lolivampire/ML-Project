"""load_balancer_demo.py

WEEK-17/Simulasi sederhana tiga algoritma load balancing tanpa network call.
Fokus pada kejelasan logika distribusinya.
"""

from abc import ABC, abstractmethod
import hashlib
import itertools


class LoadBalancer(ABC):
    """Base class abstrak untuk mendefinisikan interface Load Balancer."""

    def __init__(self, servers: list[str]) -> None:
        if not servers:
            raise ValueError("Daftar server tidak boleh kosong.")
        self.servers = list(servers)

    @abstractmethod
    def get_server(self, client_ip: str) -> str:
        """Memilih dan mengembalikan server tujuan berdasarkan strategi algoritma.

        Args:
            client_ip: String alamat IP milik client.

        Returns:
            str: Nama server yang terpilih.
        """
        pass


class RoundRobinBalancer(LoadBalancer):
    """Distribusi giliran berurutan (sequential) tanpa melihat beban server."""

    def __init__(self, servers: list[str]) -> None:
        super().__init__(servers)
        self._cycle = itertools.cycle(self.servers)

    def get_server(self, client_ip: str) -> str:
        """Mengambil server berikutnya dalam antrean rotasi."""
        return next(self._cycle)


class LeastConnectionBalancer(LoadBalancer):
    """Distribusi berdasarkan jumlah koneksi aktif yang paling sedikit."""

    def __init__(self, servers: list[str]) -> None:
        super().__init__(servers)
        self.active_connections: dict[str, int] = {s: 0 for s in self.servers}

    def get_server(self, client_ip: str) -> str:
        """Memilih server dengan jumlah koneksi terendah saat ini."""
        # Mencari key (server) dengan value (koneksi) terkecil
        chosen = min(self.active_connections, key=lambda s: self.active_connections[s])
        self.active_connections[chosen] += 1
        return chosen

    def release_connection(self, server: str) -> None:
        """Menurunkan jumlah koneksi aktif ketika request selesai diproses."""
        if server in self.active_connections and self.active_connections[server] > 0:
            self.active_connections[server] -= 1


class IPHashBalancer(LoadBalancer):
    """Distribusi berdasarkan hash IP client untuk menjaga sticky session."""

    def get_server(self, client_ip: str) -> str:
        """Memetakan IP client secara konsisten ke server yang sama."""
        hash_hex = hashlib.md5(client_ip.encode()).hexdigest()
        hash_value = int(hash_hex, 16)
        index = hash_value % len(self.servers)
        return self.servers[index]


def run_simulation(balancer: LoadBalancer, clients: list[str], label: str) -> None:
    """Helper function untuk menjalankan dan menampilkan hasil simulasi."""
    print(f"\n=== {label} ===")
    for ip in clients:
        target = balancer.get_server(ip)
        
        # Tampilkan status koneksi khusus untuk Least Connection
        if isinstance(balancer, LeastConnectionBalancer):
            print(f"{ip} -> {target} (State Koneksi: {balancer.active_connections})")
        else:
            print(f"{ip} -> {target}")


def main() -> None:
    servers = ["server-A", "server-B", "server-C"]
    clients = ["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.3", "10.0.0.2"]

    # Inisialisasi masing-masing balancer
    rr_balancer = RoundRobinBalancer(servers)
    lc_balancer = LeastConnectionBalancer(servers)
    ih_balancer = IPHashBalancer(servers)

    # Eksekusi simulasi
    run_simulation(rr_balancer, clients, "Round-Robin Balancer")
    run_simulation(lc_balancer, clients, "Least-Connection Balancer")
    run_simulation(ih_balancer, clients, "IP-Hash Balancer")


if __name__ == "__main__":
    main()