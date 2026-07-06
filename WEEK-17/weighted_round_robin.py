"""weighted_round_robin.py

Implementasi algoritma Smooth Weighted Round-Robin Load Balancing.
Mendistribusikan request secara seimbang dan halus (interleaved) dengan 
memperhitungkan bobot tanpa membebani memori.
"""

from abc import ABC, abstractmethod


class LoadBalancer(ABC):
    """Base class abstrak untuk mendefinisikan interface Load Balancer."""

    def __init__(self, servers: list[str]) -> None:
        if not servers:
            raise ValueError("Daftar server tidak boleh kosong.")
        self.servers = list(servers)

    @abstractmethod
    def get_server(self, client_ip: str) -> str:
        """Memilih dan mengembalikan server tujuan berdasarkan strategi algoritma."""
        pass


class WeightedRoundRobinBalancer(LoadBalancer):
    """Varian Weighted Round-Robin menggunakan algoritma Smooth Distribution.

    Menggunakan pendekatan state-based untuk menghemat memori menjadi O(N) dan 
    menghindari penumpukan request (traffic burst) pada satu server di awal putaran.
    """

    def __init__(self, servers: dict[str, int]) -> None:
        """Inisialisasi konfigurasi server beserta bobotnya.

        Args:
            servers: Dictionary berisi pemetaan nama_server (str) ke bobot (int).

        Raises:
            ValueError: Jika dictionary kosong atau terdapat bobot yang <= 0.
        """
        if not servers:
            raise ValueError("Daftar server dan bobot tidak boleh kosong.")

        for server, weight in servers.items():
            if weight <= 0:
                raise ValueError(f"Bobot untuk {server} harus lebih besar dari 0.")

        # Inisialisasi base class menggunakan daftar nama server (keys)
        super().__init__(list(servers.keys()))
        
        self.weights = servers
        
        # State dinamis untuk melacak bobot berjalan di memori (Hanya berukuran sebanyak jumlah server)
        self.current_weights = {server: 0 for server in self.servers}
        self.total_weight = sum(servers.values())

    def get_server(self, client_ip: str) -> str:
        """Memilih server berikutnya menggunakan kalkulasi bobot dinamis.

        Cara kerja setiap request:
        1. Setiap server mengumpulkan bobot dinamisnya (current += weight).
        2. Pilih server dengan current_weight terbesar.
        3. Kurangi current_weight server terpilih dengan total_weight seluruh server.
        """
        # 1. Tambahkan current_weight dengan bobot asli masing-masing server
        for server in self.servers:
            self.current_weights[server] += self.weights[server]

        # 2. Cari server dengan current_weight terbesar saat ini
        chosen = max(self.current_weights, key=lambda s: self.current_weights[s])

        # 3. Kurangi current_weight milik server terpilih dengan total_weight seluruh konfigurasi
        self.current_weights[chosen] -= self.total_weight

        return chosen


def main() -> None:
    """Fungsi utama untuk menjalankan simulasi dan pengujian manual."""
    # Skenario ekstrim untuk pembuktian efisiensi memori dan kehalusan distribusi
    server_config = {
        "server-A": 10000,
        "server-B": 1,
        "server-C": 5000,
    }

    print("=== Konfigurasi Smooth Weighted Round-Robin ===")
    for server, weight in server_config.items():
        print(f"- {server} (Weight: {weight})")

    # Inisialisasi balancer (Memori tetap sangat kecil karena tidak mengekspansi list)
    balancer = WeightedRoundRobinBalancer(server_config)

    print("\n=== Simulasi 10 Request Pertama ===")
    print("Memperlihatkan distribusi yang saling bersilangan (interleaved), bukan menumpuk:")
    
    clients = [f"10.0.0.{i}" for i in range(1, 11)]
    for ip in clients:
        target = balancer.get_server(ip)
        print(f"{ip} -> {target} | Current State Score: {balancer.current_weights}")


if __name__ == "__main__":
    main()