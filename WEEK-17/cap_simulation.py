"""cap_simulation.py

Simulasi interaktif Teorema CAP menggunakan pendekatan berbasis Node Spesifik.
Menunjukkan perbedaan perilaku READ dan WRITE pada CPNode (Prioritas Konsistensi)
dan APNode (Prioritas Ketersediaan) saat terjadi Network Partition.
"""

from typing import Dict


class Network:
    """Mengelola status koneksi antarnode dalam cluster."""

    def __init__(self) -> None:
        self.is_up: bool = True


class CPNode:
    """Node yang memprioritaskan Konsistensi (Consistency).

    Jika jaringan terputus, node ini akan menolak operasi baca dan tulis
    karena tidak bisa menjamin validitas data terbaru dengan node pasangannya.
    """

    def __init__(self, node_id: str, network: Network) -> None:
        self.node_id = node_id
        self.network = network
        self.storage: Dict[str, int] = {}

    def write(self, key: str, value: int, peer: "CPNode" = None) -> bool:
        """Menulis data ke node lokal dan mereplikasikannya ke peer jika jaringan aktif."""
        if not self.network.is_up:
            print(f"[{self.node_id} - CP] WRITE REJECTED: Jaringan down. Tidak bisa sinkronisasi.")
            return False

        self.storage[key] = value
        if peer:
            peer.storage[key] = value
        print(f"[{self.node_id} - CP] WRITE SUCCESS: {key}={value} (Tersinkronisasi ke seluruh cluster)")
        return True

    def read(self, key: str) -> int | None:
        """Membaca data berdasarkan key. Menolak jika jaringan sedang terputus."""
        if not self.network.is_up:
            print(f"[{self.node_id} - CP] READ REJECTED: Jaringan terputus. Tidak dapat menjamin data terbaru!")
            return None

        val = self.storage.get(key, None)
        print(f"[{self.node_id} - CP] READ SUCCESS: {key}={val}")
        return val


class APNode:
    """Node yang memprioritaskan Ketersediaan (Availability).

    Node ini akan selalu melayani permintaan baca dan tulis kapan pun,
    meskipun jaringan terputus dan data yang dikembalikan berpotensi usang (stale).
    """

    def __init__(self, node_id: str, network: Network) -> None:
        self.node_id = node_id
        self.network = network
        self.storage: Dict[str, int] = {}

    def write(self, key: str, value: int, peer: "APNode" = None) -> bool:
        """Menulis data ke node lokal. Tetap sukses walau gagal replikasi saat jaringan down."""
        self.storage[key] = value
        if not self.network.is_up:
            print(f"[{self.node_id} - AP] WRITE LOCAL SUCCESS: {key}={value} (Peringatan: Gagal replikasi ke peer)")
            return True

        if peer:
            peer.storage[key] = value
        print(f"[{self.node_id} - AP] WRITE SUCCESS: {key}={value} (Tersinkronisasi ke seluruh cluster)")
        return True

    def read(self, key: str) -> int | None:
        """Membaca data lokal. Memberikan peringatan jika jaringan terputus."""
        if not self.network.is_up:
            print(f"[{self.node_id} - AP] READ WARNING: Jaringan terputus. Mengembalikan data lokal yang mungkin basi (stale).")

        val = self.storage.get(key, None)
        print(f"[{self.node_id} - AP] READ SUCCESS: {key}={val}")
        return val


def run_scenario() -> None:
    """Menjalankan simulasi siklus hidup cluster dalam menghadapi Network Partition."""
    network = Network()

    # Inisialisasi sepasang node untuk masing-masing mazhab CAP
    cp_node_a = CPNode("Node-A", network)
    cp_node_b = CPNode("Node-B", network)

    ap_node_a = APNode("Node-A", network)
    ap_node_b = APNode("Node-B", network)

    print("==================================================")
    print("FASE 1: Kondisi Jaringan Normal (Replikasi Berjalan)")
    print("==================================================")
    # Tulis data mula-mula melalui Node-A
    cp_node_a.write("saldo", 1000, cp_node_b)
    ap_node_a.write("saldo", 1000, ap_node_b)

    print("\n==================================================")
    print("FASE 2: Terjadi Network Partition (Jaringan Down)")
    print("==================================================")
    network.is_up = False
    print("-> Sistem mendeteksi gangguan komunikasi antarnode.")

    # Poin 3: Pemanggilan method read tepat setelah partisi jaringan terjadi
    print("\n[Eksekusi READ pada Node-B saat Jaringan Down]")
    cp_node_b.read("saldo")
    ap_node_b.read("saldo")

    print("\n==================================================")
    print("FASE 3: Operasi WRITE Baru Saat Terisolasi")
    print("==================================================")
    print("[Mencoba update data baru saldo=2500 melalui Node-A]")
    cp_node_a.write("saldo", 2500, cp_node_b)
    ap_node_a.write("saldo", 2500, ap_node_b)

    print("\n==================================================")
    print("FASE 4: Dampak Akhir pada Konsistensi Data")
    print("==================================================")
    print("[Eksekusi READ ulang pada Node-B untuk melihat perbedaan akhir]")
    cp_node_b.read("saldo")
    ap_node_b.read("saldo")


if __name__ == "__main__":
    run_scenario()