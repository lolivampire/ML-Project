"""
instagram_case_study.py
W17D05 - System Design Case Study: Instagram-scale

Simulasi strategi hybrid fan-out untuk sistem feed berskala besar:
- User biasa (follower sedikit)    -> Fan-out on WRITE (Push model)
- User selebriti (follower banyak) -> Fan-out on READ (Pull model)

Refactoring ini mengoptimalkan struktur data, mengenkapsulasi state,
dan menambahkan metrik untuk menghitung biaya fan-out (write cost).
"""

from collections import defaultdict
from dataclasses import dataclass, field


# ── CONSTANTS ────────────────────────────────────────────────
CELEBRITY_THRESHOLD: int = 10_000


# ── DOMAIN MODELS ────────────────────────────────────────────
@dataclass
class User:
    """Model data untuk merepresentasikan seorang pengguna."""
    user_id: str
    followers: list[str] = field(default_factory=list)

    @property
    def is_celebrity(self) -> bool:
        """Menentukan status selebriti berdasarkan ambang batas follower."""
        return len(self.followers) >= CELEBRITY_THRESHOLD


# ── CORE SERVICE ─────────────────────────────────────────────
class FeedService:
    """
    Layanan inti untuk mengelola logika fan-out dan pembacaan feed.
    Kelas ini mengenkapsulasi semua state database.
    """

    def __init__(self) -> None:
        self._pushed_feeds: dict[str, list[str]] = defaultdict(list)
        self._celebrity_posts: dict[str, list[str]] = defaultdict(list)

    @property
    def pushed_feeds(self) -> dict[str, list[str]]:
        """Mengakses feed yang sudah di-push (read-only) untuk keperluan metrik."""
        return self._pushed_feeds

    def create_post(self, author: User, post_id: str) -> None:
        """Memproses post baru dan mengeksekusi strategi fan-out yang sesuai."""
        if author.is_celebrity:
            self._handle_celebrity_post(author, post_id)
        else:
            self._handle_regular_post(author, post_id)

    def get_feed(self, user_id: str, following: list[str]) -> list[str]:
        """Mengonstruksi feed dengan menggabungkan data push dan pull."""
        pushed_posts = self._pushed_feeds.get(user_id, [])
        pulled_posts = []
        for followed_id in following:
            if followed_id in self._celebrity_posts:
                pulled_posts.extend(self._celebrity_posts[followed_id])
        return pushed_posts + pulled_posts

    # ── Private Methods ──────────────────────────────────────
    def _handle_regular_post(self, author: User, post_id: str) -> None:
        """Logika Fan-out on Write untuk user biasa."""
        for follower_id in author.followers:
            self._pushed_feeds[follower_id].append(post_id)
        print(f"[PUSH] {author.user_id} (follower: {len(author.followers)}) "
              f"-> Post '{post_id}' di-push ke semua feed follower.")

    def _handle_celebrity_post(self, author: User, post_id: str) -> None:
        """Logika Fan-out on Read untuk user selebriti."""
        self._celebrity_posts[author.user_id].append(post_id)
        print(f"[PULL] {author.user_id} adalah selebriti "
              f"(follower: {len(author.followers)}) -> Post '{post_id}' disimpan, tidak di-push.")


# ── METRICS & ANALYSIS ───────────────────────────────────────
def count_fanout_cost(feed_store: dict[str, list[str]]) -> dict[str, int]:
    """Hitung berapa banyak 'tulisan' (write operation) terjadi per user
    akibat fan-out on write.

    Args:
        feed_store: Dictionary berisi pemetaan user_id ke daftar post_id.

    Returns:
        Dictionary berisi pemetaan user_id ke total jumlah post yang diterima.
    """
    return {user_id: len(posts) for user_id, posts in feed_store.items()}


# ── SIMULATION RUNNER ────────────────────────────────────────
def main() -> None:
    """Fungsi eksekusi utama untuk mendemonstrasikan alur kerja dan metrik FeedService."""
    feed_service = FeedService()

    regular_user = User(user_id="budi", followers=["ani", "citra"])
    celebrity = User(user_id="artis_x", followers=[f"fan_{i}" for i in range(15_000)])

    print("=" * 60)
    print("Simulasi Hybrid Fan-out & Analisis Biaya")
    print("=" * 60)

    feed_service.create_post(regular_user, "post_001")
    feed_service.create_post(celebrity, "post_002")

    print("-" * 60)

    ani_following = ["budi", "artis_x"]
    ani_feed = feed_service.get_feed(user_id="ani", following=ani_following)
    print(f"Feed milik 'ani' (mengikuti: {ani_following}):")
    print(f"Result -> {ani_feed}")

    print("-" * 60)
    print("Analisis Biaya Fan-out (Write Cost per User):")
    fanout_cost = count_fanout_cost(feed_service.pushed_feeds)
    for user_id, cost in fanout_cost.items():
        print(f" -> User '{user_id}': menerima {cost} post (biaya write = {cost})")
    print("=" * 60)


if __name__ == "__main__":
    main()

# ── CATATAN ARSITEKTUR: CP vs AP untuk Redis Cache ───────────
# 100 miliar poin untuk pemilihan AP. Kenapa? Karena saat jaringan
# terputus, dunia tidak akan kiamat jika user hanya melihat postingan
# yang tertunda beberapa detik. Yang membuat user meradang adalah saat
# aplikasinya tidak bisa dibuka sama sekali (unavailable). Jadi jelas
# kita pilih availability -- lebih baik menampilkan data lama (stale)
# daripada menampilkan layar putih.