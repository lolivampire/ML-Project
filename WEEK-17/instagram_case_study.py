"""instagram_case_study.py

Week 17 Day 05 — FeedService: simulasi hybrid fan-out dengan Parameter Injection.
Menghilangkan konstanta global untuk mendukung pengujian yang terisolasi 
dan fleksibilitas multi-konteks.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    """Entitas pengguna di dalam sistem.
    
    Kelas ini sekarang murni menjadi struktur data (Anemic/Clean Data Carrier) 
    dan tidak memegang kebijakan bisnis global tentang kriteria selebriti.
    """
    user_id: int
    username: str
    follower_count: int = 0


class FeedCacheRepository:
    """Lapisan Penyimpanan Cache."""

    def __init__(self) -> None:
        self._write_count: int = 0

    def write_to_feed_cache(self, follower_id: int, author_id: int) -> None:
        """Simulasi I/O penulisan cache feed."""
        self._write_count += 1

    @property
    def write_count(self) -> int:
        """Mengembalikan total hitungan operasi penulisan."""
        return self._write_count


class FeedService:
    """Lapisan Bisnis (Domain Service).
    
    Kebijakan ambang batas selebriti sekarang di-inject saat inisialisasi,
    membuat service ini dinamis dan mudah diuji.
    """

    def __init__(self, cache_repo: FeedCacheRepository, celebrity_threshold: int = 1_000_000) -> None:
        """Inisialisasi FeedService dengan Dependency Injection.
        
        Args:
            cache_repo: Instance dari FeedCacheRepository.
            celebrity_threshold: Ambang batas jumlah pengikut untuk strategi fan-out.
        """
        self._cache_repo = cache_repo
        self._celebrity_threshold = celebrity_threshold

    def _is_celebrity(self, user: User) -> bool:
        """Metode internal untuk mengevaluasi status selebriti berdasarkan konfigurasi instance.
        
        Rekomendasi: Logika ini berada di Service karena status 'selebriti' untuk fitur feed
        adalah aturan operasional infrastruktur, bukan sifat intrinsik dari entitas User.
        """
        return user.follower_count >= self._celebrity_threshold

    def handle_new_post(self, author: User, follower_ids: Sequence[int]) -> str:
        """Mendistribusikan postingan berdasarkan evaluasi ambang batas tingkat instance."""
        if self._is_celebrity(author):
            return self._handle_celebrity_post(author, follower_ids)
        return self._handle_regular_post(author, follower_ids)

    def count_fanout_cost(self, author: User, follower_count: int) -> int:
        """Menghitung estimasi operasi penulisan berdasarkan ambang batas tingkat instance."""
        if self._is_celebrity(author):
            return 0
        return follower_count

    def _handle_regular_post(self, author: User, follower_ids: Sequence[int]) -> str:
        for follower_id in follower_ids:
            self._cache_repo.write_to_feed_cache(follower_id, author.user_id)
        return f"Pushed to {len(follower_ids)} follower feeds (regular)"

    def _handle_celebrity_post(self, author: User, follower_ids: Sequence[int]) -> str:
        return f"Skipped push for {len(follower_ids)} followers (celebrity — pull on read)"


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Skenario 1: Penggunaan Produksi Normal (Threshold = 1.000.000)
    # -----------------------------------------------------------------
    repo_prod = FeedCacheRepository()
    service_prod = FeedService(cache_repo=repo_prod, celebrity_threshold=1_000_000)

    budi = User(user_id=1, username="budi", follower_count=500)
    print("[PROD]", service_prod.handle_new_post(budi, follower_ids=range(500)))

    # -----------------------------------------------------------------
    # Skenario 2: Simulasi Unit Testing Ringan (Threshold Di-inject = 3)
    # -----------------------------------------------------------------
    # Kita bisa menguji jalur kode "Celebrity" tanpa perlu membuat loop 1 juta data!
    repo_test = FeedCacheRepository()
    service_test = FeedService(cache_repo=repo_test, celebrity_threshold=3)

    micro_influencer = User(user_id=9, username="micro_seleb", follower_count=5)
    
    # Karena threshold di-inject bernilai 3, maka user dengan 5 follower langsung dianggap selebriti
    print("[TEST]", service_test.handle_new_post(micro_influencer, follower_ids=range(5)))
    print("[TEST] Fanout cost:", service_test.count_fanout_cost(micro_influencer, 5))