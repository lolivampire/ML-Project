"""
hello_clean.py
Week 01 - Day 01: Python Best Practices Demo

Demonstrasi: type hints, docstring, naming convention,
dan fungsi yang punya satu tanggung jawab.

"""

import numpy as np

# ── CONSTANTS ────────────────────────────────────────────────
SAMPLE_DATA: list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ── FUNCTIONS ────────────────────────────────────────────────

def calculate_mean(data: list[float]) -> float:
    """Calculate the mean of a list of numbers.

    Args:
        data (list[float]): A list of numbers.

    Returns:
        float: The mean of the numbers.
    """
    if not data:
        raise ValueError("Data list cannot be empty.")
    
    total = sum(data)
    count = len(data)
    
    mean = total / count
    return mean

def calculate_range(data: list[float]) -> float:
    """ Menghitung selisih antara nilai maksimum dan minimum dalam sebuah daftar angka. """
    if not data:
        raise ValueError("Data list cannot be empty.")  
    max_value = max(data)
    min_value = min(data)

    return max_value - min_value   

def calculate_median(data: list[float]) -> float:
    """Menghitung nilai tengah dari sebuah daftar angka."""
    
    #MANUAL
    if not data:
        raise ValueError("Data list cannot be empty.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]
    

    #PAKAI NUMPY
def calculate_median_numpy(data: list[float]) -> float:
    if not data:
        raise ValueError("Data list cannot be empty.")
    return float(np.median(data))


def summarize_data(data: list[float]) -> dict:
    """Menyediakan ringkasan statistik dasar untuk sebuah daftar angka ."""
    if not data:
        raise ValueError("Data list cannot be empty.")
    return {
        "mean": calculate_mean(data),
        "range": calculate_range(data),
        "median": calculate_median_numpy(data),
        "min": min(data),
        "max": max(data)
    }


# ── MAIN EXECUTION ────────────────────────────────────────────────

def main() -> None:
    """Main function to execute the data analysis."""
    try:
        stats = summarize_data(SAMPLE_DATA)
    except ValueError as e:
        print(f"Error: {e}")
        return
    print("=" * 35)
    print("Data Analysis Summary")
    print("=" * 35)
    for key, value in stats.items():
        print(f"{key:<8}: {value:.2f}")
    print("=" * 35)

if __name__ == "__main__":
    main()