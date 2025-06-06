from ._download_tennis import fetch_atp_data
from ._download_data import fetch_csrank_data, fetch_higher_education_data
from ._download_movers import fetch_movers_data
from ._make_synthetic import fetch_synthetic_data


__all__ = [
    "fetch_atp_data",
    "fetch_csrank_data",
    "fetch_higher_education_data",
    "fetch_movers_data",
    "fetch_synthetic_data",
]
