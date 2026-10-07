from src.data.adapters.base import DatasetAdapter
from src.data.adapters.demo import DemoAdapter
from src.data.adapters.generic_csv import GenericCycleCSVAdapter
from src.data.adapters.nasa import NasaAdapter

ADAPTERS = {"nasa": NasaAdapter, "demo": DemoAdapter}
__all__ = ["DatasetAdapter", "DemoAdapter", "GenericCycleCSVAdapter", "NasaAdapter", "ADAPTERS"]
