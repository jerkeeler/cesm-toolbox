from dataclasses import dataclass
from enum import Enum


class ProxySource(Enum):
    jiang_2020 = "jiang_2020"


class ProxyLocality(Enum):
    marine = "marine"
    terrestrial = "terrestrial"


@dataclass()
class ProxyData:
    lon: float
    lat: float
    d18o: float
    d18o_std: float
    sst: float
    sst_std: float
    glassy: bool
    source: ProxySource
    locality: ProxyLocality
    name: str
