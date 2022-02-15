from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProxySource(Enum):
    jiang_2020 = "jiang_2020"
    van_dijk_2020 = "van_dijk_2020"


class ProxyLocality(Enum):
    marine = "marine"
    terrestrial = "terrestrial"


@dataclass()
class ProxyData:
    lat: float
    d18o: float
    d18o_std: float
    source: ProxySource
    locality: ProxyLocality
    name: str
    lon: Optional[float] = None
    temp: Optional[float] = None
    temp_std: Optional[float] = None
    glassy: bool = False
