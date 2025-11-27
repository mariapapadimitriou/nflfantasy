"""
Data Source Interface - Abstract class for easy replacement of data sources
Implement this interface to swap out data providers
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class NFLDataSource(ABC):
    """Abstract base class for NFL data sources"""

    @abstractmethod
    def load_player_stats(self, seasons: List[int]) -> pd.DataFrame:
        """Load player statistics for given seasons"""
        pass

    @abstractmethod
    def load_pbp(self, seasons: List[int]) -> pd.DataFrame:
        """Load play-by-play data for given seasons"""
        pass

    @abstractmethod
    def load_rosters(self, seasons: List[int]) -> pd.DataFrame:
        """Load roster data for given seasons"""
        pass

    @abstractmethod
    def load_schedules(self, seasons: List[int]) -> pd.DataFrame:
        """Load schedule data for given seasons"""
        pass

    @abstractmethod
    def load_teams(self) -> pd.DataFrame:
        """Load team master list"""
        pass

    @abstractmethod
    def load_injuries(self) -> pd.DataFrame:
        """Load team master list"""
        pass


class NFLReadPyDataSource(NFLDataSource):
    """Implementation using nflreadpy library"""

    def __init__(self):
        try:
            import nflreadpy as nfl

            self.nfl = nfl
        except ImportError:
            raise ImportError(
                "nflreadpy is required. Install with: pip install nflreadpy"
            )

    def load_player_stats(self, seasons: List[int]) -> pd.DataFrame:
        return self.nfl.load_player_stats(seasons=seasons).to_pandas()

    def load_pbp(self, seasons: List[int]) -> pd.DataFrame:
        return self.nfl.load_pbp(seasons=seasons).to_pandas()

    def load_rosters(self, seasons: List[int]) -> pd.DataFrame:
        return self.nfl.load_rosters(seasons=seasons).to_pandas()

    def load_schedules(self, seasons: List[int]) -> pd.DataFrame:
        return self.nfl.load_schedules(seasons=seasons).to_pandas()

    def load_teams(self) -> pd.DataFrame:
        return self.nfl.load_teams().to_pandas()

    def load_injuries(self) -> pd.DataFrame:
        return self.nfl.load_injuries().to_pandas()


# Factory function to get data source
def get_data_source(source_type: str = "nflreadpy") -> NFLDataSource:
    """
    Factory function to get the appropriate data source

    Args:
        source_type: Type of data source ('nflreadpy', 'csv', 'api', etc.)

    Returns:
        NFLDataSource instance
    """
    if source_type == "nflreadpy":
        return NFLReadPyDataSource()
    else:
        raise ValueError(f"Unknown data source type: {source_type}")

