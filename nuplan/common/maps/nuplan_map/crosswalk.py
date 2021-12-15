import pandas as pd
from nuplan.common.maps.abstract_map_objects import Crosswalk
from nuplan.common.maps.maps_datatypes import VectorLayer
from nuplan.common.maps.nuplan_map.utils import get_element_with_fid
from shapely.geometry import Polygon


class NuPlanCrosswalk(Crosswalk):
    def __init__(self, crosswalk_id: str, crosswalks_df: VectorLayer) -> None:
        super().__init__(crosswalk_id)
        self._crosswalks_df = crosswalks_df
        self._crosswalk = None

    @property
    def polygon(self) -> Polygon:
        """ Inherited from superclass """
        return self._get_crosswalk().geometry

    def _get_crosswalk(self) -> pd.Series:
        """
        Gets the series from the crosswalks dataframe containing crosswalk's id
        :return: the respective series from the crosswalks dataframe
        """
        if self._crosswalk is None:
            self._crosswalk = get_element_with_fid(self._crosswalks_df, "fid", self.id)
        return self._crosswalk
