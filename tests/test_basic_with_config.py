import os
import time
from thortils import (launch_controller,
                      convert_scene_to_grid_map)
from thortils.scene import SceneDataset
from thortils.utils.visual import GridMapVisualizer
from thortils.agent import thor_reachable_positions
from thortils.grid_map import GridMap
from thortils.basic_config import config

controller = launch_controller(config)
