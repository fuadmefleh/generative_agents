from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import random
from app.models.models import Point2D
from app.models.agent import AgentState

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class WorldMapTileType(str, Enum):
    """Types of tiles in the world map."""
    LAND = 1
    OBSTACLE = 0
    WATER = -1


class WorldMapTile(BaseModel):
    """Representation of a tile in the world map."""
    type: WorldMapTileType
    location: Point2D

class WorldState(BaseModel):
    """Overall state of the world."""
    time: datetime
    agents: Dict[str, AgentState] = Field(default_factory=dict)
    objects: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Object ID to properties mapping

class WorldMap(BaseModel):
    """Representation of the world map."""
    name: str
    dimensions: Point2D  # Width and Height
    landmarks: Dict[str, Point2D] = Field(default_factory=dict)  # Landmark name to location mapping
    tiles: Dict[str, WorldMapTile] = Field(default_factory=dict)  # Tile ID is a Point2D to tile mapping
    world_objects: Dict[str, Any] = Field(default_factory=dict)  # Tile ID to object properties mapping
    arena_mappings: Dict[str, Any] = Field(default_factory=dict)  # Tile ID to arena properties mapping
    sector_mappings: Dict[str, Any] = Field(default_factory=dict)  # Tile ID to sector properties mapping

    arena_main_locations_to_positions: Dict[str, list[Point2D]]
    arena_main_locations_to_sub_locations: Dict[str, set]
    
    arena_sub_locations_to_positions: Dict[Tuple[str, str], list[Point2D]] # (Main Location, Sub Location) to positions

    sector_locations_to_positions: Dict[str, list[Point2D]]
    game_objects_to_sectors: Dict[str, list[Dict[str, Any]]]
    game_objects_to_arena_locations: Dict[str,  Dict[str, list[Dict[str, Any]]]]
    tiled_map: Any

    
    

    @staticmethod
    def _carve_tile(tiles: Dict[str, WorldMapTile], x: int, y: int) -> None:
        """Helper to carve out a single tile."""
        tile_id = f"{x},{y}"
        if tile_id in tiles:
            tiles[tile_id].type = WorldMapTileType.LAND

    @staticmethod
    def _rooms_overlap(room1: Tuple[int, int, int, int], 
                    room2: Tuple[int, int, int, int]) -> bool:
        """Check if two rooms overlap (with 1 tile padding)."""
        x1, y1, w1, h1 = room1
        x2, y2, w2, h2 = room2
        
        return not (x1 + w1 + 1 < x2 or x2 + w2 + 1 < x1 or
                    y1 + h1 + 1 < y2 or y2 + h2 + 1 < y1)

class World(BaseModel):
    """Comprehensive world model."""
    state: WorldState
    map: WorldMap


    def get_free_tiles(self) -> Dict[str, WorldMapTile]:
        """Return an array of tiles that are free (LAND type)."""
        return [tile for tile in self.map.tiles.values() if tile.type == WorldMapTileType.LAND]
    
    def get_map_of_sub_location_to_positions_for_main_location(self, main_location_name: str) -> Dict[str, list[Point2D]]:
        """Get a mapping of sub location names to their positions for a given main location."""
        mapping = {}
        for (main_loc, sub_loc), positions in self.map.arena_sub_locations_to_positions.items():
            if main_loc == main_location_name:
                mapping[sub_loc] = positions
        return mapping
    
    def get_sub_locations_positions_for_main_location(self, main_location_name: str) -> list[Point2D]:
        """Get all positions for a given main location across all its sub locations."""
        positions = []
        for (main_loc, sub_loc), pos_list in self.map.arena_sub_locations_to_positions.items():
            if main_loc == main_location_name:
                positions.extend(pos_list)
        return positions
    
    def get_sub_location_name_for_main_location_and_position(self, main_location_name: str, position: Point2D) -> Optional[str]:
        """Get the sub location name for a given main location and position."""
        for (main_loc, sub_loc), positions in self.map.arena_sub_locations_to_positions.items():
            if main_loc == main_location_name:
                for pos in positions:
                    if pos.x == position.x and pos.y == position.y:
                        return sub_loc
        return None
    
    def get_free_tiles_in_main_sector(self, main_location: str) -> Dict[str, WorldMapTile]:
        """Return an array of free tiles in the specified main sector."""
        free_tiles = self.get_free_tiles()
        sector_tiles = []
        for tile in free_tiles:
            tile_id = f"{int(tile.location.x)},{int(tile.location.y)}"
            if tile_id in self.map.sector_mappings:
                sector_info = self.map.sector_mappings[tile_id]
                if sector_info.get("main_location_name") == main_location:
                    sector_tiles.append(tile)
        return sector_tiles
    
    def get_objects_in_sector(self, sector_name: str) -> List[Dict[str, Any]]:
        """Return a list of objects located in the specified sector."""
        objects_in_sector = []
        
        return self.map.game_objects_to_arena_locations[sector_name] if sector_name in self.map.game_objects_to_arena_locations else []
    

        for sector_name, arena_info in self.map.game_objects_to_arena_locations.items():
            for sub_location_name, info in arena_info.items():
                for obj in info:
                    tile_id = obj.get("tile_id")
                    if tile_id in self.map.sector_mappings:
                        sector_info = self.map.sector_mappings[tile_id]
                        if sector_info.get("sector_name") == sector_name:
                            objects_in_sector.append(obj)
        return objects_in_sector
           

    def find_path(self, start: Point2D, end: Point2D) -> Optional[list[Point2D]]:
        """Find a path from start to end using A* algorithm."""
        free_tiles = self.get_free_tiles()
        
        # Create a grid for pathfinding
        width = int(self.map.dimensions.x)
        height = int(self.map.dimensions.y)
        matrix = []
        for y in range(height):
            row = []
            for x in range(width):
                tile_id = f"{x},{y}"
                if tile_id in self.map.tiles and self.map.tiles[tile_id].type == WorldMapTileType.LAND:
                    row.append(1)  # Walkable
                else:
                    row.append(0)  # Not walkable
            matrix.append(row)
        
        grid = Grid(matrix=matrix)
        start_node = grid.node(int(start.x), int(start.y))
        end_node = grid.node(int(end.x), int(end.y))
        
        finder = AStarFinder()
        path, _ = finder.find_path(start_node, end_node, grid)
        
        if not path:
            return None
        
        return [Point2D(x=float(x), y=float(y)) for x, y in path]
    
    def get_all_main_locations(self) -> List[str]:
        """Get a list of all main locations in the world."""
        return list(self.map.arena_main_locations_to_positions.keys())