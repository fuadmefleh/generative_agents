from typing import Optional, Dict, Any, Tuple
import random
from app.models.world import WorldMap, WorldMapTileType, WorldMapTile
from app.models.models import Point2D

import pytmx


def recompute_tmx_gid(gid, flipx=False, flipy=False, flipdiag=False):
    raw_gid = gid
    if flipx:
        raw_gid |= 0x80000000
    if flipy:
        raw_gid |= 0x40000000
    if flipdiag:
        raw_gid |= 0x20000000
    return raw_gid


def create_world(
    name: str,
    width: int,
    height: int,
    num_rooms: int = random.randint(10, 20),
    min_room_size: int = 3,
    max_room_size: int = 8,
    num_main_branches: int = random.randint(7, 12),
    num_sub_branches: int = 2,
    seed: Optional[int] = None
) -> 'WorldMap':
    """Factory method to create a dungeon with a looping main corridor, branches, sub-branches, and rooms."""
    if seed is not None:
        random.seed(seed)

    tiled_map = pytmx.TiledMap('./app/map/the_ville.tmx')
    objects_id_to_names_file = './app/map/game_object_blocks.csv' # id, location_name, accessible, name
    objects_id_to_names = {}
    with open(objects_id_to_names_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                obj_id = int(parts[0])
                location_name = parts[1]
                accessible = parts[2].lower() == 'true'
                name = parts[3]
                objects_id_to_names[obj_id] = {
                    'location_name': location_name,
                    'accessible': accessible,
                    'name': name
                }

    # Load arena mappings
    arena_id_to_names_file = './app/map/arena_blocks.csv' # id, world_name, main_location_name, sub_location_name
    arenas_id_to_names = {}
    with open(arena_id_to_names_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                arena_id = int(parts[0])
                world_name = parts[1]
                main_location_name = parts[2]
                sub_location_name = parts[3]
                arenas_id_to_names[arena_id] = {
                    'world_name': world_name,
                    'main_location_name': main_location_name,
                    'sub_location_name': sub_location_name
                }

    # Load sector mappings
    sector_id_to_names_file = './app/map/sector_blocks.csv' # id, world_name, main_location_name
    sectors_id_to_names = {}
    with open(sector_id_to_names_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                sector_id = int(parts[0])
                world_name = parts[1]
                main_location_name = parts[2]
                sectors_id_to_names[sector_id] = {
                    'world_name': world_name,
                    'main_location_name': main_location_name
                }

    world_objects = _load_objects_from_tmx(tiled_map, objects_id_to_names)
    arena_mappings = _load_arena_mappings_from_tmx(tiled_map, arenas_id_to_names)
    sector_mappings = _load_sector_mappings_from_tmx(tiled_map, sectors_id_to_names)

    arena_main_locations_to_positions = _map_arena_main_locations_to_positions(arena_mappings) # arena main location name to list of positions
    arena_main_locations_to_sub_locations = _map_arena_main_locations_to_sub_locations(arena_mappings)  # arena main location name to set of sub location names
    arena_sub_locations_to_positions = _map_arena_sub_locations_to_positions(arena_mappings)  # (Main Location, Sub Location) to positions

    sector_locations_to_positions = _map_sector_locations_to_positions(sector_mappings)

    game_objects_to_sectors = _map_game_objects__to_sectors(sector_mappings, world_objects)
    game_objects_to_arena_locations = _map_game_objects_to_arena_locations(arena_mappings, world_objects)

    #rint( game_objects__to_sectors )
    #print( game_objects_to_arena_locations )

    layer = tiled_map.layernames["Collisions"]
    # Start with all obstacles
    tiles = {}
    for x in range(tiled_map.width):
        for y in range(tiled_map.height):
            tile_id = f"{x},{y}"
            tiles[tile_id] = WorldMapTile(
                type=WorldMapTileType.LAND,
                location=Point2D(x=float(x), y=float(y))
            )
    

    # Build the collision and walkable tiles
    for x, y, image in layer.tiles():
        tile_id = f"{x},{y}"
        tiles[tile_id].type = WorldMapTileType.OBSTACLE
    
    return WorldMap(
        name=name,
        dimensions=Point2D(x=float(tiled_map.width), y=float(tiled_map.height)),
        tiles=tiles,
        tiled_map=tiled_map,
        world_objects=world_objects,
        arena_mappings=arena_mappings,
        sector_mappings=sector_mappings,
        arena_main_locations_to_positions=arena_main_locations_to_positions,
        arena_main_locations_to_sub_locations=arena_main_locations_to_sub_locations,
        arena_sub_locations_to_positions=arena_sub_locations_to_positions,
        sector_locations_to_positions=sector_locations_to_positions,
        game_objects_to_sectors=game_objects_to_sectors,
        game_objects_to_arena_locations=game_objects_to_arena_locations
    )

def _map_game_objects_to_arena_locations(arena_mappings: Dict[str, Any], world_objects: Dict[str, Any]) -> Dict[str, list[Dict[str, Any]]]:
    """Create a mapping from arena location names. mapping[main_location_name][sub_location_name] = [list of game objects]"""
    mapping = {}
    for tile_id, arena_info in arena_mappings.items():
        sub_location_name = arena_info.get("sub_location_name", None)
        if sub_location_name and tile_id in world_objects:
            main_location_name = arena_info.get("main_location_name", None)
            if main_location_name not in mapping:
                mapping[main_location_name] = {}
            if sub_location_name not in mapping[main_location_name]:
                mapping[main_location_name][sub_location_name] = []
            mapping[main_location_name][sub_location_name].append(world_objects[tile_id])
    return mapping

def _map_game_objects__to_sectors(sector_mappings: Dict[str, Any], world_objects: Dict[str, Any]) -> Dict[str, list[Dict[str, Any]]]:
    """Create a mapping from sector main location names to their game objects."""
    mapping = {}
    for tile_id, sector_info in sector_mappings.items():
        main_location_name = sector_info.get("main_location_name", None)
        if main_location_name and tile_id in world_objects:
            if main_location_name not in mapping:
                mapping[main_location_name] = []
            mapping[main_location_name].append(world_objects[tile_id])
    return mapping

def _map_sector_locations_to_positions(sector_mappings: Dict[str, Any]) -> Dict[str, list[Point2D]]:
    """Create a mapping from sector main location names to their positions."""
    mapping = {}
    for tile_id, sector_info in sector_mappings.items():
        main_location_name = sector_info.get("main_location_name", None)
        if main_location_name:
            if main_location_name not in mapping:
                mapping[main_location_name] = []
            x_str, y_str = tile_id.split(',')
            mapping[main_location_name].append(Point2D(x=float(x_str), y=float(y_str)))
    return mapping

def _map_arena_sub_locations_to_positions(arena_mappings: Dict[str, Any]) -> Dict[Tuple[str, str], list[Point2D]]:
    """Create a mapping from arena sub location names to their positions."""
    mapping = {}
    for tile_id, arena_info in arena_mappings.items():
        sub_location_name = arena_info.get("sub_location_name", None)
        if sub_location_name:
            main_location_name = arena_info.get("main_location_name", None)
            if main_location_name:
                key = (main_location_name, sub_location_name)
                if key not in mapping:
                    mapping[key] = []
            x_str, y_str = tile_id.split(',')
            mapping[key].append(Point2D(x=float(x_str), y=float(y_str)))
    return mapping

def _map_arena_main_locations_to_sub_locations(arena_mappings: Dict[str, Any]) -> Dict[str, set]:
    """Create a mapping from arena main location names to their sub location names."""
    mapping = {}
    for tile_id, arena_info in arena_mappings.items():
        main_location_name = arena_info.get("main_location_name", None)
        sub_location_name = arena_info.get("sub_location_name", None)
        if main_location_name and sub_location_name:
            if main_location_name not in mapping:
                mapping[main_location_name] = set()
            mapping[main_location_name].add(sub_location_name)
    return mapping

def _map_arena_main_locations_to_positions(arena_mappings: Dict[str, Any]) -> Dict[str, list[Point2D]]:
    """Create a mapping from arena main location names to their positions."""
    mapping = {}
    for tile_id, arena_info in arena_mappings.items():
        main_location_name = arena_info.get("main_location_name", None)
        if main_location_name:
            if main_location_name not in mapping:
                mapping[main_location_name] = []
            x_str, y_str = tile_id.split(',')
            mapping[main_location_name].append(Point2D(x=float(x_str), y=float(y_str)))
    return mapping

def _load_sector_mappings_from_tmx(tiled_map: pytmx.TiledMap, sectors_id_to_names: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Load sector mappings from the Tiled map's sector layer."""
    sector_layer = tiled_map.layernames.get("Sector Blocks", None)
    sector_mappings = {}
    if sector_layer:
        for y in range(sector_layer.height):
            for x in range(sector_layer.width):
                gid = sector_layer.data[y][x]
                if gid == 0:
                    continue
                raw_tmx_gid = tiled_map.tiledgidmap[gid]
                tile_id = f"{x},{y}"

                # print( sectors_id_to_names.get(gid, None) )
                sector_mappings[tile_id] = {
                    "sector_id": f"sector_{x}_{y}",
                    "location": {
                        "x": float(x),
                        "y": float(y)
                    },
                    "main_location_name": sectors_id_to_names.get(raw_tmx_gid, {}).get("main_location_name", "Unknown"),
                    "world_name": sectors_id_to_names.get(raw_tmx_gid, {}).get("world_name", "Unknown"),
                }
    return sector_mappings


def _load_objects_from_tmx(tiled_map: pytmx.TiledMap, objects_id_to_names: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Load objects from the Tiled map's object layer."""
    objects_layer = tiled_map.layernames["Object Interaction Blocks"]
    world_objects = {}
    for y in range(objects_layer.height):
        for x in range(objects_layer.width):
            gid = objects_layer.data[y][x]
            if gid == 0:
                continue
            raw_tmx_gid = tiled_map.tiledgidmap[gid]

            # raw tmx gid is the id for the object
            obj_properties = objects_id_to_names.get(raw_tmx_gid, None)
            if obj_properties:
                tile_id = f"{x},{y}"
                world_objects[tile_id] = {
                    "id": f"object_{x}_{y}",
                    "type": "game_object",
                    "name": obj_properties["name"],
                    "location": {
                        "x": float(x),
                        "y": float(y)
                    },
                    "properties": obj_properties
                }
    return world_objects


def _load_arena_mappings_from_tmx(tiled_map: pytmx.TiledMap, arenas_id_to_names: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Load arena mappings from the Tiled map's arena layer."""
    arena_layer = tiled_map.layernames.get("Arena Blocks", None)
    arena_mappings = {}
    if arena_layer:
        for y in range(arena_layer.height):
            for x in range(arena_layer.width):
                gid = arena_layer.data[y][x]
                if gid == 0:
                    continue
                raw_tmx_gid = tiled_map.tiledgidmap[gid]
                tile_id = f"{x},{y}"

                # print( arenas_id_to_names.get(gid, None) )
                arena_mappings[tile_id] = {
                    "arena_id": f"arena_{x}_{y}",
                    "location": {
                        "x": float(x),
                        "y": float(y)
                    },
                    "main_location_name": arenas_id_to_names.get(raw_tmx_gid, {}).get("main_location_name", "Unknown"),
                    "sub_location_name": arenas_id_to_names.get(raw_tmx_gid, {}).get("sub_location_name", "Unknown"),
                    "world_name": arenas_id_to_names.get(raw_tmx_gid, {}).get("world_name", "Unknown"),
                }
    return arena_mappings