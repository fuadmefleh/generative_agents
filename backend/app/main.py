"""Main FastAPI application for Generative Agents."""
import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

from app.models.agent import Agent, AgentState
from app.api.routes.websocket import get_connection_manager, broadcast_simulation_event
from app.api.routes import websocket
from app.models.models import Point2D
from app.models.simulation import Simulation
from app.models.world import WorldMap, WorldState, World, WorldMapTileType
from app.llm.client import OllamaClient

# Load environment variables
load_dotenv()


# Background task for simulation tick
async def simulation_loop(simulation: Simulation):
    """Background task that ticks the simulation every second."""
    logger.info("Starting simulation loop...")
    try:
        while True:
            try:
                # Tick the simulation in a thread pool to avoid blocking
                await asyncio.to_thread(simulation.tick)
                logger.debug("Simulation tick completed")
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Error in simulation tick: {e}")

            # Broadcast updated world state to all WebSocket clients
            try:
                connection_manager = get_connection_manager()
                for websocket in connection_manager.active_connections:
                    try:
                        await connection_manager.send_world_update(websocket, simulation, simulation.world)
                    except Exception as e:
                        logger.error(f"Error sending world map to client: {e}")
                
            except Exception as e:
                logger.error(f"Error broadcasting world update: {e}")

            # Wait 1 second before next tick
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Simulation loop cancelled")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Generative Agents application...")
    
    # Initialize database
    # await init_db()
    
    # Initialize vector store
    # vector_store = VectorStore()
    # await vector_store.initialize()
    # app.state.vector_store = vector_store
    
    # Initialize LLM client
    llm_client = OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gen-agent-model")
    )
    app.state.llm_client = llm_client
    
    simulation = Simulation.create( logger )

    # Store world in app.state
    app.state.simulation = simulation
    app.state.world = simulation.world
    world = app.state.world
    logger.info(f"World initialized: {world.map.name} ({world.map.dimensions.x}x{world.map.dimensions.y})")
    
    # Optionally broadcast world initialization (if any clients are connected)
    try:
        await broadcast_simulation_event("world_initialized", {
            "world_name": world.map.name,
            "dimensions": {
                "x": float(world.map.dimensions.x),
                "y": float(world.map.dimensions.y)
            }
        })
    except Exception as e:
        logger.warning(f"Could not broadcast world initialization: {e}")
    
    # Start the simulation loop as a background task
    simulation_task = asyncio.create_task(simulation_loop(simulation))
    app.state.simulation_task = simulation_task
    logger.info("Simulation loop started")
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop the simulation loop
    if hasattr(app.state, 'simulation_task'):
        logger.info("Stopping simulation loop...")
        app.state.simulation_task.cancel()
        try:
            await app.state.simulation_task
        except asyncio.CancelledError:
            logger.info("Simulation loop stopped")
    
    # Cleanup world
    if hasattr(app.state, 'world'):
        logger.info("Cleaning up world...")
        delattr(app.state, 'world')
    
    # Cleanup other resources
    # if hasattr(app.state, 'vector_store'):
    #     await app.state.vector_store.close()
    # await engine.dispose()
    
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Generative Agents API",
    description="Implementation of Generative Agents paper using FastAPI and Ollama",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:6010",
        "http://localhost:3000", 
        "http://localhost:59827"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
# app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Generative Agents API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/world")
async def get_world_info():
    """Get world information."""
    if not hasattr(app.state, 'world'):
        return {"error": "World not initialized"}
    
    world = app.state.world
    return {
        "name": world.map.name,
        "dimensions": {
            "x": float(world.map.dimensions.x),
            "y": float(world.map.dimensions.y)
        },
        "num_tiles": len(world.map.tiles),
        "num_landmarks": len(world.map.landmarks),
        "num_agents": len(world.state.agents),
        "current_time": world.state.time.isoformat()
    }


@app.post("/api/world/regenerate")
async def regenerate_world(width: int = 50, height: int = 50, num_rooms: int = 8, seed: int = None):
    """Regenerate the world map."""
    logger.info(f"Regenerating world: {width}x{height}, rooms={num_rooms}, seed={seed}")
    
    # Create new world
    world_map = WorldMap.create(
        name=f"Dungeon (Seed: {seed or 'random'})",
        width=width,
        height=height,
        num_rooms=num_rooms,
        seed=seed
    )
    world_state = WorldState(time=datetime.utcnow())
    new_world = World(map=world_map, state=world_state)
    
    # Update app state
    app.state.world = new_world
    
    # Broadcast update to all WebSocket clients
    await broadcast_simulation_event("world_regenerated", {
        "world_name": new_world.map.name,
        "dimensions": {
            "x": float(new_world.map.dimensions.x),
            "y": float(new_world.map.dimensions.y)
        }
    })
    
    # Also broadcast the full world map to connected clients
    connection_manager = get_connection_manager()
    for websocket in connection_manager.active_connections:
        try:
            await connection_manager.send_world_map(websocket, new_world)
        except Exception as e:
            logger.error(f"Error sending world map to client: {e}")
    
    logger.info(f"World regenerated: {new_world.map.name}")
    
    return {
        "message": "World regenerated successfully",
        "world_name": new_world.map.name,
        "dimensions": {
            "x": float(new_world.map.dimensions.x),
            "y": float(new_world.map.dimensions.y)
        }
    }


@app.post("/api/agents/spawn")
async def spawn_agent(name: str, x: float, y: float, description: str = ""):
    """Spawn a new agent at the specified location."""
    if not hasattr(app.state, 'world'):
        return {"error": "World not initialized"}
    
    world = app.state.world
    
    # Check if location is valid (not an obstacle)
    tile_id = f"{int(x)},{int(y)}"
    if tile_id not in world.map.tiles:
        return {"error": "Invalid location"}
    
    tile = world.map.tiles[tile_id]
    if tile.type == WorldMapTileType.OBSTACLE:
        return {"error": "Cannot spawn agent on obstacle"}
    
    # Create agent
    agent_id = f"agent_{len(world.state.agents) + 1}"
    agent_state = AgentState(
        location=Point2D(x=x, y=y),
        status="idle"
    )
    
    # You might want to create a full Agent object here instead
    # agent = Agent(agent_id=agent_id, name=name, ...)
    
    # Add to world
    world.state.agents[agent_id] = agent_state
    
    # Broadcast to WebSocket clients
    await broadcast_simulation_event("agent_spawned", {
        "agent_id": agent_id,
        "name": name,
        "location": {"x": x, "y": y},
        "description": description
    })
    
    logger.info(f"Agent spawned: {agent_id} at ({x}, {y})")
    
    return {
        "message": "Agent spawned successfully",
        "agent_id": agent_id,
        "name": name,
        "location": {"x": x, "y": y}
    }