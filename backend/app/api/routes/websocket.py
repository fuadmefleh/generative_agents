"""WebSocket routes for real-time agent updates."""
import asyncio
import json
from typing import Dict, Set
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from loguru import logger

router = APIRouter()

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.agent_subscriptions: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        print( 'ehere')
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        
        # Remove from all agent subscriptions
        for agent_id in list(self.agent_subscriptions.keys()):
            self.agent_subscriptions[agent_id].discard(websocket)
            if not self.agent_subscriptions[agent_id]:
                del self.agent_subscriptions[agent_id]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def subscribe_to_agent(self, websocket: WebSocket, agent_id: str):
        """Subscribe a WebSocket to updates for a specific agent."""
        if agent_id not in self.agent_subscriptions:
            self.agent_subscriptions[agent_id] = set()
        self.agent_subscriptions[agent_id].add(websocket)
        logger.info(f"WebSocket subscribed to agent: {agent_id}")
    
    async def unsubscribe_from_agent(self, websocket: WebSocket, agent_id: str):
        """Unsubscribe a WebSocket from agent updates."""
        if agent_id in self.agent_subscriptions:
            self.agent_subscriptions[agent_id].discard(websocket)
            if not self.agent_subscriptions[agent_id]:
                del self.agent_subscriptions[agent_id]
        logger.info(f"WebSocket unsubscribed from agent: {agent_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_agent_subscribers(self, agent_id: str, message: str):
        """Broadcast a message to all WebSockets subscribed to a specific agent."""
        if agent_id in self.agent_subscriptions:
            disconnected = set()
            for connection in self.agent_subscriptions[agent_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to agent subscriber: {e}")
                    disconnected.add(connection)
            
            # Clean up disconnected connections
            for connection in disconnected:
                self.disconnect(connection)
    
    async def send_world_map(self, websocket: WebSocket, world):
        """Send the world map data to a specific WebSocket."""
        if not world or not world.map:
            await self.send_personal_message(
                json.dumps({
                    "type": "error",
                    "data": {"message": "World map not initialized"}
                }),
                websocket
            )
            return
        
        try:
            # Convert world map to JSON-serializable format
            world_map_data = {
                "type": "world_map",
                "data": {
                    "name": world.map.name,
                    "dimensions": {
                        "x": float(world.map.dimensions.x),
                        "y": float(world.map.dimensions.y)
                    },
                    "landmarks": {
                        name: {
                            "x": float(loc.x),
                            "y": float(loc.y)
                        }
                        for name, loc in world.map.landmarks.items()
                    },
                    "tiles": {
                        tile_id: {
                            "type": tile.type.name if hasattr(tile.type, 'name') else str(tile.type),
                            "location": {
                                "x": float(tile.location.x),
                                "y": float(tile.location.y)
                            }
                        }
                        for tile_id, tile in world.map.tiles.items()
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.send_personal_message(json.dumps(world_map_data), websocket)
            logger.info(f"Sent world map to WebSocket: {world.map.name}")
            
        except Exception as e:
            logger.error(f"Error sending world map: {e}")
            await self.send_personal_message(
                json.dumps({
                    "type": "error",
                    "data": {"message": f"Failed to send world map: {str(e)}"}
                }),
                websocket
            )
    
    async def send_world_update(self, websocket: WebSocket, simulation, world):
        """Send the updated world state to a specific WebSocket."""
        if not world or not world.state:
            await self.send_personal_message(
                json.dumps({
                    "type": "error",
                    "data": {"message": "World state not initialized"}
                }),
                websocket
            )
            return

        try:
            agents = []
            for agent_id, agent_state in world.state.agents.items():
                # Get the full agent from simulation if available
                full_agent = None
                if hasattr(simulation, 'agents') and agent_id in simulation.agents:
                    full_agent = simulation.agents[agent_id]
                
                # Handle current_action safely - it can be None
                current_action_data = None
                if hasattr(agent_state, 'current_action') and agent_state.current_action is not None:
                    current_action_data = {
                        "action_type": agent_state.current_action.action_type.value if hasattr(agent_state.current_action.action_type, 'value') else str(agent_state.current_action.action_type),
                        "reasoning": agent_state.current_action.reasoning,
                        "target_location": agent_state.current_action.target_location,
                        "target_object": agent_state.current_action.target_object,
                        "target_agent": agent_state.current_action.target_agent,
                        "inner_thought": agent_state.current_action.inner_thought
                    }
                
                # Build agent data with enhanced information
                agent_data = {
                    "agent_id": agent_id,
                    "name": agent_state.name if hasattr(agent_state, 'name') else agent_id,
                    "location": {
                        "x": float(agent_state.location.x),
                        "y": float(agent_state.location.y)
                    },
                    "current_action": current_action_data,
                    "status": agent_state.status if hasattr(agent_state, 'status') else "active",
                    "inner_thought": agent_state.current_inner_thought if hasattr(agent_state, 'current_inner_thought') else "",
                    "is_sleeping": agent_state.is_sleeping if hasattr(agent_state, 'is_sleeping') else False,
                    "objects_in_use": agent_state.objects_in_use if hasattr(agent_state, 'objects_in_use') else []
                }
                
                # Add needs and emotions data if full agent is available
                if full_agent:
                    agent_data["needs"] = {
                        "energy": float(full_agent.needs.energy),
                        "hunger": float(full_agent.needs.hunger),
                        "bladder": float(full_agent.needs.bladder),
                        "hygiene": float(full_agent.needs.hygiene),
                        "social": float(full_agent.needs.social),
                        "fun": float(full_agent.needs.fun),
                        "comfort": float(full_agent.needs.comfort),
                        "fulfillment": float(full_agent.needs.fulfillment)
                    }
                    agent_data["emotions"] = {
                        "primary_emotion": full_agent.emotions.primary_emotion.value,
                        "emotion_intensity": float(full_agent.emotions.emotion_intensity),
                        "mood_baseline": float(full_agent.emotions.mood_baseline),
                        "stress_level": float(full_agent.emotions.stress_level)
                    }
                    agent_data["wellbeing"] = full_agent.needs.get_wellbeing_score()
                
                agents.append(agent_data)
            world_data = {
                "type": "world_update",
                "data": {
                    "time": world.state.time.isoformat(),
                    "num_agents": len(world.state.agents),
                    "agents": agents
                },
                "timestamp": datetime.now().isoformat(),
                "world_time": simulation.world_time.isoformat()
            }

            await self.send_personal_message(json.dumps(world_data), websocket)
            logger.info(f"Sent world update to WebSocket: {world.map.name}")

        except Exception as e:
            logger.error(f"Error sending world update: {e}")
            await self.send_personal_message(
                json.dumps({
                    "type": "error",
                    "data": {"message": f"Failed to send world update: {str(e)}"}
                }),
                websocket
            )

    async def send_agents_list(self, websocket: WebSocket, world):
        """Send the current agents list to a specific WebSocket."""
        if not world or not world.state:
            await self.send_personal_message(
                json.dumps({
                    "type": "error",
                    "data": {"message": "World state not initialized"}
                }),
                websocket
            )
            return
        
        try:
            agents_data = {
                "type": "agents_list",
                "data": {
                    "agents": [
                        {
                            "agent_id": agent_id,
                            "name": agent.name if hasattr(agent, 'name') else agent_id,
                            "description": agent.description if hasattr(agent, 'description') else "",
                            "traits": agent.traits if hasattr(agent, 'traits') else [],
                            "location": {
                                "x": float(agent.location.x),
                                "y": float(agent.location.y)
                            },
                            "status": agent.status if hasattr(agent, 'status') else "active"
                            #"state": agent.model_dump() if hasattr(agent, 'model_dump') else {}
                        }
                        for agent_id, agent in world.state.agents.items()
                    ]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await self.send_personal_message(json.dumps(agents_data), websocket)
            logger.info(f"Sent agents list to WebSocket: {len(world.state.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error sending agents list: {e}")
            await self.send_personal_message(
                json.dumps({
                    "type": "error",
                    "data": {"message": f"Failed to send agents list: {str(e)}"}
                }),
                websocket
            )


# Create connection manager instance
manager = ConnectionManager()


@router.websocket("")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    # Access app.state through websocket.app
    world = getattr(websocket.app.state, 'world', None)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            message_type = message.get("type")
            
            if message_type == "ping":
                # Respond to ping
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            
            elif message_type == "get_world_map":
                # Send world map data
                await manager.send_world_map(websocket, world)
            
            elif message_type == "get_agents":
                # Send current agents list
                await manager.send_agents_list(websocket, world)
            
            elif message_type == "subscribe":
                # Subscribe to agent updates
                agent_id = message.get("agent_id")
                if agent_id:
                    await manager.subscribe_to_agent(websocket, agent_id)
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "subscribed",
                            "agent_id": agent_id,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                else:
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "data": {"message": "Missing agent_id"}}),
                        websocket
                    )
            
            elif message_type == "unsubscribe":
                # Unsubscribe from agent updates
                agent_id = message.get("agent_id")
                if agent_id:
                    await manager.unsubscribe_from_agent(websocket, agent_id)
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "unsubscribed",
                            "agent_id": agent_id,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                else:
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "data": {"message": "Missing agent_id"}}),
                        websocket
                    )
            
            else:
                # Unknown message type
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "data": {"message": f"Unknown message type: {message_type}"}
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_agent_update(agent_id: str, update_type: str, data: dict):
    """Broadcast an agent update to all subscribers.
    
    Args:
        agent_id: ID of the agent
        update_type: Type of update (e.g., "location", "action", "dialogue")
        data: Update data
    """
    message = json.dumps({
        "type": "agent_update",
        "agent_id": agent_id,
        "update_type": update_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    })
    
    await manager.broadcast_to_agent_subscribers(agent_id, message)


async def broadcast_simulation_event(event_type: str, data: dict):
    """Broadcast a simulation-wide event.
    
    Args:
        event_type: Type of event (e.g., "time_update", "new_agent", "agent_removed")
        data: Event data
    """
    message = json.dumps({
        "type": "simulation_event",
        "event_type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    })
    
    await manager.broadcast(message)


async def broadcast_world_map_update(app):
    """Broadcast world map update to all connected clients.
    
    Args:
        app: FastAPI app instance with state.world
    """
    world = getattr(app.state, 'world', None)
    if world and world.map:
        for websocket in manager.active_connections:
            try:
                await manager.send_world_map(websocket, world)
            except Exception as e:
                logger.error(f"Error broadcasting world map update: {e}")


# Export manager for use in other modules
def get_connection_manager():
    """Get the WebSocket connection manager."""
    return manager