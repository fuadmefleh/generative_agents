// frontend/src/components/WorldMapCanvas.tsx
import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useWorldMap, useAgents, useSimulationStore } from '../store/simulationStore';
import { ThoughtBubble } from './ThoughtBubble';
import { AnimatePresence } from 'framer-motion';

interface Point2D {
  x: number;
  y: number;
}

const TILE_COLORS = {
  LAND: '#dcdcc8',
  OBSTACLE: '#282828',
  WATER: '#3264c8',
} as const;

const AGENT_COLOR = '#ff0000';
const LANDMARK_COLOR = '#00ff00';
const GRID_COLOR = '#646464';
const HOVER_COLOR = '#ffff00';
const SELECTED_AGENT_COLOR = '#ffff00';

// Agent status colors for visualization
const STATUS_COLORS = {
  sleeping: '#9d4edd',
  moving: '#06d6a0',
  acting: '#ffd60a',
  waiting: '#adb5bd',
  idle: '#6c757d'
} as const;

export const WorldMapCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const worldMap = useWorldMap();
  const agents = useAgents();
  const { selectAgent, selectedAgent } = useSimulationStore();
  
  const [tileSize, setTileSize] = useState(20);
  const [camera, setCamera] = useState({ x: 0, y: 0 });
  const [hoveredTile, setHoveredTile] = useState<Point2D | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<Point2D | null>(null);
  const [hoveredAgent, setHoveredAgent] = useState<string | null>(null);

  // Zoom controls
  const handleZoom = useCallback((delta: number, mouseX?: number, mouseY?: number) => {
    const newTileSize = Math.max(5, Math.min(100, tileSize + delta));
    
    if (mouseX !== undefined && mouseY !== undefined && worldMap) {
      // Zoom towards mouse position
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const rect = canvas.getBoundingClientRect();
      const x = mouseX - rect.left;
      const y = mouseY - rect.top;
      
      const worldX = x / tileSize + camera.x;
      const worldY = y / tileSize + camera.y;
      
      const newCameraX = worldX - x / newTileSize;
      const newCameraY = worldY - y / newTileSize;
      
      setCamera({ x: newCameraX, y: newCameraY });
    }
    
    setTileSize(newTileSize);
  }, [tileSize, camera, worldMap]);

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !worldMap) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Update hovered tile and agent
    const tileX = Math.floor(x / tileSize + camera.x);
    const tileY = Math.floor(y / tileSize + camera.y);
    
    if (tileX >= 0 && tileX < worldMap.dimensions.x && 
        tileY >= 0 && tileY < worldMap.dimensions.y) {
      setHoveredTile({ x: tileX, y: tileY });
    } else {
      setHoveredTile(null);
    }

    // Check if hovering over an agent
    const hoveredAgentId = Object.values(agents).find(
      agent => Math.floor(agent.location.x) === tileX && 
               Math.floor(agent.location.y) === tileY
    )?.agent_id || null;
    setHoveredAgent(hoveredAgentId);

    // Handle dragging
    if (isDragging && dragStart) {
      const dx = (e.clientX - dragStart.x) / tileSize;
      const dy = (e.clientY - dragStart.y) / tileSize;
      
      setCamera(prev => ({
        x: Math.max(0, prev.x - dx),
        y: Math.max(0, prev.y - dy),
      }));
      
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setDragStart(null);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -2 : 2;
    handleZoom(delta, e.clientX, e.clientY);
  };

  const handleClick = (e: React.MouseEvent) => {
    if (!worldMap) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const worldX = Math.floor(x / tileSize + camera.x);
    const worldY = Math.floor(y / tileSize + camera.y);
    
    // Check if clicked on an agent
    const clickedAgent = Object.values(agents).find(
      agent => Math.floor(agent.location.x) === worldX && 
               Math.floor(agent.location.y) === worldY
    );
    
    if (clickedAgent) {
      selectAgent(clickedAgent);
    } else {
      selectAgent(null);
    }
  };

  // Draw the map
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !worldMap) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Calculate visible tile range
    const startX = Math.max(0, Math.floor(camera.x));
    const startY = Math.max(0, Math.floor(camera.y));
    const endX = Math.min(
      worldMap.dimensions.x,
      Math.ceil(camera.x + canvas.width / tileSize) + 1
    );
    const endY = Math.min(
      worldMap.dimensions.y,
      Math.ceil(camera.y + canvas.height / tileSize) + 1
    );

    // Draw tiles
    for (let y = startY; y < endY; y++) {
      for (let x = startX; x < endX; x++) {
        const tileId = `${x},${y}`;
        const tile = worldMap.tiles[tileId];
        
        if (!tile) continue;

        const screenX = (x - camera.x) * tileSize;
        const screenY = (y - camera.y) * tileSize;

        // Draw tile
        ctx.fillStyle = TILE_COLORS[tile.type] || '#ff00ff';
        ctx.fillRect(screenX, screenY, tileSize, tileSize);

        // Draw grid (only if tiles are large enough)
        if (tileSize >= 10) {
          ctx.strokeStyle = GRID_COLOR;
          ctx.lineWidth = 1;
          ctx.strokeRect(screenX, screenY, tileSize, tileSize);
        }
      }
    }

    // Draw landmarks
    Object.entries(worldMap.landmarks || {}).forEach(([name, location]) => {
      const screenX = (location.x - camera.x) * tileSize;
      const screenY = (location.y - camera.y) * tileSize;
      
      if (screenX >= -tileSize && screenX < canvas.width &&
          screenY >= -tileSize && screenY < canvas.height) {
        ctx.fillStyle = LANDMARK_COLOR;
        ctx.beginPath();
        ctx.arc(
          screenX + tileSize / 2,
          screenY + tileSize / 2,
          tileSize / 3,
          0,
          Math.PI * 2
        );
        ctx.fill();
        
        // Draw landmark name
        if (tileSize >= 15) {
          ctx.fillStyle = '#ffffff';
          ctx.font = `${Math.floor(tileSize / 2)}px Arial`;
          ctx.fillText(name.substring(0, 3), screenX + 2, screenY + tileSize / 2);
        }
      }
    });

    // Draw agents with enhanced visualization
    Object.values(agents).forEach((agent) => {
      const screenX = (agent.location.x - camera.x) * tileSize;
      const screenY = (agent.location.y - camera.y) * tileSize;
      
      if (screenX >= -tileSize && screenX < canvas.width &&
          screenY >= -tileSize && screenY < canvas.height) {
        
        const isSelected = selectedAgent?.agent_id === agent.agent_id;
        const isHovered = hoveredAgent === agent.agent_id;
        
        // Draw glow for selected/hovered agent
        if (isSelected || isHovered) {
          ctx.shadowBlur = 20;
          ctx.shadowColor = isSelected ? '#ffff00' : '#ffffff';
        }
        
        // Draw agent circle with status-based color
        const statusColor = agent.is_sleeping ? STATUS_COLORS.sleeping :
                           STATUS_COLORS[agent.status as keyof typeof STATUS_COLORS] || AGENT_COLOR;
        
        ctx.fillStyle = isSelected ? '#ffff00' : statusColor;
        ctx.beginPath();
        ctx.arc(
          screenX + tileSize / 2,
          screenY + tileSize / 2,
          tileSize / 2,
          0,
          Math.PI * 2
        );
        ctx.fill();
        
        // Reset shadow
        ctx.shadowBlur = 0;
        
        // Draw selection ring
        if (isSelected) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        
        // Draw status indicator
        if (agent.is_sleeping) {
          ctx.fillStyle = '#ffffff';
          ctx.font = '12px Arial';
          ctx.fillText('💤', screenX + tileSize / 2 + 8, screenY + tileSize / 2 - 8);
        } else if (agent.objects_in_use && agent.objects_in_use.length > 0) {
          ctx.fillStyle = '#ffd60a';
          ctx.beginPath();
          ctx.arc(screenX + tileSize / 2 + 8, screenY + tileSize / 2 - 8, 3, 0, Math.PI * 2);
          ctx.fill();
        }
        
        // Draw agent name/initials
        if (tileSize >= 15) {
          ctx.fillStyle = '#ffffff';
          ctx.font = `${Math.floor(tileSize / 2)}px Arial`;
          ctx.textAlign = 'center';
          ctx.fillText(
            agent.name.substring(0, 2),
            screenX + tileSize / 2,
            screenY + tileSize / 2 + 4
          );
        }
      }
    });

    // Draw hovered tile highlight
    if (hoveredTile) {
      const screenX = (hoveredTile.x - camera.x) * tileSize;
      const screenY = (hoveredTile.y - camera.y) * tileSize;
      
      ctx.strokeStyle = HOVER_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(screenX, screenY, tileSize, tileSize);
    }

  }, [worldMap, agents, camera, tileSize, hoveredTile, selectedAgent]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const speed = 5 / tileSize;
      
      switch (e.key) {
        case 'ArrowLeft':
        case 'a':
          setCamera(prev => ({ ...prev, x: Math.max(0, prev.x - speed) }));
          break;
        case 'ArrowRight':
        case 'd':
          if (worldMap) {
            setCamera(prev => ({ 
              ...prev, 
              x: Math.min(worldMap.dimensions.x, prev.x + speed) 
            }));
          }
          break;
        case 'ArrowUp':
        case 'w':
          setCamera(prev => ({ ...prev, y: Math.max(0, prev.y - speed) }));
          break;
        case 'ArrowDown':
        case 's':
          if (worldMap) {
            setCamera(prev => ({ 
              ...prev, 
              y: Math.min(worldMap.dimensions.y, prev.y + speed) 
            }));
          }
          break;
        case '+':
        case '=':
          handleZoom(2);
          break;
        case '-':
        case '_':
          handleZoom(-2);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tileSize, worldMap, handleZoom]);

  if (!worldMap) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-white">
        <div className="text-center">
          <div className="text-xl mb-2">No world map loaded</div>
          <div className="text-sm text-gray-400">Waiting for world data from server...</div>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative w-full h-full bg-black">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onClick={handleClick}
        className="cursor-grab active:cursor-grabbing"
      />
      
      {/* Thought Bubbles */}
      <AnimatePresence>
        {Object.values(agents).map(agent => {
          if (!agent.inner_thought && !agent.current_action?.inner_thought) return null;
          
          const screenX = (agent.location.x - camera.x) * tileSize + tileSize / 2;
          const screenY = (agent.location.y - camera.y) * tileSize;
          
          // Only show thought bubbles for visible agents and if zoomed in enough
          if (screenX < 0 || screenX > (containerRef.current?.clientWidth || 0) ||
              screenY < 0 || screenY > (containerRef.current?.clientHeight || 0) ||
              tileSize < 20) {
            return null;
          }
          
          const thought = agent.inner_thought || agent.current_action?.inner_thought || '';
          const emotion = agent.emotions?.primary_emotion;
          
          return (
            <ThoughtBubble
              key={`thought-${agent.agent_id}`}
              thought={thought}
              x={screenX}
              y={screenY}
              emotion={emotion}
            />
          );
        })}
      </AnimatePresence>
      
      {/* Info overlay */}
      <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white p-3 rounded text-sm">
        <div className="font-bold mb-2">{worldMap.name}</div>
        <div>Size: {worldMap.dimensions.x} × {worldMap.dimensions.y}</div>
        <div>Zoom: {tileSize}px</div>
        {hoveredTile && (
          <div className="mt-2 pt-2 border-t border-gray-600">
            <div>Position: ({hoveredTile.x}, {hoveredTile.y})</div>
            <div>
              Type: {worldMap.tiles[`${hoveredTile.x},${hoveredTile.y}`]?.type || 'Unknown'}
            </div>
          </div>
        )}
      </div>
      
      {/* Controls info */}
      <div className="absolute bottom-4 right-4 bg-black bg-opacity-75 text-white p-3 rounded text-xs">
        <div className="font-bold mb-1">Controls:</div>
        <div>WASD/Arrows - Move camera</div>
        <div>Mouse drag - Pan</div>
        <div>Scroll/+/- - Zoom</div>
        <div>Click agent - Select</div>
      </div>
      
      {/* Status Legend */}
      <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white p-3 rounded text-xs">
        <div className="font-bold mb-2">Agent Status:</div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: STATUS_COLORS.sleeping }}></div>
          <span>💤 Sleeping</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: STATUS_COLORS.moving }}></div>
          <span>🚶 Moving</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: STATUS_COLORS.acting }}></div>
          <span>⚡ Acting</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: STATUS_COLORS.waiting }}></div>
          <span>⏸️ Waiting</span>
        </div>
      </div>
    </div>
  );
};