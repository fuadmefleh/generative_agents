import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ZoomIn, ZoomOut, Move, Layers, Eye, EyeOff, Users } from 'lucide-react';

// Agent types
interface Agent {
  agent_id: string;
  name: string;
  location: {
    x: number;
    y: number;
  };
  color?: string;
  radius?: number;
  [key: string]: any; // Allow additional properties
}

// Tiled JSON format types
interface TiledTileset {
  firstgid: number;
  source?: string;
  name?: string;
  tilewidth?: number;
  tileheight?: number;
  tilecount?: number;
  columns?: number;
  image?: string;
  imagewidth?: number;
  imageheight?: number;
}

interface TiledLayer {
  id: number;
  name: string;
  type: 'tilelayer' | 'objectgroup' | 'imagelayer' | 'group';
  visible: boolean;
  opacity: number;
  width?: number;
  height?: number;
  data?: number[];
  objects?: TiledObject[];
  layers?: TiledLayer[];
  x: number;
  y: number;
}

interface TiledObject {
  id: number;
  name: string;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  visible: boolean;
  properties?: any[];
}

interface TiledMap {
  width: number;
  height: number;
  tilewidth: number;
  tileheight: number;
  orientation: string;
  renderorder: string;
  layers: TiledLayer[];
  tilesets: TiledTileset[];
  backgroundcolor?: string;
  infinite?: boolean;
  nextlayerid: number;
  nextobjectid: number;
  tiledversion: string;
  version: string;
}

interface TiledJSONRendererProps {
  mapUrl?: string;
  baseUrl?: string;
  className?: string;
  showControls?: boolean;
  showLayerPanel?: boolean;
  showAgents?: boolean;
  initialScale?: number;
  onMapLoaded?: (map: TiledMap) => void;
  // Agent-related props
  agents?: Agent[];
  selectedAgentId?: string | null;
  onAgentClick?: (agent: Agent) => void;
  onAgentHover?: (agent: Agent | null) => void;
  agentColor?: string;
  selectedAgentColor?: string;
  hoveredAgentColor?: string;
  agentRadius?: number;
  showAgentNames?: boolean;
  agentNameColor?: string;
  agentNameFont?: string;
}

// Debounce utility
const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// Throttle utility
const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

const TiledJSONRenderer: React.FC<TiledJSONRendererProps> = ({ 
  mapUrl,
  baseUrl = '',
  className = '',
  showControls = true,
  showLayerPanel: initialShowLayerPanel = true,
  showAgents: initialShowAgents = true,
  initialScale = 1,
  onMapLoaded,
  // Agent props
  agents = [],
  selectedAgentId = null,
  onAgentClick,
  onAgentHover,
  agentColor = '#ff0000',
  selectedAgentColor = '#ffff00',
  hoveredAgentColor = '#00ff00',
  agentRadius = 0.4, // As fraction of tile size
  showAgentNames = true,
  agentNameColor = '#ffffff',
  agentNameFont = 'Arial'
}) => {
  const [mapData, setMapData] = useState<TiledMap | null>(null);
  const [tilesetImages, setTilesetImages] = useState<Map<number, HTMLImageElement>>(new Map());
  const [externalTilesets, setExternalTilesets] = useState<Map<number, TiledTileset>>(new Map());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [scale, setScale] = useState(initialScale);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [visibleLayers, setVisibleLayers] = useState<Record<number, boolean>>({});
  const [showLayerPanel, setShowLayerPanel] = useState(initialShowLayerPanel);
  const [showAgentsLayer, setShowAgentsLayer] = useState(initialShowAgents);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [hoveredAgent, setHoveredAgent] = useState<Agent | null>(null);
  const [hoveredTile, setHoveredTile] = useState<{ x: number, y: number } | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragStateRef = useRef({ isDragging: false, startX: 0, startY: 0 });
  const tileCache = useRef<Map<number, any>>(new Map());
  const layerCanvasCache = useRef<Map<string, HTMLCanvasElement>>(new Map());
  const renderRequestRef = useRef<number>();

  // Canvas sizing with ResizeObserver
  useEffect(() => {
    if (!containerRef.current) return;
    
    const resizeObserver = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ width, height });
    });
    
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  // Load external tileset JSON
  const loadExternalTileset = async (source: string, firstgid: number): Promise<TiledTileset> => {
    const tilesetUrl = source.startsWith('http') ? source : `${baseUrl}${source}`;
    const response = await fetch(tilesetUrl);
    const tilesetData = await response.json();
    
    return {
      ...tilesetData,
      firstgid
    };
  };

  // Load single tileset with callback for progressive loading
  const loadSingleTileset = async (
    tileset: TiledTileset, 
    images: Map<number, HTMLImageElement>,
    externalMap: Map<number, TiledTileset>
  ): Promise<void> => {
    let actualTileset = tileset;
    
    if (tileset.source) {
      actualTileset = await loadExternalTileset(tileset.source, tileset.firstgid);
      externalMap.set(tileset.firstgid, actualTileset);
    }
    
    if (actualTileset.image) {
      return new Promise<void>((resolve) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        
        img.onload = () => {
          images.set(actualTileset.firstgid, img);
          resolve();
        };
        
        img.onerror = () => {
          console.error(`Failed to load tileset image: ${actualTileset.image}`);
          resolve();
        };
        
        const imageUrl = actualTileset.image.startsWith('http') 
          ? actualTileset.image 
          : `${baseUrl}${actualTileset.image}`;
        
        img.src = imageUrl;
      });
    }
  };

  // Load tileset images with progressive rendering
  const loadTilesetImages = async (map: TiledMap): Promise<void> => {
    const images = new Map<number, HTMLImageElement>();
    const externalTilesetMap = new Map<number, TiledTileset>();
    
    for (const tileset of map.tilesets) {
      await loadSingleTileset(tileset, images, externalTilesetMap);
      setTilesetImages(new Map(images));
      setExternalTilesets(new Map(externalTilesetMap));
    }
  };

  // Get tile source rectangle from GID with caching
  const getTileSourceRect = useCallback((gid: number, map: TiledMap) => {
    if (gid === 0) return null;

    const cacheKey = gid;
    if (tileCache.current.has(cacheKey)) {
      return tileCache.current.get(cacheKey);
    }

    const FLIPPED_HORIZONTALLY_FLAG = 0x80000000;
    const FLIPPED_VERTICALLY_FLAG = 0x40000000;
    const FLIPPED_DIAGONALLY_FLAG = 0x20000000;
    
    const flippedH = (gid & FLIPPED_HORIZONTALLY_FLAG) !== 0;
    const flippedV = (gid & FLIPPED_VERTICALLY_FLAG) !== 0;
    const flippedD = (gid & FLIPPED_DIAGONALLY_FLAG) !== 0;
    
    gid = gid & ~(FLIPPED_HORIZONTALLY_FLAG | FLIPPED_VERTICALLY_FLAG | FLIPPED_DIAGONALLY_FLAG);

    let tileset: TiledTileset | null = null;
    let tilesetImage: HTMLImageElement | null = null;
    
    for (let i = map.tilesets.length - 1; i >= 0; i--) {
      if (map.tilesets[i].firstgid <= gid) {
        tileset = externalTilesets.get(map.tilesets[i].firstgid) || map.tilesets[i];
        tilesetImage = tilesetImages.get(map.tilesets[i].firstgid) || null;
        break;
      }
    }

    if (!tileset || !tilesetImage) return null;

    const localId = gid - tileset.firstgid;
    const columns = tileset.columns || Math.floor((tileset.imagewidth || 0) / (tileset.tilewidth || 32));
    const tileWidth = tileset.tilewidth || 32;
    const tileHeight = tileset.tileheight || 32;
    
    const col = localId % columns;
    const row = Math.floor(localId / columns);

    const result = {
      image: tilesetImage,
      x: col * tileWidth,
      y: row * tileHeight,
      width: tileWidth,
      height: tileHeight,
      flippedH,
      flippedV,
      flippedD
    };

    tileCache.current.set(cacheKey, result);
    return result;
  }, [tilesetImages, externalTilesets]);

  // Calculate viewport bounds for culling
  const getViewportBounds = useCallback((map: TiledMap, layerWidth?: number, layerHeight?: number) => {
    const width = layerWidth || map.width;
    const height = layerHeight || map.height;
    
    return {
      left: Math.floor(Math.max(0, -offset.x / (map.tilewidth * scale))),
      top: Math.floor(Math.max(0, -offset.y / (map.tileheight * scale))),
      right: Math.ceil(Math.min(width, (canvasSize.width - offset.x) / (map.tilewidth * scale))),
      bottom: Math.ceil(Math.min(height, (canvasSize.height - offset.y) / (map.tileheight * scale)))
    };
  }, [scale, offset, canvasSize]);

  // Render a single layer with viewport culling
  const renderLayer = useCallback((ctx: CanvasRenderingContext2D, layer: TiledLayer, map: TiledMap) => {
    if (!visibleLayers[layer.id] && visibleLayers[layer.id] !== undefined) {
      return;
    }

    ctx.save();
    ctx.globalAlpha = layer.opacity;
    ctx.translate(layer.x, layer.y);

    if (layer.type === 'tilelayer' && layer.data) {
      const width = layer.width || map.width;
      const height = layer.height || map.height;
      
      const bounds = getViewportBounds(map, width, height);
      
      for (let y = bounds.top; y < bounds.bottom; y++) {
        for (let x = bounds.left; x < bounds.right; x++) {
          const index = y * width + x;
          const gid = layer.data[index];
          
          if (gid === 0) continue;
          
          const tile = getTileSourceRect(gid, map);
          if (!tile) continue;
          
          const destX = x * map.tilewidth;
          const destY = y * map.tileheight;
          
          ctx.save();
          
          if (tile.flippedH || tile.flippedV || tile.flippedD) {
            ctx.translate(destX + map.tilewidth / 2, destY + map.tileheight / 2);
            
            if (tile.flippedD) {
              ctx.rotate(Math.PI / 2);
              ctx.scale(-1, 1);
            }
            if (tile.flippedH) ctx.scale(-1, 1);
            if (tile.flippedV) ctx.scale(1, -1);
            
            ctx.drawImage(
              tile.image,
              tile.x, tile.y,
              tile.width, tile.height,
              -map.tilewidth / 2, -map.tileheight / 2,
              map.tilewidth, map.tileheight
            );
          } else {
            ctx.drawImage(
              tile.image,
              tile.x, tile.y,
              tile.width, tile.height,
              destX, destY,
              map.tilewidth, map.tileheight
            );
          }
          
          ctx.restore();
        }
      }
    } else if (layer.type === 'objectgroup' && layer.objects) {
      const bounds = getViewportBounds(map);
      const visibleObjects = layer.objects.filter(obj => {
        const objRight = (obj.x + obj.width) / map.tilewidth;
        const objBottom = (obj.y + obj.height) / map.tileheight;
        const objLeft = obj.x / map.tilewidth;
        const objTop = obj.y / map.tileheight;
        
        return objRight >= bounds.left && objLeft <= bounds.right &&
               objBottom >= bounds.top && objTop <= bounds.bottom;
      });

      if (visibleObjects.length > 0) {
        ctx.strokeStyle = '#00ff00';
        ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
        ctx.lineWidth = 2;
        
        visibleObjects.forEach(obj => {
          if (!obj.visible) return;
          
          ctx.save();
          ctx.translate(obj.x, obj.y);
          ctx.rotate(obj.rotation * Math.PI / 180);
          
          if (obj.width && obj.height) {
            ctx.strokeRect(0, 0, obj.width, obj.height);
            ctx.fillRect(0, 0, obj.width, obj.height);
          } else {
            ctx.fillStyle = '#00ff00';
            ctx.fillRect(-3, -3, 6, 6);
          }
          
          if (obj.name) {
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px sans-serif';
            ctx.fillText(obj.name, 0, -5);
          }
          
          ctx.restore();
        });
      }
    } else if (layer.type === 'group' && layer.layers) {
      layer.layers.forEach(childLayer => {
        renderLayer(ctx, childLayer, map);
      });
    }

    ctx.restore();
  }, [visibleLayers, getTileSourceRect, getViewportBounds]);

  // Render agents on the map
  const renderAgents = useCallback((ctx: CanvasRenderingContext2D, map: TiledMap) => {
    if (!showAgentsLayer || !agents || agents.length === 0) return;

    const bounds = getViewportBounds(map);
    
    // Filter agents that are in viewport
    const visibleAgents = agents.filter(agent => {
      const tileX = agent.location.x;
      const tileY = agent.location.y;
      return tileX >= bounds.left - 1 && tileX <= bounds.right + 1 &&
             tileY >= bounds.top - 1 && tileY <= bounds.bottom + 1;
    });

    visibleAgents.forEach(agent => {
      const screenX = agent.location.x * map.tilewidth;
      const screenY = agent.location.y * map.tileheight;
      
      // Determine agent color based on state
      let color = agent.color || agentColor;
      if (agent.agent_id === selectedAgentId) {
        color = selectedAgentColor;
      } else if (hoveredAgent?.agent_id === agent.agent_id) {
        color = hoveredAgentColor;
      }
      
      // Draw agent circle
      ctx.save();
      ctx.fillStyle = color;
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      
      const radius = (agent.radius || agentRadius) * map.tilewidth;
      
      ctx.beginPath();
      ctx.arc(
        screenX + map.tilewidth / 2,
        screenY + map.tileheight / 2,
        radius,
        0,
        Math.PI * 2
      );
      ctx.fill();
      ctx.stroke();
      
      // Draw selection ring
      if (agent.agent_id === selectedAgentId) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(
          screenX + map.tilewidth / 2,
          screenY + map.tileheight / 2,
          radius + 3,
          0,
          Math.PI * 2
        );
        ctx.stroke();
      }
      
      // Draw agent name/ID
      if (showAgentNames && map.tilewidth >= 15) {
        ctx.fillStyle = agentNameColor;
        ctx.font = `${Math.max(10, Math.floor(map.tilewidth / 3))}px ${agentNameFont}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Use first 2-3 characters of name or full name if tile is large
        const displayName = map.tilewidth >= 30 
          ? agent.name 
          : agent.name.substring(0, map.tilewidth >= 20 ? 3 : 2);
        
        ctx.fillText(
          displayName,
          screenX + map.tilewidth / 2,
          screenY + map.tileheight / 2
        );
      }
      
      ctx.restore();
    });
  }, [agents, showAgentsLayer, selectedAgentId, hoveredAgent, agentColor, 
      selectedAgentColor, hoveredAgentColor, agentRadius, showAgentNames, 
      agentNameColor, agentNameFont, getViewportBounds]);

  // Main render function
  const render = useCallback(() => {
    if (!mapData || !canvasRef.current) return;

    if (renderRequestRef.current) {
      cancelAnimationFrame(renderRequestRef.current);
    }

    renderRequestRef.current = requestAnimationFrame(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      // Clear canvas
      if (mapData.backgroundcolor) {
        ctx.fillStyle = mapData.backgroundcolor;
      } else {
        ctx.fillStyle = '#1a1a1a';
      }
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Apply transformations
      ctx.save();
      ctx.translate(offset.x, offset.y);
      ctx.scale(scale, scale);

      // Enable pixel-perfect rendering
      ctx.imageSmoothingEnabled = false;

      // Render all tile layers
      mapData.layers.forEach(layer => {
        renderLayer(ctx, layer, mapData);
      });

      // Render agents on top of tiles
      renderAgents(ctx, mapData);

      // Draw hovered tile highlight
      if (hoveredTile) {
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 2 / scale; // Adjust line width for scale
        ctx.strokeRect(
          hoveredTile.x * mapData.tilewidth,
          hoveredTile.y * mapData.tileheight,
          mapData.tilewidth,
          mapData.tileheight
        );
      }

      ctx.restore();
    });
  }, [mapData, scale, offset, renderLayer, renderAgents, hoveredTile]);

  // Render when dependencies change
  useEffect(() => {
    render();
  }, [render]);

  // Clear caches when map data changes
  useEffect(() => {
    tileCache.current.clear();
    layerCanvasCache.current.clear();
  }, [mapData]);

  // Load map from URL
  const loadMap = async () => {
    if (!mapUrl) return;
    
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(mapUrl);
      if (!response.ok) {
        throw new Error(`Failed to load map: ${response.statusText}`);
      }
      
      const map: TiledMap = await response.json();
      setMapData(map);
      
      const layers: Record<number, boolean> = {};
      const initializeLayers = (layerList: TiledLayer[]) => {
        layerList.forEach(layer => {
          layers[layer.id] = layer.visible;
          if (layer.type === 'group' && layer.layers) {
            initializeLayers(layer.layers);
          }
        });
      };
      initializeLayers(map.layers);
      setVisibleLayers(layers);
      
      await loadTilesetImages(map);
      
      if (onMapLoaded) {
        onMapLoaded(map);
      }
      
      console.log('Map loaded successfully:', map);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      console.error('Error loading map:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMap();
  }, [mapUrl]);

  // Convert screen coordinates to world/tile coordinates
  const screenToWorld = useCallback((screenX: number, screenY: number) => {
    if (!mapData) return null;
    
    const worldX = (screenX - offset.x) / scale;
    const worldY = (screenY - offset.y) / scale;
    
    const tileX = Math.floor(worldX / mapData.tilewidth);
    const tileY = Math.floor(worldY / mapData.tileheight);
    
    return { x: tileX, y: tileY, worldX, worldY };
  }, [mapData, offset, scale]);

  // Check if an agent is at the given tile position
  const getAgentAtPosition = useCallback((tileX: number, tileY: number) => {
    return agents.find(agent => 
      Math.floor(agent.location.x) === tileX && 
      Math.floor(agent.location.y) === tileY
    );
  }, [agents]);

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    dragStateRef.current = {
      isDragging: true,
      startX: e.clientX - offset.x,
      startY: e.clientY - offset.y
    };
  };

  const handleMouseMove = useCallback(
    throttle((e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas || !mapData) return;

      const rect = canvas.getBoundingClientRect();
      const screenX = e.clientX - rect.left;
      const screenY = e.clientY - rect.top;

      // Update hovered tile and check for hovered agent
      const worldPos = screenToWorld(screenX, screenY);
      if (worldPos) {
        if (worldPos.x >= 0 && worldPos.x < mapData.width && 
            worldPos.y >= 0 && worldPos.y < mapData.height) {
          setHoveredTile({ x: worldPos.x, y: worldPos.y });
          
          // Check for hovered agent
          const agent = getAgentAtPosition(worldPos.x, worldPos.y);
          if (agent !== hoveredAgent) {
            setHoveredAgent(agent || null);
            if (onAgentHover) {
              onAgentHover(agent || null);
            }
          }
        } else {
          setHoveredTile(null);
          if (hoveredAgent) {
            setHoveredAgent(null);
            if (onAgentHover) {
              onAgentHover(null);
            }
          }
        }
      }

      // Handle dragging
      if (dragStateRef.current.isDragging) {
        setOffset({
          x: e.clientX - dragStateRef.current.startX,
          y: e.clientY - dragStateRef.current.startY
        });
      }
    }, 16),
    [mapData, screenToWorld, getAgentAtPosition, hoveredAgent, onAgentHover]
  );

  const handleMouseUp = () => {
    dragStateRef.current.isDragging = false;
  };

  const handleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !mapData) return;

    const rect = canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    const worldPos = screenToWorld(screenX, screenY);
    if (worldPos) {
      // Check if clicked on an agent
      const agent = getAgentAtPosition(worldPos.x, worldPos.y);
      if (agent && onAgentClick) {
        onAgentClick(agent);
      }
    }
  }, [mapData, screenToWorld, getAgentAtPosition, onAgentClick]);

  // Debounced wheel handler
  const handleWheel = useMemo(
    () => debounce((e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setScale(prev => Math.max(0.1, Math.min(5, prev * delta)));
    }, 16),
    []
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const wheelHandler = (e: WheelEvent) => handleWheel(e);
    canvas.addEventListener('wheel', wheelHandler, { passive: false });

    return () => {
      canvas.removeEventListener('wheel', wheelHandler);
    };
  }, [handleWheel]);

  // Toggle layer visibility
  const toggleLayer = (layerId: number) => {
    setVisibleLayers(prev => ({
      ...prev,
      [layerId]: !prev[layerId]
    }));
  };

  // Toggle agents layer
  const toggleAgentsLayer = () => {
    setShowAgentsLayer(prev => !prev);
  };

  // Memoize all layers
  const allLayers = useMemo(() => {
    if (!mapData) return [];
    
    const getAllLayers = (layers: TiledLayer[]): TiledLayer[] => {
      const result: TiledLayer[] = [];
      layers.forEach(layer => {
        result.push(layer);
        if (layer.type === 'group' && layer.layers) {
          result.push(...getAllLayers(layer.layers));
        }
      });
      return result;
    };
    
    return getAllLayers(mapData.layers);
  }, [mapData]);

  // Control handlers
  const handleResetView = useCallback(() => {
    setScale(initialScale);
    setOffset({ x: 0, y: 0 });
  }, [initialScale]);

  const handleZoomIn = useCallback(() => {
    setScale(prev => Math.min(5, prev * 1.2));
  }, []);

  const handleZoomOut = useCallback(() => {
    setScale(prev => Math.max(0.1, prev * 0.8));
  }, []);

  return (
    <div ref={containerRef} className={`relative bg-gray-950 ${className}`} style={{ width: '100%', height: '100%' }}>
      {/* Controls Bar */}
      {showControls && (
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
          <button
            onClick={handleZoomIn}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded text-white transition"
            title="Zoom In"
          >
            <ZoomIn size={20} />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded text-white transition"
            title="Zoom Out"
          >
            <ZoomOut size={20} />
          </button>
          <button
            onClick={handleResetView}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded text-white transition"
            title="Reset View"
          >
            <Move size={20} />
          </button>
          <button
            onClick={() => setShowLayerPanel(!showLayerPanel)}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded text-white transition"
            title="Toggle Layers Panel"
          >
            <Layers size={20} />
          </button>
          <button
            onClick={toggleAgentsLayer}
            className={`p-2 rounded text-white transition ${
              showAgentsLayer ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-800 hover:bg-gray-700'
            }`}
            title="Toggle Agents"
          >
            <Users size={20} />
          </button>
          <span className="px-3 py-2 bg-gray-800 rounded text-white text-sm">
            {(scale * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Layers Panel */}
      {showLayerPanel && mapData && (
        <div className="absolute top-4 right-4 z-10 w-64 bg-gray-800 rounded-lg shadow-lg p-3 max-h-96 overflow-y-auto">
          <h3 className="text-white text-sm font-bold mb-2 flex items-center gap-2">
            <Layers size={16} />
            Layers
          </h3>
          
          {/* Agents layer toggle */}
          {agents.length > 0 && (
            <div className="mb-2 pb-2 border-b border-gray-700">
              <div className="flex items-center justify-between p-1.5 bg-gray-700 rounded hover:bg-gray-600 transition">
                <label className="flex items-center gap-2 cursor-pointer flex-1">
                  <button
                    onClick={toggleAgentsLayer}
                    className="text-white hover:text-blue-400"
                  >
                    {showAgentsLayer ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                  <span className="text-xs text-white">Agents ({agents.length})</span>
                </label>
                <span className="text-xs text-blue-400">
                  <Users size={14} />
                </span>
              </div>
            </div>
          )}
          
          {/* Map layers */}
          <div className="space-y-1">
            {allLayers.map((layer) => (
              <div
                key={layer.id}
                className="flex items-center justify-between p-1.5 bg-gray-700 rounded hover:bg-gray-600 transition"
              >
                <label className="flex items-center gap-2 cursor-pointer flex-1">
                  <button
                    onClick={() => toggleLayer(layer.id)}
                    className="text-white hover:text-blue-400"
                  >
                    {visibleLayers[layer.id] !== false ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                  <span className="text-xs text-white truncate">{layer.name}</span>
                </label>
                <span className="text-xs text-gray-400">
                  {layer.type === 'tilelayer' ? 'Tile' : 
                   layer.type === 'objectgroup' ? 'Obj' : 
                   layer.type === 'group' ? 'Grp' : 'Img'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Overlay */}
      {mapData && (
        <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white p-3 rounded text-sm">
          <div className="font-bold mb-1">Map Info</div>
          <div className="text-xs space-y-1">
            <div>Size: {mapData.width} × {mapData.height}</div>
            <div>Zoom: {(scale * 100).toFixed(0)}%</div>
            {hoveredTile && (
              <>
                <div className="mt-2 pt-2 border-t border-gray-600">
                  <div>Tile: ({hoveredTile.x}, {hoveredTile.y})</div>
                </div>
              </>
            )}
            {hoveredAgent && (
              <div className="mt-2 pt-2 border-t border-gray-600">
                <div className="font-bold">{hoveredAgent.name}</div>
                <div>ID: {hoveredAgent.agent_id}</div>
                <div>Pos: ({hoveredAgent.location.x.toFixed(1)}, {hoveredAgent.location.y.toFixed(1)})</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-20 bg-black bg-opacity-50">
          <div className="text-white text-xl">Loading map...</div>
        </div>
      )}
      
      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center z-20">
          <div className="bg-red-900 border border-red-700 rounded p-4 max-w-md">
            <h3 className="text-red-200 font-bold mb-2">Error loading map</h3>
            <p className="text-red-300">{error}</p>
            <button 
              onClick={loadMap}
              className="mt-3 px-4 py-2 bg-red-700 hover:bg-red-600 text-white rounded"
            >
              Retry
            </button>
          </div>
        </div>
      )}
      
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        className={`cursor-move ${loading ? 'opacity-50' : ''}`}
        style={{ imageRendering: 'pixelated' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
      />
    </div>
  );
};

export default TiledJSONRenderer;