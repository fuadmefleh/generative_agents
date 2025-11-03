// frontend/src/components/PixiTiledJSONLoader.tsx
import { useEffect, useRef } from 'react';
import { Application as PIXIApp, Texture, Rectangle, Assets } from 'pixi.js';
import { CompositeTilemap } from '@pixi/tilemap';

interface PixiTiledJSONLoaderProps {
  app: PIXIApp;
  mapUrl: string;
  baseUrl?: string;
  scale?: number;
}

export function PixiTiledJSONLoader({ 
  app, 
  mapUrl, 
  baseUrl = '',
  scale = 0.75
}: PixiTiledJSONLoaderProps) {
  const tilemapRef = useRef<CompositeTilemap | null>(null);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    let mounted = true;
    let tilemap: CompositeTilemap | null = null;

    // Wait for app.stage to be available
    const waitForStage = () => {
      if (!mounted) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
        return;
      }

      if (app && app.stage) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
        loadMap();
      }
    };

    const loadMap = async () => {
      try {
        const response = await fetch(mapUrl);
        const map = await response.json();
        
        if (!mounted) return;

        tilemap = new CompositeTilemap();

        // Load tilesets
        const tilesets = [];
        for (const ts of map.tilesets) {
          let tileset = ts;
          
          if (ts.source) {
            const res = await fetch(`${baseUrl}${ts.source}`);
            tileset = { ...(await res.json()), firstgid: ts.firstgid };
          }

          if (tileset.image) {
            const imageUrl = tileset.image.startsWith('http') 
              ? tileset.image 
              : `${baseUrl}${tileset.image}`;
            const texture = await Assets.load(imageUrl);
            tilesets.push({ tileset, texture });
          }
        }

        if (!mounted) return;

        // Render tiles
        for (const layer of map.layers) {
          if (layer.type === 'tilelayer' && layer.data) {
            const width = layer.width || map.width;
            const height = layer.height || map.height;

            for (let i = 0; i < layer.data.length; i++) {
              let gid = layer.data[i];
              if (gid === 0) continue;

              // Strip flip flags
              gid = gid & 0x1FFFFFFF;

              // Find tileset
              let tilesetData = null;
              for (const ts of tilesets) {
                if (gid >= ts.tileset.firstgid && 
                    gid < ts.tileset.firstgid + ts.tileset.tilecount) {
                  tilesetData = ts;
                  break;
                }
              }

              if (!tilesetData) continue;

              const { tileset, texture } = tilesetData;
              const localId = gid - tileset.firstgid;
              const cols = tileset.columns || Math.floor(tileset.imagewidth / tileset.tilewidth);
              
              const tileX = (localId % cols) * tileset.tilewidth;
              const tileY = Math.floor(localId / cols) * tileset.tileheight;
              
              const x = (i % width) * map.tilewidth;
              const y = Math.floor(i / width) * map.tileheight;

              const tileTexture = new Texture({
                source: texture.source,
                frame: new Rectangle(tileX, tileY, tileset.tilewidth, tileset.tileheight)
              });

              tilemap.tile(tileTexture, x, y);
            }
          }
        }

        if (mounted && app && app.stage) {
          tilemap.scale.set(scale);
          app.stage.addChild(tilemap);
          tilemapRef.current = tilemap;
        }
      } catch (error) {
        console.error('Error loading map:', error);
      }
    };

    // Check every 50ms for stage to be ready
    intervalRef.current = setInterval(waitForStage, 50);
    waitForStage(); // Also check immediately

    return () => {
      mounted = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (tilemap && app && app.stage) {
        try {
          app.stage.removeChild(tilemap);
          tilemap.destroy();
        } catch (e) {
          // Already destroyed
        }
      }
    };
  }, [app, mapUrl, baseUrl, scale]);

  return null;
}

export default PixiTiledJSONLoader;