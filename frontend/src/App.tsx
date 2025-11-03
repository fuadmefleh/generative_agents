// frontend/src/App.tsx
import React, { useEffect, useRef, useState } from 'react';
import { Toaster } from 'react-hot-toast';
import { Application as PixiApplication, extend } from '@pixi/react';
import { Container, Graphics, Sprite, Application as PIXIApp } from 'pixi.js';
import { useSimulationStore } from './store/simulationStore';
import { useWebSocket } from './hooks/useWebSocket';
import TiledJSONRenderer from './components/TiledJSONRenderer';
import PixiTiledJSONLoader from './components/PixiTiledJSONLoader';
import { Map, Grid3x3, FileJson, FileCode } from 'lucide-react';
import './App.css';

// Extend tells @pixi/react what Pixi.js components are available
extend({
  Container,
  Graphics,
  Sprite,
});

function App() {
  // include selectAgent so sidebar and renderer can select an agent
  const { agents, selectedAgent, isConnected, currentTime, selectAgent } = useSimulationStore();
  const { connect, disconnect } = useWebSocket();
  const appRef = useRef<PIXIApp | null>(null);
  const [rendererType, setRendererType] = useState<'pixi' | 'canvas'>('canvas');
  const [mapFormat] = useState<'json'>('json'); // Only JSON now
  
  const formattedWorldTime = currentTime
    ? (currentTime instanceof Date ? currentTime : new Date(currentTime)).toLocaleString()
    : '—';
  
  // Map URL - change this to match your backend
  const mapUrl = 'http://localhost:6010/map.json'; // Changed from map.tmx to map.json
  const baseUrl = 'http://localhost:6010/';
  
  const agentArray = Object.values(agents);
  
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);
  
  return (
    <div className="app">
      <Toaster position="top-right" />
      
      <div className="flex h-screen">
        {/* Sidebar */}
        <div className="w-80 bg-gray-800 text-white p-4 overflow-y-auto">
          <div className="mb-4">
            <h2 className="text-xl font-bold mb-2">Simulation</h2>
            <div className={`text-sm ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              {isConnected ? '● Connected' : '● Disconnected'}
            </div>
            <div className="text-xs text-gray-300 mt-2">
              <span className="font-semibold">World Time:</span> {formattedWorldTime}
            </div>
          </div>

          <div className="mb-4">
            <h2 className="text-xl font-bold mb-2">Simulation</h2>
            <div className={`text-sm ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              {isConnected ? '● Connected' : '● Disconnected'}
            </div>
          </div>
          
          {/* Map Format Info */}
          <div className="mb-4 p-3 bg-gray-700 rounded">
            <div className="flex items-center gap-2 text-sm mb-1">
              <FileJson size={16} />
              <span className="font-semibold">Map Format: JSON</span>
            </div>
            <p className="text-xs text-gray-400">
              Using Tiled JSON export format for better browser compatibility
            </p>
          </div>
          
          {/* Renderer Toggle */}
          <div className="mb-4 p-3 bg-gray-700 rounded">
            <h3 className="text-sm font-semibold mb-2">Renderer</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setRendererType('pixi')}
                className={`flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded text-xs transition ${
                  rendererType === 'pixi' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                }`}
                title="High-performance WebGL renderer"
              >
                <Grid3x3 size={14} />
                Pixi.js
              </button>
              <button
                onClick={() => setRendererType('canvas')}
                className={`flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded text-xs transition ${
                  rendererType === 'canvas' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                }`}
                title="Canvas renderer with layer controls"
              >
                <Map size={14} />
                Canvas
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              {rendererType === 'pixi' 
                ? 'WebGL-accelerated rendering' 
                : 'Interactive layer controls'}
            </p>
          </div>
          
          {/* Agents List */}
          <div className="mb-4">
            <h3 className="text-lg font-semibold mb-2">Agents ({Object.keys(agents).length})</h3>
            <div className="space-y-2">
              {Object.values(agents).map((agent) => {
                const agentAction = agent.current_action ?? agent.action ?? agent.currentAction ?? '—';
                return (
                <div
                  key={agent.agent_id}
                  className={`p-2 rounded cursor-pointer ${
                    selectedAgent?.agent_id === agent.agent_id
                      ? 'bg-blue-600'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                  onClick={() => selectAgent(agent)}
                >
                  <div className="font-semibold">{agent.name}</div>
                  <div className="text-xs text-yellow-300">Action: {agentAction}</div>
                  <div className="text-xs text-gray-300">
                    Position: ({Math.floor(agent.location.x)}, {Math.floor(agent.location.y)})
                  </div>
                  <div className="text-xs text-gray-400">{agent.status}</div>
                </div>
                );
              })}
            </div>
          </div>
          
          {/* Selected Agent Details */}
          {selectedAgent && (
            <div className="mt-4 p-3 bg-gray-700 rounded">
              <h3 className="font-semibold mb-2">Selected Agent</h3>
              <div className="text-sm space-y-1">
                <div><strong>Name:</strong> {selectedAgent.name}</div>
                <div><strong>ID:</strong> {selectedAgent.agent_id}</div>
                <div><strong>Status:</strong> {selectedAgent.status}</div>
                <div><strong>Action:</strong> {selectedAgent.current_action ?? selectedAgent.action ?? selectedAgent.currentAction ?? '—'}</div>
                <div><strong>Location:</strong> ({selectedAgent.location.x.toFixed(1)}, {selectedAgent.location.y.toFixed(1)})</div>
                {selectedAgent.description && (
                  <div><strong>Description:</strong> {selectedAgent.description}</div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Main map view */}
        <div className="flex-1 relative">
          {rendererType === 'pixi' ? (
            <PixiApplication
              ref={appRef}
              width={window.innerWidth - 320}
              height={window.innerHeight}
              background={0x1a1a1a}
              antialias={true}
            >
              {appRef.current && (
                <PixiTiledJSONLoader
                  app={appRef.current}
                  mapUrl={mapUrl}
                  baseUrl={baseUrl}
                  scale={0.75}
                  onMapLoaded={(map) => console.log('Pixi map loaded:', map)}
                  onError={(error) => console.error('Pixi map error:', error)}
                />
              )}
            </PixiApplication>
          ) : (
            <TiledJSONRenderer 
              mapUrl={mapUrl}
              baseUrl={baseUrl}
              className="w-full h-full"
              initialScale={0.75}
              showControls={true}
              showLayerPanel={true}
              onMapLoaded={(map) => console.log('Canvas map loaded:', map)}
              // ADD THESE NEW PROPS:
              agents={agentArray}
              selectedAgentId={selectedAgent?.agent_id}
              onAgentClick={(agent) => selectAgent(agent)}
              onAgentHover={(agent) => {
                // Optional: Add hover logic if needed
                // console.log('Hovering agent:', agent);
              }}
              showAgents={true}
              agentColor="#ff0000"
              selectedAgentColor="#ffff00"
              hoveredAgentColor="#00ff00"
              showAgentNames={true}
              agentRadius={0.4}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;