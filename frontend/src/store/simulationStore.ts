import { create } from 'zustand';
import { devtools } from 'zustand/middleware';


interface Point2D {
  x: number;
  y: number;
}

interface WorldMapTile {
  type: 'LAND' | 'OBSTACLE' | 'WATER';
  location: Point2D;
}

interface WorldMap {
  name: string;
  dimensions: Point2D;
  landmarks: Record<string, Point2D>;
  tiles: Record<string, WorldMapTile>;
}

interface Agent {
  agent_id: string;
  name: string;
  description: string;
  traits: string[];
  location: string;
  status: string;
  state?: any;
}

interface SimulationState {
  // Agents
  agents: Record<string, Agent>;
  selectedAgent: Agent | null;

  // World
  worldMap: WorldMap | null;
  
  // Simulation state
  isRunning: boolean;
  isPaused: boolean;
  currentTime: Date;
  speedMultiplier: number;
  tickInterval: number;
  
  // WebSocket
  isConnected: boolean;
  
  // Actions
  setAgents: (agents: Agent[]) => void;
  addAgent: (agent: Agent) => void;
  updateAgent: (agentId: string, update: Partial<Agent>) => void;
  removeAgent: (agentId: string) => void;
  selectAgent: (agent: Agent | null) => void;
  
  setRunning: (running: boolean) => void;
  setPaused: (paused: boolean) => void;
  setCurrentTime: (time: Date) => void;
  setSpeedMultiplier: (speed: number) => void;
  setConnected: (connected: boolean) => void;
  
  // WebSocket connection
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;
}

export const useSimulationStore = create<SimulationState>()(
  devtools(
    (set, get) => ({
      // Initial state
      agents: {},
      selectedAgent: null,
      worldMap: null,
      isRunning: false,
      isPaused: false,
      currentTime: new Date(),
      speedMultiplier: 1.0,
      tickInterval: 5.0,
      isConnected: false,

      // Agent actions
      setAgents: (agents) =>
        set((state) => ({
          agents: agents.reduce((acc, agent) => {
            acc[agent.agent_id] = agent;
            return acc;
          }, {} as Record<string, Agent>),
        })),

      addAgent: (agent) =>
        set((state) => ({
          agents: { ...state.agents, [agent.agent_id]: agent },
        })),

      updateAgent: (agentId, update) =>
        set((state) => ({
          agents: {
            ...state.agents,
            [agentId]: { ...state.agents[agentId], ...update },
          },
        })),

      removeAgent: (agentId) =>
        set((state) => {
          const { [agentId]: removed, ...rest } = state.agents;
          return { agents: rest };
        }),

      selectAgent: (agent) =>
        set({ selectedAgent: agent }),

      setWorldMap: (map) =>
        set({ worldMap: map }),

      // Simulation actions
      setRunning: (running) => set({ isRunning: running }),
      setPaused: (paused) => set({ isPaused: paused }),
      setCurrentTime: (time) => set({ currentTime: time }),
      setSpeedMultiplier: (speed) => set({ speedMultiplier: speed }),
      setConnected: (connected) => set({ isConnected: connected }),

      // WebSocket actions
      connectWebSocket: () => {
        // WebSocket connection logic will be handled by the useWebSocket hook
        set({ isConnected: true });
      },

      disconnectWebSocket: () => {
        set({ isConnected: false });
      },
    }),
    {
      name: 'simulation-store',
    }
  )
);

// Selector hooks for common use cases
export const useAgents = () => useSimulationStore((state) => state.agents);
export const useSelectedAgent = () => useSimulationStore((state) => state.selectedAgent);
export const useWorldMap = () => useSimulationStore((state) => state.worldMap);
export const useSimulationStatus = () =>
  useSimulationStore((state) => ({
    isRunning: state.isRunning,
    isPaused: state.isPaused,
    speedMultiplier: state.speedMultiplier,
  }));
