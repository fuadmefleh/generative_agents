import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://192.168.18.145:9010';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Agent endpoints
  async createAgent(data: {
    name: string;
    description: string;
    traits: string[];
    location: string;
  }) {
    const response = await apiClient.post('/api/agents', data);
    return response.data;
  },

  async getAgents() {
    const response = await apiClient.get('/api/agents');
    return response.data;
  },

  async getAgent(agentId: string) {
    const response = await apiClient.get(`/api/agents/${agentId}`);
    return response.data;
  },

  async deleteAgent(agentId: string) {
    const response = await apiClient.delete(`/api/agents/${agentId}`);
    return response.data;
  },

  async getAgentState(agentId: string) {
    const response = await apiClient.get(`/api/agents/${agentId}/state`);
    return response.data;
  },

  async getAgentMemories(agentId: string, n: number = 20) {
    const response = await apiClient.get(`/api/agents/${agentId}/memories`, {
      params: { n },
    });
    return response.data;
  },

  async retrieveMemories(agentId: string, query: string, n: number = 5) {
    const response = await apiClient.post(`/api/agents/${agentId}/memories/retrieve`, {
      query,
      n,
    });
    return response.data;
  },

  async perceive(agentId: string, observation: string, location?: string) {
    const response = await apiClient.post(`/api/agents/${agentId}/perceive`, {
      observation,
      location,
    });
    return response.data;
  },

  async triggerReflection(agentId: string) {
    const response = await apiClient.post(`/api/agents/${agentId}/reflect`);
    return response.data;
  },

  async createPlan(agentId: string, timeHorizonHours: number = 24) {
    const response = await apiClient.post(`/api/agents/${agentId}/plan`, {
      time_horizon_hours: timeHorizonHours,
    });
    return response.data;
  },

  async moveAgent(agentId: string, destination: string) {
    const response = await apiClient.post(`/api/agents/${agentId}/move`, {
      destination,
    });
    return response.data;
  },

  async converseAgents(agentId: string, otherAgentId: string, initialMessage?: string) {
    const response = await apiClient.post(`/api/agents/${agentId}/converse`, {
      other_agent_id: otherAgentId,
      initial_message: initialMessage,
    });
    return response.data;
  },

  async getAgentSummary(agentId: string) {
    const response = await apiClient.get(`/api/agents/${agentId}/summary`);
    return response.data;
  },

  async getMemoryStats(agentId: string) {
    const response = await apiClient.get(`/api/agents/${agentId}/memory-stats`);
    return response.data;
  },

  // Simulation endpoints
  async startSimulation(config?: {
    tick_interval?: number;
    speed_multiplier?: number;
    auto_reflect?: boolean;
    auto_plan?: boolean;
    enable_conversations?: boolean;
  }) {
    const response = await apiClient.post('/api/simulation/start', config || {});
    return response.data;
  },

  async pauseSimulation() {
    const response = await apiClient.post('/api/simulation/pause');
    return response.data;
  },

  async resumeSimulation() {
    const response = await apiClient.post('/api/simulation/resume');
    return response.data;
  },

  async stopSimulation() {
    const response = await apiClient.post('/api/simulation/stop');
    return response.data;
  },

  async getSimulationStatus() {
    const response = await apiClient.get('/api/simulation/status');
    return response.data;
  },

  async setSimulationSpeed(speedMultiplier: number) {
    const response = await apiClient.post('/api/simulation/speed', null, {
      params: { speed_multiplier: speedMultiplier },
    });
    return response.data;
  },

  async initializeWorld(locations: any, objects?: any) {
    const response = await apiClient.post('/api/simulation/initialize-world', {
      locations,
      objects: objects || {},
    });
    return response.data;
  },

  async spawnDefaultAgents() {
    const response = await apiClient.post('/api/simulation/spawn-agents');
    return response.data;
  },
};

// Error interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.data);
    } else if (error.request) {
      console.error('Network Error:', error.message);
    } else {
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);