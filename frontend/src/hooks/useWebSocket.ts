// frontend/src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import toast from 'react-hot-toast';

interface WebSocketMessage {
  type: string;
  data?: any;
  agent_id?: string;
  update_type?: string;
  event_type?: string;
  timestamp?: string;
  world_time?: string;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  const {
    setConnected,
    updateAgent,
    addAgent,
    removeAgent,
    setCurrentTime,
    setRunning,
    setPaused,
    setWorldMap,
  } = useSimulationStore();

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      switch (message.type) {
        case 'pong':
          break;

        case 'world_map':
          if (message.data) {
            setWorldMap(message.data);
            toast.success('World map loaded');
          }
          break;

        case 'world_update':
          if (message.data?.agents) {
            message.data.agents.forEach((agent: any) => {
              updateAgent(agent.agent_id, agent);
            });
          }

          // If the server includes a world_time field on world_update, update store
          const worldTime = message.world_time ?? message.data?.world_time;
          if (worldTime) {
            setCurrentTime(new Date(worldTime));
          }
          break;

        case 'agent_update':
          if (message.agent_id && message.data) {
            updateAgent(message.agent_id, message.data);
          }
          break;

        case 'agent_state':
          if (message.data) {
            updateAgent(message.data.agent_id, message.data);
          }
          break;

        case 'agents_list':
          if (message.data?.agents) {
            message.data.agents.forEach((agent: any) => {
              addAgent(agent);
            });
          }
          break;

        case 'simulation_event':
          handleSimulationEvent(message);
          break;

        case 'conversation':
          if (message.data) {
            toast.success(`Conversation between agents at ${message.data.location}`);
          }
          break;

        case 'error':
          toast.error(message.data?.message || 'WebSocket error');
          break;

        default:
          console.log('Unhandled message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [updateAgent, addAgent, setWorldMap]);

  const handleSimulationEvent = (message: WebSocketMessage) => {
    switch (message.event_type) {
      case 'time_update':
        if (message.data?.current_time) {
          setCurrentTime(new Date(message.data.current_time));
        }
        break;

      case 'simulation_started':
        setRunning(true);
        setPaused(false);
        toast.success('Simulation started');
        break;

      case 'simulation_paused':
        setPaused(true);
        toast.info('Simulation paused');
        break;

      case 'simulation_resumed':
        setPaused(false);
        toast.info('Simulation resumed');
        break;

      case 'simulation_stopped':
        setRunning(false);
        setPaused(false);
        toast.info('Simulation stopped');
        break;

      case 'agent_spawned':
        if (message.data) {
          toast.success(`Agent ${message.data.name} spawned`);
        }
        break;

      case 'world_initialized':
        toast.success('World initialized');
        break;

      default:
        console.log('Unhandled simulation event:', message.event_type);
    }
  };

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = import.meta.env.VITE_WS_URL || 'ws://192.168.18.145:9010/ws';
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
        reconnectAttemptsRef.current = 0;
        
        wsRef.current?.send(JSON.stringify({ type: 'ping' }));
        wsRef.current?.send(JSON.stringify({ type: 'get_agents' }));
        wsRef.current?.send(JSON.stringify({ type: 'get_world_map' })); // Request world map
      };

      wsRef.current.onmessage = handleMessage;

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        toast.error('WebSocket connection error');
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          toast.loading(`Reconnecting... (attempt ${reconnectAttemptsRef.current})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay);
        } else {
          toast.error('Failed to connect to server');
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      toast.error('Failed to connect to server');
    }
  }, [setConnected, handleMessage, setWorldMap]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setConnected(false);
  }, [setConnected]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  const subscribeToAgent = useCallback((agentId: string) => {
    sendMessage({ type: 'subscribe', agent_id: agentId });
  }, [sendMessage]);

  const unsubscribeFromAgent = useCallback((agentId: string) => {
    sendMessage({ type: 'unsubscribe', agent_id: agentId });
  }, [sendMessage]);

  useEffect(() => {
    const heartbeatInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        sendMessage({ type: 'ping' });
      }
    }, 30000);

    return () => clearInterval(heartbeatInterval);
  }, [sendMessage]);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    sendMessage,
    subscribeToAgent,
    unsubscribeFromAgent,
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  };
}