// frontend/src/components/AgentDetailsPanel.tsx
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, Heart, Zap, Coffee, Droplets, Sparkles, Smile, 
  Battery, Utensils, Users, Brain, Home, MapPin 
} from 'lucide-react';

interface AgentNeeds {
  energy: number;
  hunger: number;
  bladder: number;
  hygiene: number;
  social: number;
  fun: number;
  comfort: number;
  fulfillment: number;
}

interface AgentEmotions {
  primary_emotion: string;
  emotion_intensity: number;
  mood_baseline: number;
  stress_level: number;
}

interface Agent {
  agent_id: string;
  name: string;
  location: { x: number; y: number };
  current_action?: {
    action_type: string;
    reasoning: string;
    target_object?: string;
    inner_thought?: string;
  };
  status: string;
  inner_thought?: string;
  is_sleeping?: boolean;
  objects_in_use?: string[];
  needs?: AgentNeeds;
  emotions?: AgentEmotions;
  wellbeing?: number;
}

interface AgentDetailsPanelProps {
  agent: Agent;
  onClose: () => void;
}

const NeedBar: React.FC<{ 
  label: string; 
  value: number; 
  icon: React.ReactNode; 
  color: string;
  inverted?: boolean;
}> = ({ label, value, icon, color, inverted = false }) => {
  // For inverted needs (hunger, bladder), lower is better
  const displayValue = inverted ? 100 - value : value;
  const barColor = displayValue > 70 ? 'bg-green-500' :
                   displayValue > 40 ? 'bg-yellow-500' :
                   'bg-red-500';
  
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <div className={`${color}`}>{icon}</div>
          <span className="text-sm font-medium text-white">{label}</span>
        </div>
        <span className="text-xs text-gray-300">{Math.round(displayValue)}%</span>
      </div>
      <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
        <motion.div
          className={`h-full ${barColor}`}
          initial={{ width: 0 }}
          animate={{ width: `${displayValue}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
    </div>
  );
};

const EmotionIndicator: React.FC<{ emotion: string; intensity: number }> = ({ emotion, intensity }) => {
  const emotionEmojis: { [key: string]: string } = {
    happy: "😊",
    sad: "😢",
    angry: "😠",
    anxious: "😰",
    excited: "🤩",
    bored: "😑",
    content: "😌",
    stressed: "😓",
    lonely: "😔",
    neutral: "😐"
  };

  const emoji = emotionEmojis[emotion.toLowerCase()] || "🙂";
  const intensityColor = intensity > 0.7 ? "text-yellow-400" :
                          intensity > 0.4 ? "text-blue-400" :
                          "text-gray-400";

  return (
    <div className="flex items-center gap-2">
      <span className="text-3xl">{emoji}</span>
      <div>
        <div className="text-sm font-semibold text-white capitalize">{emotion}</div>
        <div className="flex items-center gap-1">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className={`w-2 h-2 rounded-full ${
                i < intensity * 5 ? intensityColor : 'bg-gray-700'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export const AgentDetailsPanel: React.FC<AgentDetailsPanelProps> = ({ agent, onClose }) => {
  const needs = agent.needs;
  const emotions = agent.emotions;
  const actionText = agent.current_action?.reasoning || "Idle";
  const innerThought = agent.inner_thought || agent.current_action?.inner_thought || "...";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: 300 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 300 }}
        transition={{ type: "spring", damping: 25 }}
        className="fixed right-4 top-4 bottom-4 w-96 glass-panel border border-white/20 rounded-2xl shadow-2xl overflow-y-auto z-50"
      >
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-br from-blue-600/90 to-purple-600/90 backdrop-blur-xl p-4 border-b border-white/20">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xl font-bold text-white">{agent.name}</h2>
            <motion.button
              whileHover={{ scale: 1.1, rotate: 90 }}
              whileTap={{ scale: 0.9 }}
              onClick={onClose}
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
            >
              <X size={20} className="text-white" />
            </motion.button>
          </div>
          
          {/* Status Badge */}
          <div className="flex items-center gap-2">
            <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
              agent.is_sleeping ? 'bg-purple-500/50' :
              agent.status === 'moving' ? 'bg-blue-500/50' :
              agent.status === 'acting' ? 'bg-green-500/50' :
              'bg-gray-500/50'
            }`}>
              {agent.is_sleeping ? '💤 Sleeping' : 
               agent.status === 'moving' ? '🚶 Moving' :
               agent.status === 'acting' ? '⚡ Acting' :
               '⏸️ Waiting'}
            </div>
            {agent.wellbeing !== undefined && (
              <div className="px-3 py-1 rounded-full text-xs font-semibold bg-white/20">
                ❤️ {Math.round(agent.wellbeing)}%
              </div>
            )}
          </div>
        </div>

        <div className="p-4 space-y-6">
          {/* Current Action */}
          <motion.div 
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            className="glass-card p-4 rounded-xl border border-blue-400/30"
          >
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="text-yellow-400" size={18} />
              <h3 className="text-sm font-semibold text-white">Current Action</h3>
            </div>
            <p className="text-white text-sm mb-2">{actionText}</p>
            {agent.objects_in_use && agent.objects_in_use.length > 0 && (
              <div className="text-xs text-blue-300">
                Using: {agent.objects_in_use.join(', ')}
              </div>
            )}
          </motion.div>

          {/* Thought Bubble */}
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="relative glass-card p-4 rounded-xl border border-purple-400/30"
          >
            <div className="flex items-center gap-2 mb-2">
              <Brain className="text-purple-400" size={18} />
              <h3 className="text-sm font-semibold text-white">Thinking...</h3>
            </div>
            <p className="text-purple-200 text-sm italic">"{innerThought}"</p>
            <div className="absolute -bottom-2 right-8 w-4 h-4 bg-purple-500/30 border-r border-b border-purple-400/30 transform rotate-45"></div>
          </motion.div>

          {/* Location */}
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.15 }}
            className="glass-card p-3 rounded-xl border border-white/10"
          >
            <div className="flex items-center gap-2">
              <MapPin className="text-green-400" size={16} />
              <span className="text-sm text-gray-300">
                Position: ({Math.floor(agent.location.x)}, {Math.floor(agent.location.y)})
              </span>
            </div>
          </motion.div>

          {/* Emotions */}
          {emotions && (
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="glass-card p-4 rounded-xl border border-white/10"
            >
              <div className="flex items-center gap-2 mb-3">
                <Smile className="text-pink-400" size={18} />
                <h3 className="text-sm font-semibold text-white">Emotional State</h3>
              </div>
              <EmotionIndicator 
                emotion={emotions.primary_emotion} 
                intensity={emotions.emotion_intensity}
              />
              <div className="mt-3 pt-3 border-t border-white/10">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Mood:</span>
                  <span className={`font-semibold ${
                    emotions.mood_baseline > 0.3 ? 'text-green-400' :
                    emotions.mood_baseline < -0.3 ? 'text-red-400' :
                    'text-gray-300'
                  }`}>
                    {emotions.mood_baseline > 0.3 ? '😊 Positive' :
                     emotions.mood_baseline < -0.3 ? '😔 Negative' :
                     '😐 Neutral'}
                  </span>
                </div>
                {emotions.stress_level > 0.5 && (
                  <div className="flex justify-between text-xs mt-1">
                    <span className="text-gray-400">Stress:</span>
                    <span className="text-orange-400 font-semibold">
                      ⚠️ {Math.round(emotions.stress_level * 100)}%
                    </span>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Needs */}
          {needs && (
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.25 }}
              className="glass-card p-4 rounded-xl border border-white/10"
            >
              <div className="flex items-center gap-2 mb-4">
                <Heart className="text-red-400" size={18} />
                <h3 className="text-sm font-semibold text-white">Needs</h3>
              </div>
              
              <NeedBar 
                label="Energy" 
                value={needs.energy} 
                icon={<Battery size={16} />}
                color="text-green-400"
              />
              <NeedBar 
                label="Hunger" 
                value={needs.hunger} 
                icon={<Utensils size={16} />}
                color="text-orange-400"
                inverted
              />
              <NeedBar 
                label="Bladder" 
                value={needs.bladder} 
                icon={<Droplets size={16} />}
                color="text-blue-400"
                inverted
              />
              <NeedBar 
                label="Hygiene" 
                value={needs.hygiene} 
                icon={<Sparkles size={16} />}
                color="text-cyan-400"
              />
              <NeedBar 
                label="Social" 
                value={needs.social} 
                icon={<Users size={16} />}
                color="text-purple-400"
              />
              <NeedBar 
                label="Fun" 
                value={needs.fun} 
                icon={<Smile size={16} />}
                color="text-pink-400"
              />
              <NeedBar 
                label="Comfort" 
                value={needs.comfort} 
                icon={<Home size={16} />}
                color="text-yellow-400"
              />
              <NeedBar 
                label="Fulfillment" 
                value={needs.fulfillment} 
                icon={<Zap size={16} />}
                color="text-indigo-400"
              />
            </motion.div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};
