// frontend/src/components/SimulationControls.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, FastForward, RotateCcw, Settings } from 'lucide-react';

interface SimulationControlsProps {
  isRunning: boolean;
  isPaused: boolean;
  speed: number;
  onPlayPause: () => void;
  onSpeedChange: (speed: number) => void;
  onReset: () => void;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  isRunning,
  isPaused,
  speed,
  onPlayPause,
  onSpeedChange,
  onReset
}) => {
  const speedOptions = [0.5, 1, 2, 4, 8];

  return (
    <motion.div
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="fixed top-4 left-1/2 -translate-x-1/2 z-40"
    >
      <div className="glass-panel border border-white/20 rounded-2xl shadow-2xl px-6 py-3">
        <div className="flex items-center gap-4">
          {/* Play/Pause Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onPlayPause}
            className={`p-3 rounded-xl transition-all ${
              isRunning && !isPaused
                ? 'bg-gradient-to-r from-green-600 to-green-500 shadow-lg shadow-green-500/30'
                : 'bg-gradient-to-r from-blue-600 to-blue-500 shadow-lg shadow-blue-500/30'
            }`}
          >
            {isRunning && !isPaused ? (
              <Pause size={20} className="text-white" />
            ) : (
              <Play size={20} className="text-white" />
            )}
          </motion.button>

          {/* Speed Controls */}
          <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-xl border border-white/10">
            <FastForward size={16} className="text-blue-400" />
            <div className="flex gap-1">
              {speedOptions.map((speedOption) => (
                <motion.button
                  key={speedOption}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => onSpeedChange(speedOption)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    speed === speedOption
                      ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                      : 'bg-white/10 text-gray-300 hover:bg-white/20'
                  }`}
                >
                  {speedOption}x
                </motion.button>
              ))}
            </div>
          </div>

          {/* Reset Button */}
          <motion.button
            whileHover={{ scale: 1.05, rotate: -180 }}
            whileTap={{ scale: 0.95 }}
            onClick={onReset}
            className="p-3 rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 transition-all"
            title="Reset simulation"
          >
            <RotateCcw size={20} className="text-white" />
          </motion.button>

          {/* Status Indicator */}
          <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-xl border border-white/10">
            <div className={`w-2 h-2 rounded-full ${
              isRunning && !isPaused ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
            }`} />
            <span className="text-xs font-medium text-white">
              {isRunning && !isPaused ? 'Running' : isPaused ? 'Paused' : 'Stopped'}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
