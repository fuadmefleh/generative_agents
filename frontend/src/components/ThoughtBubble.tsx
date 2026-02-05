// frontend/src/components/ThoughtBubble.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface ThoughtBubbleProps {
  thought: string;
  x: number;
  y: number;
  emotion?: string;
}

export const ThoughtBubble: React.FC<ThoughtBubbleProps> = ({ thought, x, y, emotion }) => {
  const getEmotionEmoji = (emotion?: string) => {
    if (!emotion) return '💭';
    const emotionMap: { [key: string]: string } = {
      happy: '😊',
      sad: '😢',
      angry: '😠',
      anxious: '😰',
      excited: '🤩',
      bored: '😑',
      content: '😌',
      stressed: '😓',
      lonely: '😔',
      neutral: '💭'
    };
    return emotionMap[emotion.toLowerCase()] || '💭';
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0, y: -20 }}
      transition={{ type: "spring", damping: 15 }}
      className="absolute pointer-events-none"
      style={{
        left: `${x}px`,
        top: `${y - 60}px`,
        transform: 'translateX(-50%)',
        zIndex: 1000
      }}
    >
      <div className="relative">
        <div className="bg-white/95 backdrop-blur-sm text-gray-800 text-xs px-3 py-2 rounded-2xl shadow-lg max-w-[150px] border border-gray-200">
          <div className="flex items-start gap-1">
            <span>{getEmotionEmoji(emotion)}</span>
            <span className="flex-1 leading-tight">{thought}</span>
          </div>
        </div>
        {/* Bubble tail */}
        <div className="absolute -bottom-2 left-1/2 -translate-x-1/2">
          <div className="w-3 h-3 bg-white/95 border-b border-r border-gray-200 transform rotate-45"></div>
        </div>
      </div>
    </motion.div>
  );
};
