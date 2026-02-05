import React from 'react';
import { motion } from 'framer-motion';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  pulse?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  className = '',
  pulse = false,
}) => {
  const variantClasses = {
    primary: 'bg-gradient-to-r from-blue-600 to-blue-500 text-white',
    secondary: 'bg-white/10 text-gray-300 border border-white/20',
    success: 'bg-gradient-to-r from-green-600 to-green-500 text-white',
    warning: 'bg-gradient-to-r from-yellow-600 to-yellow-500 text-white',
    danger: 'bg-gradient-to-r from-red-600 to-red-500 text-white',
    info: 'bg-gradient-to-r from-cyan-600 to-cyan-500 text-white',
  };
  
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-[10px]',
    md: 'px-3 py-1 text-xs',
    lg: 'px-4 py-1.5 text-sm',
  };
  
  const BadgeComponent = pulse ? motion.span : 'span';
  const motionProps = pulse ? {
    animate: { scale: [1, 1.05, 1] },
    transition: { repeat: Infinity, duration: 2 },
  } : {};
  
  return (
    <BadgeComponent
      className={`inline-flex items-center justify-center rounded-full font-semibold ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      {...motionProps}
    >
      {children}
    </BadgeComponent>
  );
};

interface StatusBadgeProps {
  status: 'online' | 'offline' | 'idle' | 'busy';
  showDot?: boolean;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  showDot = true,
  className = '',
}) => {
  const statusConfig = {
    online: { color: 'bg-green-400', text: 'Online', dotColor: 'bg-green-400' },
    offline: { color: 'bg-gray-500', text: 'Offline', dotColor: 'bg-gray-400' },
    idle: { color: 'bg-yellow-500', text: 'Idle', dotColor: 'bg-yellow-400' },
    busy: { color: 'bg-red-500', text: 'Busy', dotColor: 'bg-red-400' },
  };
  
  const config = statusConfig[status];
  
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-white/10 text-white ${className}`}>
      {showDot && (
        <motion.span
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className={`w-2 h-2 rounded-full ${config.dotColor}`}
          style={{
            boxShadow: `0 0 8px ${config.color.replace('bg-', 'rgba(').replace('-400', ', 0.6)').replace('-500', ', 0.6)')}`,
          }}
        />
      )}
      {config.text}
    </span>
  );
};
