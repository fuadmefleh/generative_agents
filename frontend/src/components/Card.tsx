import React from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hoverable?: boolean;
  elevated?: boolean;
  onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  hoverable = false,
  elevated = false,
  onClick,
}) => {
  const baseClasses = 'glass-card rounded-xl border border-white/10 transition-all duration-300';
  const hoverClasses = hoverable ? 'cursor-pointer hover:border-blue-400/30 hover:shadow-lg' : '';
  const elevatedClasses = elevated ? 'shadow-2xl' : '';
  
  const CardComponent = hoverable ? motion.div : 'div';
  const motionProps = hoverable ? {
    whileHover: { scale: 1.02, y: -4 },
    whileTap: { scale: 0.98 },
  } : {};
  
  return (
    <CardComponent
      className={`${baseClasses} ${hoverClasses} ${elevatedClasses} ${className}`}
      onClick={onClick}
      {...motionProps}
    >
      {children}
    </CardComponent>
  );
};

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
  icon?: React.ReactNode;
}

export const CardHeader: React.FC<CardHeaderProps> = ({
  children,
  className = '',
  icon,
}) => {
  return (
    <div className={`flex items-center gap-2 p-4 border-b border-white/10 ${className}`}>
      {icon}
      <h3 className="font-semibold text-white">{children}</h3>
    </div>
  );
};

interface CardBodyProps {
  children: React.ReactNode;
  className?: string;
}

export const CardBody: React.FC<CardBodyProps> = ({
  children,
  className = '',
}) => {
  return (
    <div className={`p-4 ${className}`}>
      {children}
    </div>
  );
};

interface CardFooterProps {
  children: React.ReactNode;
  className?: string;
}

export const CardFooter: React.FC<CardFooterProps> = ({
  children,
  className = '',
}) => {
  return (
    <div className={`flex items-center gap-2 p-4 border-t border-white/10 ${className}`}>
      {children}
    </div>
  );
};
