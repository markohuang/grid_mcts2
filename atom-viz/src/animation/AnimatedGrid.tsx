// Animated grid with full-board AOD laser tweezer visualization

import React, { useMemo } from 'react';
import { AtomPositions } from '../types';
import { AnimationStep, AnimationMove, PlaybackSettings } from './types';

interface AnimatedGridProps {
  rows: number;
  cols: number;
  basePositions: AtomPositions;
  currentStep: AnimationStep | null;
  stepProgress: number;
  settings: PlaybackSettings;
}

const GROUP_COLORS = [
  '#4ecdc4', '#ff6b6b', '#ffd93d', '#6c5ce7', '#00b894',
  '#fd79a8', '#74b9ff', '#e17055', '#a29bfe', '#55efc4',
];

const COLORS = {
  gridBgReconfig: '#0a0f14',
  gridBgGate: '#0a0e18',
  gridLine: '#1e2530',
  trap: '#2a3444',
  atomInactive: '#4a5568',
  atomText: '#0a0e14',
  laserH: '#ff6b6b',
  laserV: '#4ecdc4',
  gateOverlap: '#ffd93d',
};

const easeInOutCubic = (t: number): number => {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

export const AnimatedGrid: React.FC<AnimatedGridProps> = ({
  rows,
  cols,
  basePositions,
  currentStep,
  stepProgress,
  settings,
}) => {
  const cellSize = 56;
  const padding = 40;
  const svgWidth = cols * cellSize + padding * 2;
  const svgHeight = rows * cellSize + padding * 2;

  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

  const getReconfigPhase = (progress: number) => {
    const pickupRatio = settings.pickupDuration / settings.stepDuration;
    const settlingRatio = settings.settlingDuration / settings.stepDuration;
    const movingRatio = 1 - pickupRatio - settlingRatio;

    if (progress < pickupRatio) {
      return { 
        phase: 'pickup' as const, 
        phaseProgress: progress / pickupRatio,
        movementProgress: 0,
      };
    } else if (progress < pickupRatio + movingRatio) {
      const moveProgress = (progress - pickupRatio) / movingRatio;
      return { 
        phase: 'moving' as const, 
        phaseProgress: moveProgress,
        movementProgress: easeInOutCubic(moveProgress),
      };
    } else {
      return { 
        phase: 'settling' as const, 
        phaseProgress: (progress - pickupRatio - movingRatio) / settlingRatio,
        movementProgress: 1,
      };
    }
  };

  const getGatePhase = (progress: number) => {
    const pickupEnd = 0.08;
    const forwardEnd = 0.4;
    const overlapEnd = 0.6;
    const backwardEnd = 0.92;

    if (progress < pickupEnd) {
      return {
        phase: 'pickup' as const,
        phaseProgress: progress / pickupEnd,
        movementProgress: 0,
      };
    } else if (progress < forwardEnd) {
      const p = (progress - pickupEnd) / (forwardEnd - pickupEnd);
      return {
        phase: 'forward' as const,
        phaseProgress: p,
        movementProgress: easeInOutCubic(p),
      };
    } else if (progress < overlapEnd) {
      return {
        phase: 'overlap' as const,
        phaseProgress: (progress - forwardEnd) / (overlapEnd - forwardEnd),
        movementProgress: 1,
      };
    } else if (progress < backwardEnd) {
      const p = (progress - overlapEnd) / (backwardEnd - overlapEnd);
      return {
        phase: 'backward' as const,
        phaseProgress: p,
        movementProgress: 1 - easeInOutCubic(p),
      };
    } else {
      return {
        phase: 'settling' as const,
        phaseProgress: (progress - backwardEnd) / (1 - backwardEnd),
        movementProgress: 0,
      };
    }
  };

  const { atomRenderData, horizontalLasers, verticalLasers, isGatePhase } = useMemo(() => {
    const atomMoves = new Map<number, AnimationMove>();

    if (currentStep) {
      currentStep.moves.forEach(move => {
        atomMoves.set(move.atomId, move);
      });
    }

    let animPhase: { phase: string; phaseProgress: number; movementProgress: number };
    let isGate = false;

    if (!currentStep) {
      animPhase = { phase: 'idle', phaseProgress: 0, movementProgress: 0 };
    } else if (currentStep.phase === 'gate') {
      animPhase = getGatePhase(stepProgress);
      isGate = true;
    } else {
      animPhase = getReconfigPhase(stepProgress);
    }

    const { phase, phaseProgress, movementProgress } = animPhase;

    // Calculate hover offset
    let hoverOffset = 0;
    if (phase === 'pickup') {
      hoverOffset = phaseProgress * 8;
    } else if (phase === 'settling') {
      hoverOffset = (1 - phaseProgress) * 8;
    } else if (phase === 'forward' || phase === 'backward' || phase === 'moving' || phase === 'overlap') {
      hoverOffset = 8;
    }

    // Build atom render data
    const renderData: Array<{
      atomId: number;
      x: number;
      y: number;
      currentRow: number;
      currentCol: number;
      scale: number;
      color: string;
      isMoving: boolean;
      isOverlapping: boolean;
      overlappingWith?: number;
    }> = [];

    // Track moving atoms' current positions for laser calculation
    const movingAtomPositions: Array<{ atomId: number; row: number; col: number }> = [];

    Object.entries(basePositions).forEach(([idStr, pos]) => {
      const atomId = parseInt(idStr);
      const move = atomMoves.get(atomId);

      if (move && currentStep) {
        const currentCol = lerp(move.from.col, move.to.col, movementProgress);
        const currentRow = lerp(move.from.row, move.to.row, movementProgress);
        
        // Store for laser calculation
        movingAtomPositions.push({ atomId, row: currentRow, col: currentCol });

        // Pixel positions
        const x = padding + currentCol * cellSize + cellSize / 2;
        const y = padding + currentRow * cellSize + cellSize / 2 - hoverOffset;
        
        let scale: number;
        let isOverlapping = false;
        let overlappingWith: number | undefined;

        if (phase === 'pickup') {
          scale = 1 + phaseProgress * 0.15;
        } else if (phase === 'settling') {
          scale = 1.15 - phaseProgress * 0.15;
        } else if (phase === 'overlap') {
          scale = 1.25;
          isOverlapping = true;
          overlappingWith = move.partnerAtomId;
        } else {
          scale = 1.15;
        }

        renderData.push({
          atomId,
          x, y,
          currentRow,
          currentCol,
          scale,
          color: isOverlapping ? COLORS.gateOverlap : GROUP_COLORS[currentStep.groupIndex % GROUP_COLORS.length],
          isMoving: true,
          isOverlapping,
          overlappingWith,
        });
      } else {
        // Static atom
        const x = padding + pos.col * cellSize + cellSize / 2;
        const y = padding + pos.row * cellSize + cellSize / 2;
        renderData.push({
          atomId,
          x, y,
          currentRow: pos.row,
          currentCol: pos.col,
          scale: 1,
          color: COLORS.atomInactive,
          isMoving: false,
          isOverlapping: false,
        });
      }
    });

    // Build full-board lasers grouped by row/column
    const hLasers: Array<{ row: number; atomIds: number[] }> = [];
    const vLasers: Array<{ col: number; atomIds: number[] }> = [];

    if (currentStep && movingAtomPositions.length > 0 && phase !== 'idle') {
      // Group atoms by their current row (with small tolerance for floating point)
      const rowGroups = new Map<number, number[]>();
      const colGroups = new Map<number, number[]>();

      movingAtomPositions.forEach(({ atomId, row, col }) => {
        // Round to avoid floating point grouping issues
        const rowKey = Math.round(row * 1000) / 1000;
        const colKey = Math.round(col * 1000) / 1000;

        if (!rowGroups.has(rowKey)) rowGroups.set(rowKey, []);
        rowGroups.get(rowKey)!.push(atomId);

        if (!colGroups.has(colKey)) colGroups.set(colKey, []);
        colGroups.get(colKey)!.push(atomId);
      });

      // Create one horizontal laser per unique row
      rowGroups.forEach((atomIds, row) => {
        hLasers.push({ row, atomIds });
      });

      // Create one vertical laser per unique column
      colGroups.forEach((atomIds, col) => {
        vLasers.push({ col, atomIds });
      });
    }

    // Calculate laser opacity
    let laserOpacity = 0.7;
    if (phase === 'pickup') {
      laserOpacity = phaseProgress * 0.7;
    } else if (phase === 'settling') {
      laserOpacity = (1 - phaseProgress) * 0.7;
    }

    return { 
      atomRenderData: renderData, 
      horizontalLasers: hLasers.map(l => ({ ...l, opacity: laserOpacity })),
      verticalLasers: vLasers.map(l => ({ ...l, opacity: laserOpacity })),
      isGatePhase: isGate,
      hoverOffset,
    };
  }, [basePositions, currentStep, stepProgress, settings, rows, cols]);

  const bgColor = isGatePhase ? COLORS.gridBgGate : COLORS.gridBgReconfig;
  const borderColor = isGatePhase ? '#6c5ce766' : '#4ecdc466';

  // Calculate hover offset for laser positioning
  let hoverOffset = 0;
  if (currentStep) {
    const animPhase = currentStep.phase === 'gate' ? getGatePhase(stepProgress) : getReconfigPhase(stepProgress);
    const { phase, phaseProgress } = animPhase;
    if (phase === 'pickup') {
      hoverOffset = phaseProgress * 8;
    } else if (phase === 'settling') {
      hoverOffset = (1 - phaseProgress) * 8;
    } else if (phase === 'forward' || phase === 'backward' || phase === 'moving' || phase === 'overlap') {
      hoverOffset = 8;
    }
  }

  return (
    <svg
      width={svgWidth}
      height={svgHeight}
      style={{
        background: bgColor,
        borderRadius: '12px',
        border: `2px solid ${borderColor}`,
        transition: 'background 0.3s, border-color 0.3s',
      }}
    >
      {/* Grid lines */}
      {Array.from({ length: cols + 1 }).map((_, i) => (
        <line
          key={`v${i}`}
          x1={padding + i * cellSize}
          y1={padding}
          x2={padding + i * cellSize}
          y2={padding + rows * cellSize}
          stroke={COLORS.gridLine}
          strokeWidth="1"
        />
      ))}
      {Array.from({ length: rows + 1 }).map((_, i) => (
        <line
          key={`h${i}`}
          x1={padding}
          y1={padding + i * cellSize}
          x2={padding + cols * cellSize}
          y2={padding + i * cellSize}
          stroke={COLORS.gridLine}
          strokeWidth="1"
        />
      ))}

      {/* Trap sites */}
      {Array.from({ length: cols }).map((_, col) =>
        Array.from({ length: rows }).map((_, row) => (
          <rect
            key={`trap${col}-${row}`}
            x={padding + col * cellSize + 4}
            y={padding + row * cellSize + 4}
            width={cellSize - 8}
            height={cellSize - 8}
            fill={COLORS.trap}
            rx="4"
          />
        ))
      )}

      {/* Layer 1: Static atoms (non-moving) */}
      {atomRenderData
        .filter(atom => !atom.isMoving)
        .map(atom => (
          <g key={`atom-static-${atom.atomId}`}>
            <circle
              cx={atom.x}
              cy={atom.y}
              r={16}
              fill={atom.color}
            />
            <text
              x={atom.x}
              y={atom.y + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="11"
              fontWeight="bold"
              fill="#a0aec0"
            >
              {atom.atomId}
            </text>
          </g>
        ))}

      {/* Layer 2: Full-board horizontal lasers (one per unique row) */}
      {horizontalLasers.map((laser, i) => {
        // Laser spans full board width, at the interpolated row position
        const y = padding + laser.row * cellSize + cellSize / 2 - hoverOffset;
        const x1 = padding - 10;  // Extend slightly beyond grid
        const x2 = padding + cols * cellSize + 10;
        
        return (
          <line
            key={`laser-h-${i}`}
            x1={x1}
            y1={y}
            x2={x2}
            y2={y}
            stroke={COLORS.laserH}
            strokeWidth="4"
            opacity={laser.opacity}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${COLORS.laserH})` }}
          />
        );
      })}

      {/* Layer 3: Full-board vertical lasers (one per unique column) */}
      {verticalLasers.map((laser, i) => {
        // Laser spans full board height, at the interpolated column position
        const x = padding + laser.col * cellSize + cellSize / 2;
        const y1 = padding - 10 - hoverOffset;  // Extend slightly beyond grid, follow hover
        const y2 = padding + rows * cellSize + 10 - hoverOffset;
        
        return (
          <line
            key={`laser-v-${i}`}
            x1={x}
            y1={y1}
            x2={x}
            y2={y2}
            stroke={COLORS.laserV}
            strokeWidth="4"
            opacity={laser.opacity}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${COLORS.laserV})` }}
          />
        );
      })}

      {/* Layer 4: Moving atoms (non-overlapping) */}
      {atomRenderData
        .filter(atom => atom.isMoving && !atom.isOverlapping)
        .map(atom => (
          <g key={`atom-moving-${atom.atomId}`}>
            <circle
              cx={atom.x}
              cy={atom.y}
              r={16 * atom.scale}
              fill={atom.color}
              style={{ filter: `drop-shadow(0 0 10px ${atom.color})` }}
            />
            <text
              x={atom.x}
              y={atom.y + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={11 * atom.scale}
              fontWeight="bold"
              fill={COLORS.atomText}
            >
              {atom.atomId}
            </text>
          </g>
        ))}

      {/* Layer 5: Overlapping atoms (gate execution) - topmost */}
      {atomRenderData
        .filter(atom => atom.isOverlapping)
        .map(atom => (
          <g key={`atom-overlap-${atom.atomId}`}>
            <circle
              cx={atom.x}
              cy={atom.y}
              r={22 * atom.scale}
              fill={atom.color}
              style={{ filter: `drop-shadow(0 0 16px ${atom.color})` }}
            />
            <text
              x={atom.x}
              y={atom.y + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={10 * atom.scale}
              fontWeight="bold"
              fill={COLORS.atomText}
            >
              {atom.atomId}|{atom.overlappingWith}
            </text>
          </g>
        ))}
    </svg>
  );
};
