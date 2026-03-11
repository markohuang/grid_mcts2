import React from 'react';
import { Position, Move, Gate, AtomPositions } from './types';

interface GridProps {
  rows: number;
  cols: number;
  atoms: AtomPositions;                    // Current atom positions (before moves)
  moves?: Move[];                          // Reconfig moves
  moveGroups?: number[];                   // Parallel group for each move
  gates?: Gate[];                          // Gates to display
  gateGroups?: number[];                   // Parallel group for each gate
  activeAtoms?: Set<number>;               // Atoms involved in current layer's gates
  isEditable?: boolean;
  selectedAtom?: number | null;
  onAtomClick?: (atomId: number) => void;
  onCellClick?: (pos: Position) => void;
  onMoveClick?: (move: Move) => void;
  label?: string;
  cost?: number;
  showGateArrows?: boolean;
}

// Color palette for parallel groups
const GROUP_COLORS = [
  '#4ecdc4', // teal
  '#ff6b6b', // red
  '#ffd93d', // yellow
  '#6c5ce7', // purple
  '#00b894', // green
  '#fd79a8', // pink
  '#74b9ff', // blue
  '#e17055', // orange
  '#a29bfe', // light purple
  '#55efc4', // mint
];

const getGroupColor = (groupIndex: number) => GROUP_COLORS[groupIndex % GROUP_COLORS.length];

const COLORS = {
  gridBg: '#12171f',
  gridLine: '#1e2530',
  trap: '#2a3444',
  atomInactive: '#4a5568',
  atomText: '#0a0e14',
  atomTextInactive: '#a0aec0',
  atomSelected: '#ffffff',
  textMuted: '#6b7280',
  cardBorder: '#2a3444',
};

export const Grid: React.FC<GridProps> = ({
  rows,
  cols,
  atoms,
  moves = [],
  moveGroups = [],
  gates = [],
  gateGroups = [],
  activeAtoms,
  isEditable = false,
  selectedAtom,
  onAtomClick,
  onCellClick,
  onMoveClick,
  label,
  cost,
  showGateArrows = true,
}) => {
  const cellSize = 56;
  const padding = 30;
  const svgWidth = cols * cellSize + padding * 2;
  const svgHeight = rows * cellSize + padding * 2;

  const getCellCenter = (col: number, row: number) => ({
    x: padding + col * cellSize + cellSize / 2,
    y: padding + row * cellSize + cellSize / 2,
  });

  // Build maps for move information
  const atomMoveMap = new Map<number, { move: Move; groupIndex: number }>();
  moves.forEach((move, idx) => {
    atomMoveMap.set(move.atom, { move, groupIndex: moveGroups[idx] ?? 0 });
  });

  // Build map of atom to gate group (for gate execution view)
  const atomGateGroupMap = new Map<number, number>();
  gates.forEach(([a1, a2], idx) => {
    const group = gateGroups[idx] ?? idx;
    atomGateGroupMap.set(a1, group);
    atomGateGroupMap.set(a2, group);
  });

  // Determine which atoms are "active" (can be interacted with)
  const effectiveActiveAtoms = activeAtoms ?? new Set(Object.keys(atoms).map(Number));

  // Collect all unique groups for creating arrow markers
  const allGroups = new Set([...moveGroups, ...gateGroups]);

  // Get atom color based on context
  const getAtomColor = (atomId: number) => {
    // If atom has a move, use its move's group color
    const moveInfo = atomMoveMap.get(atomId);
    if (moveInfo !== undefined) {
      return getGroupColor(moveInfo.groupIndex);
    }
    
    // If in gate execution view, use gate group color
    const gateGroup = atomGateGroupMap.get(atomId);
    if (gateGroup !== undefined) {
      return getGroupColor(gateGroup);
    }
    
    // If not active, grey
    if (!effectiveActiveAtoms.has(atomId)) {
      return COLORS.atomInactive;
    }
    
    // Default: use atom ID for color (for active atoms without moves)
    return GROUP_COLORS[atomId % GROUP_COLORS.length];
  };

  // Check if atom is active (can be clicked/moved)
  const isAtomActive = (atomId: number) => effectiveActiveAtoms.has(atomId);

  // Get atoms that are being moved (their original positions become empty)
  const movedAtomIds = new Set(moves.map(m => m.atom));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
      {label && (
        <div style={{
          fontSize: '11px',
          fontWeight: '600',
          color: COLORS.textMuted,
          textTransform: 'uppercase',
          letterSpacing: '1px',
        }}>
          {label}
        </div>
      )}

      <svg
        width={svgWidth}
        height={svgHeight}
        style={{
          background: COLORS.gridBg,
          borderRadius: '8px',
          border: `1px solid ${COLORS.cardBorder}`,
          cursor: isEditable ? 'pointer' : 'default',
        }}
      >
        <defs>
          {/* Arrow markers for each group color */}
          {[...allGroups].map(group => (
            <marker
              key={`arrow-${group}`}
              id={`arrowhead-group-${group}`}
              markerWidth="6"
              markerHeight="5"
              refX="5"
              refY="2.5"
              orient="auto"
            >
              <polygon points="0 0, 6 2.5, 0 5" fill={getGroupColor(group)} />
            </marker>
          ))}
          {/* Default arrow marker */}
          <marker
            id="arrowhead-default"
            markerWidth="6"
            markerHeight="5"
            refX="5"
            refY="2.5"
            orient="auto"
          >
            <polygon points="0 0, 6 2.5, 0 5" fill="#ffd93d" />
          </marker>
        </defs>

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

        {/* Trap sites (clickable cells) */}
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
              style={{ cursor: isEditable ? 'pointer' : 'default' }}
              onClick={() => isEditable && onCellClick?.({ row, col })}
            />
          ))
        )}

        {/* Reconfiguration move arrows */}
        {moves.map((move, i) => {
          const start = getCellCenter(move.from.col, move.from.row);
          const end = getCellCenter(move.to.col, move.to.row);
          
          const dx = end.x - start.x;
          const dy = end.y - start.y;
          const len = Math.sqrt(dx * dx + dy * dy);
          if (len === 0) return null;
          
          const offsetStart = 18;
          const offsetEnd = 20;
          const group = moveGroups[i];
          const color = group !== undefined ? getGroupColor(group) : '#ffd93d';
          const markerId = group !== undefined ? `arrowhead-group-${group}` : 'arrowhead-default';

          return (
            <line
              key={`move-${i}`}
              x1={start.x + (dx / len) * offsetStart}
              y1={start.y + (dy / len) * offsetStart}
              x2={end.x - (dx / len) * offsetEnd}
              y2={end.y - (dy / len) * offsetEnd}
              stroke={color}
              strokeWidth="2.5"
              markerEnd={`url(#${markerId})`}
              style={{
                cursor: isEditable ? 'pointer' : 'default',
                filter: `drop-shadow(0 0 3px ${color}50)`,
              }}
              onClick={(e) => {
                e.stopPropagation();
                if (isEditable) onMoveClick?.(move);
              }}
            />
          );
        })}

        {/* Gate arrows (dashed) */}
        {showGateArrows && gates.map((gate, i) => {
          const atom1Pos = atoms[gate[0]];
          const atom2Pos = atoms[gate[1]];
          if (!atom1Pos || !atom2Pos) return null;

          const start = getCellCenter(atom1Pos.col, atom1Pos.row);
          const end = getCellCenter(atom2Pos.col, atom2Pos.row);

          const dx = end.x - start.x;
          const dy = end.y - start.y;
          const len = Math.sqrt(dx * dx + dy * dy);
          if (len === 0) return null;

          const offsetStart = 18;
          const offsetEnd = 20;
          const group = gateGroups[i];
          const color = group !== undefined ? getGroupColor(group) : '#6c5ce7';

          return (
            <line
              key={`gate-${i}`}
              x1={start.x + (dx / len) * offsetStart}
              y1={start.y + (dy / len) * offsetStart}
              x2={end.x - (dx / len) * offsetEnd}
              y2={end.y - (dy / len) * offsetEnd}
              stroke={color}
              strokeWidth="2.5"
              strokeDasharray="5,3"
              markerEnd={group !== undefined ? `url(#arrowhead-group-${group})` : undefined}
              style={{ filter: `drop-shadow(0 0 3px ${color}50)` }}
            />
          );
        })}

        {/* Ghost atoms at destinations (for atoms with planned moves) */}
        {moves.map((move, i) => {
          const destCenter = getCellCenter(move.to.col, move.to.row);
          const group = moveGroups[i] ?? 0;
          const color = getGroupColor(group);
          
          return (
            <g 
              key={`ghost-${move.atom}`}
              style={{ cursor: isEditable ? 'pointer' : 'default' }}
              onClick={(e) => {
                e.stopPropagation();
                if (isEditable) onMoveClick?.(move);
              }}
            >
              {/* Ghost circle with dashed outline */}
              <circle
                cx={destCenter.x}
                cy={destCenter.y}
                r="16"
                fill={`${color}33`}  // 20% opacity fill
                stroke={color}
                strokeWidth="2"
                strokeDasharray="4,2"
              />
              {/* Ghost label */}
              <text
                x={destCenter.x}
                y={destCenter.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="11"
                fontWeight="bold"
                fill={color}
                opacity="0.7"
              >
                {move.atom}
              </text>
            </g>
          );
        })}

        {/* Solid atoms (only those NOT being moved) */}
        {Object.entries(atoms).map(([idStr, pos]) => {
          const id = parseInt(idStr);
          
          // Skip atoms that have planned moves (they appear as ghosts at destination)
          if (movedAtomIds.has(id)) {
            return null;
          }
          
          const center = getCellCenter(pos.col, pos.row);
          const isSelected = selectedAtom === id;
          const isActive = isAtomActive(id);
          const color = getAtomColor(id);

          return (
            <g
              key={`atom-${id}`}
              style={{ cursor: isEditable && isActive ? 'pointer' : 'default' }}
              onClick={(e) => {
                e.stopPropagation();
                if (isEditable && isActive) onAtomClick?.(id);
              }}
            >
              {/* Selection ring */}
              {isSelected && (
                <circle
                  cx={center.x}
                  cy={center.y}
                  r="22"
                  fill="none"
                  stroke={COLORS.atomSelected}
                  strokeWidth="2"
                  strokeDasharray="4,2"
                />
              )}
              {/* Atom circle */}
              <circle
                cx={center.x}
                cy={center.y}
                r="16"
                fill={color}
                style={{ 
                  filter: isActive ? `drop-shadow(0 0 6px ${color}80)` : 'none',
                  opacity: isActive ? 1 : 0.5,
                }}
              />
              {/* Atom label */}
              <text
                x={center.x}
                y={center.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="11"
                fontWeight="bold"
                fill={isActive ? COLORS.atomText : COLORS.atomTextInactive}
              >
                {id}
              </text>
            </g>
          );
        })}
      </svg>

      {cost !== undefined && (
        <div style={{
          fontSize: '13px',
          color: '#4ecdc4',
          fontWeight: '600',
        }}>
          Cost: {cost} moves
        </div>
      )}
    </div>
  );
};
