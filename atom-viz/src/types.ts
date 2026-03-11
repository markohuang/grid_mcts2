// Core types for the neutral atoms simulator

export interface Position {
  row: number;
  col: number;
}

export interface Move {
  atom: number;
  from: Position;
  to: Position;
}

// A gate is a pair of atom IDs that need to interact
export type Gate = [number, number];

// A layer is a set of gates that can potentially execute together
export type Layer = Gate[];

// Circuit is a sequence of layers
export type Circuit = Layer[];

// Atom positions: atom ID -> position
export type AtomPositions = Record<number, Position>;

// Board configuration
export interface BoardConfig {
  rows: number;
  cols: number;
  initialAtoms: Record<string, Position>;
}

// Simulation data (loaded from JSON)
export interface SimulationData {
  board: BoardConfig;
  circuit: Circuit;
  plan: Move[][];  // plan[layerIndex] = reconfig moves for that layer
}

// Computed results from Python API
export interface LayerResult {
  layerIndex: number;
  gates: Gate[];
  reconfigMoves: Move[];
  reconfigGroups: number[];
  reconfigCost: number;
  gateGroups: number[];
  gateCost: number;
  atomPositions: Record<string, Position>;
}

export interface ComputedData {
  layers: LayerResult[];
  totalCost: number;
  validatedPlan: Move[][];
}

// Position key for easy comparison
export function posKey(pos: Position): string {
  return `${pos.row},${pos.col}`;
}

export function posEqual(a: Position, b: Position): boolean {
  return a.row === b.row && a.col === b.col;
}
