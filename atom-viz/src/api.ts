// API client for Python backend

const API_BASE = 'http://localhost:8000';

export interface Position {
  row: number;
  col: number;
}

export interface MoveInput {
  atom: number;
  frm: Position;
  to: Position;
}

export interface ComputeRequest {
  reconfig?: {
    moves: MoveInput[];
  };
  gate?: {
    gates: number[][];
    atomPositions: Record<string, Position>;
  };
}

export interface GroupResult {
  groups: number[];
  cost: number;
  numGroups: number;
}

export interface ComputeResponse {
  reconfigResult: GroupResult | null;
  gateResult: GroupResult | null;
  totalCost: number;
}

export interface LayerResult {
  layerIndex: number;
  gates: number[][];
  canonicalizedGates: number[][];  // Gates after canonicalization
  reconfigMoves: Array<{ atom: number; from: Position; to: Position }>;
  reconfigGroups: number[];
  reconfigCost: number;
  gateGroups: number[];
  gateCost: number;
  atomPositions: Record<string, Position>;
}

export interface ComputeAllResponse {
  layers: LayerResult[];
  totalCost: number;
  validatedPlan: Array<Array<{ atom: number; from: Position; to: Position }>>;
}

export interface SimulationData {
  board: {
    rows: number;
    cols: number;
    initialAtoms: Record<string, Position>;
  };
  circuit: number[][][];
  plan: Array<Array<{ atom: number; from?: Position; to: Position }>>;
}

/**
 * Compute parallel groups for a single reconfig + gate pair
 */
export async function computeGroups(request: ComputeRequest): Promise<ComputeResponse> {
  const response = await fetch(`${API_BASE}/compute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Compute costs for all layers given full simulation data
 */
export async function computeAllLayers(data: SimulationData): Promise<ComputeAllResponse> {
  const response = await fetch(`${API_BASE}/compute_all_layers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}

export interface GeneratePlanResponse {
  board: SimulationData['board'];
  circuit: number[][][];
  plan: Array<Array<{ atom: number; from?: { row: number; col: number }; to: { row: number; col: number } }>>;
  method: string;
  elapsed_ms: number;
  plan_cost?: number;  // optimal cost (chromatic number) — matches display
}

/**
 * Generate a random board and circuit via the backend
 */
export async function randomBoard(params: {
  rows?: number;
  cols?: number;
  num_qubits?: number;
  num_layers?: number;
  gates_per_layer?: number;
  seed?: number;
}): Promise<{ board: SimulationData['board']; circuit: number[][][] }> {
  const response = await fetch(`${API_BASE}/random_board`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error(`API error: ${response.status}`);
  return response.json();
}

/**
 * Generate a plan using kohei, dpqa, or mcts
 */
export async function generatePlan(params: {
  board: SimulationData['board'];
  circuit: number[][][];
  method: 'kouhei' | 'smt' | 'mcts';
  checkpoint_path?: string;
  num_simulations?: number;
  smt_timeout_ms?: number;
}): Promise<GeneratePlanResponse> {
  const response = await fetch(`${API_BASE}/generate_plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail ?? `API error: ${response.status}`);
  }
  return response.json();
}

/**
 * Check if API is available
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/`);
    return response.ok;
  } catch {
    return false;
  }
}
