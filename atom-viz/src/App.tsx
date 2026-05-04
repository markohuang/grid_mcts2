import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Grid } from './Grid';
import { Position, Move, AtomPositions, SimulationData, Gate } from './types';
import { computeAllLayers, checkApiHealth, ComputeAllResponse, randomBoard, generatePlan } from './api';
import { PlaybackTab } from './animation';

const STORAGE_KEY = 'atom-viz-state';
const BASELINE_COST_KEY = 'atom-viz-baseline';

type PlanMethod = 'kouhei' | 'smt' | 'mcts';
type CandidateStatus = 'generating' | 'ready' | 'error';

interface PlanCandidate {
  id: string;
  label: string;
  method: PlanMethod;
  data: SimulationData | null;
  computed: ComputeAllResponse | null;
  status: CandidateStatus;
  error?: string;
  elapsed_ms?: number;
  plan_cost?: number;
  created_at: number;
  params?: {
    model_id?: string;
    num_simulations?: number;
    backend?: string;
    device?: string;
  };
}

const methodLabel = (method: PlanMethod) => (
  method === 'kouhei' ? 'Greedy' : method === 'mcts' ? 'Model' : 'SMT'
);

const normalizeGeneratedPlan = (result: {
  board: SimulationData['board'];
  circuit: number[][][];
  plan: Array<Array<{ atom: number; from?: Position; to: Position }>>;
}): SimulationData => ({
  board: result.board,
  circuit: result.circuit.map(layer => layer.map(gate => [gate[0], gate[1]] as Gate)),
  plan: result.circuit.map((_, layerIdx) =>
    (result.plan[layerIdx] || []).map(move => ({
      atom: move.atom,
      from: move.from ?? move.to,
      to: move.to,
    }))
  ),
});

const puzzleSignature = (value: SimulationData | null) => {
  if (!value) return '';
  return JSON.stringify({ board: value.board, circuit: value.circuit });
};

const atomPositionsBeforeTask = (value: SimulationData, taskIdx: number): AtomPositions => {
  const result: AtomPositions = {};
  Object.entries(value.board.initialAtoms).forEach(([id, pos]) => {
    result[parseInt(id)] = { row: pos.row, col: pos.col };
  });
  for (let i = 0; i < taskIdx && i < value.plan.length; i++) {
    (value.plan[i] || []).forEach(move => {
      result[move.atom] = { row: move.to.row, col: move.to.col };
    });
  }
  return result;
};

interface LayerPlanViewProps {
  title: string;
  subtitle?: string;
  data: SimulationData | null;
  computed: ComputeAllResponse | null;
  layerIndex: number;
  compact?: boolean;
}

const LayerPlanView: React.FC<LayerPlanViewProps> = ({ title, subtitle, data, computed, layerIndex, compact = false }) => {
  if (!data || !computed) {
    return (
      <div style={{ padding: '18px', background: '#12171f', border: '1px solid #2a3444', borderRadius: '8px', color: '#6b7280' }}>
        {title} is not ready.
      </div>
    );
  }

  const layer = computed.layers[layerIndex];
  if (!layer) {
    return (
      <div style={{ padding: '18px', background: '#12171f', border: '1px solid #2a3444', borderRadius: '8px', color: '#ff6b6b' }}>
        {title} has no layer {layerIndex + 1}.
      </div>
    );
  }

  const gates = data.circuit[layerIndex] || [];
  const before = atomPositionsBeforeTask(data, layerIndex);
  const after = { ...before };
  (data.plan[layerIndex] || []).forEach(move => {
    after[move.atom] = { row: move.to.row, col: move.to.col };
  });
  const activeAtoms = new Set<number>();
  gates.forEach(([a, b]) => {
    activeAtoms.add(a);
    activeAtoms.add(b);
  });
  const layerCost = (layer.reconfigCost ?? 0) + (layer.gateCost ?? 0);

  return (
    <section style={{ display: 'grid', gap: compact ? '6px' : '12px', background: '#12171f', border: '1px solid #2a3444', borderRadius: '8px', padding: compact ? '8px 10px' : '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: compact ? '13px' : '15px', fontWeight: 700, color: '#e8e8e8' }}>{title}</div>
          {subtitle && <div style={{ marginTop: compact ? '1px' : '3px', fontSize: compact ? '11px' : '12px', color: '#6b7280' }}>{subtitle}</div>}
        </div>
        <div style={{ display: 'flex', gap: '10px', fontSize: compact ? '11px' : '12px', color: '#9ca3af' }}>
          <span>Total {computed.totalCost}</span>
          <span>Layer {layerCost}</span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: compact ? '10px' : '18px', alignItems: 'flex-start', justifyContent: 'center', flexWrap: 'wrap' }}>
        <div style={{ background: '#161c24', padding: compact ? '6px' : '14px', borderRadius: '8px', border: '1px solid #4ecdc455' }}>
          <Grid
            rows={data.board.rows}
            cols={data.board.cols}
            atoms={before}
            moves={layer.reconfigMoves || []}
            moveGroups={layer.reconfigGroups || []}
            gates={[]}
            gateGroups={[]}
            activeAtoms={activeAtoms}
            isEditable={false}
            label="Reconfiguration"
            cost={layer.reconfigCost ?? 0}
            showGateArrows={false}
            compact={compact}
          />
        </div>
        <div style={{ background: '#161c24', padding: compact ? '6px' : '14px', borderRadius: '8px', border: '1px solid #6c5ce755' }}>
          <Grid
            rows={data.board.rows}
            cols={data.board.cols}
            atoms={after}
            moves={[]}
            moveGroups={[]}
            gates={(layer.canonicalizedGates || layer.gates || gates) as Gate[]}
            gateGroups={layer.gateGroups || []}
            activeAtoms={activeAtoms}
            isEditable={false}
            label="Gate Execution"
            cost={layer.gateCost ?? 0}
            showGateArrows={true}
            compact={compact}
          />
        </div>
      </div>
    </section>
  );
};

const App: React.FC = () => {
  const [data, setData] = useState<SimulationData | null>(null);
  const [computed, setComputed] = useState<ComputeAllResponse | null>(null);
  // -1 = board config, 0+ = task index
  const [currentTaskIndex, setCurrentTaskIndex] = useState(-1);
  const [selectedAtom, setSelectedAtom] = useState<number | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [error, setError] = useState<string | null>(null);
  const [editingGates, setEditingGates] = useState(false);
  const [gatesInput, setGatesInput] = useState('');
  const [activeTab, setActiveTab] = useState<'edit' | 'compare' | 'playback'>('edit');
  const [baselineCost, setBaselineCost] = useState<number | null>(null);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [generateParams, setGenerateParams] = useState({ rows: 3, cols: 10, qubits: 9, tasks: 5, gatesPerLayer: 3 });
  const [showPlanModal, setShowPlanModal] = useState(false);
  const [planMethod, setPlanMethod] = useState<PlanMethod>('kouhei');
  const [mctsModelId, setMctsModelId] = useState('v3a01-cycle05');
  const [mctsNumSimulations, setMctsNumSimulations] = useState(400);
  const [planGenerating, setPlanGenerating] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<PlanCandidate[]>([]);
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<string[]>([]);
  const [compareLayerIndex, setCompareLayerIndex] = useState(0);
  const [lastPuzzleSignature, setLastPuzzleSignature] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth().then(ok => {
      setApiStatus(ok ? 'connected' : 'disconnected');
    });
  }, []);

  // Load data from localStorage or data.json
  useEffect(() => {
    const loadData = async () => {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        try {
          setData(JSON.parse(saved));
          return;
        } catch (e) {
          console.log('Failed to parse localStorage');
        }
      }

      try {
        const res = await fetch('/data.json');
        if (!res.ok) throw new Error('No data.json found');
        const json: SimulationData = await res.json();
        while (json.plan.length < json.circuit.length) {
          json.plan.push([]);
        }
        setData(json);
      } catch (err) {
        console.error(err);
        setError('Failed to load data.json');
      }
    };
    loadData();
  }, []);

  // Save to localStorage
  useEffect(() => {
    if (data) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    }
  }, [data]);

  useEffect(() => {
    const nextSignature = puzzleSignature(data);
    if (!nextSignature) return;
    if (!lastPuzzleSignature) {
      setLastPuzzleSignature(nextSignature);
      return;
    }
    if (lastPuzzleSignature !== nextSignature) {
      setCandidates([]);
      setSelectedCandidateIds([]);
      setCompareLayerIndex(0);
      setLastPuzzleSignature(nextSignature);
    }
  }, [data, lastPuzzleSignature]);

  // Compute baseline cost
  useEffect(() => {
    const computeBaseline = async () => {
      if (!data || apiStatus !== 'connected' || baselineCost !== null) return;
      
      const cached = localStorage.getItem(BASELINE_COST_KEY);
      if (cached) {
        setBaselineCost(parseInt(cached, 10));
        return;
      }
      
      try {
        const emptyPlanData = { ...data, plan: data.circuit.map(() => []) };
        const result = await computeAllLayers(emptyPlanData);
        setBaselineCost(result.totalCost);
        localStorage.setItem(BASELINE_COST_KEY, result.totalCost.toString());
      } catch (e) {
        console.log('Failed to compute baseline');
      }
    };
    computeBaseline();
  }, [data, apiStatus, baselineCost]);

  // Compute costs
  useEffect(() => {
    if (!data || apiStatus !== 'connected') return;

    computeAllLayers(data)
      .then(result => {
        setComputed(result);
        if (result.validatedPlan) {
          const newPlan = result.validatedPlan.map((moves: Move[]) =>
            moves.map(m => ({ atom: m.atom, from: m.from, to: m.to }))
          );
          if (JSON.stringify(newPlan) !== JSON.stringify(data.plan)) {
            setData(prev => prev ? { ...prev, plan: newPlan } : prev);
          }
        }
      })
      .catch(err => setError(`API Error: ${err.message}`));
  }, [data, apiStatus]);

  // Validate task index
  useEffect(() => {
    if (data && currentTaskIndex >= data.circuit.length) {
      setCurrentTaskIndex(Math.max(-1, data.circuit.length - 1));
    }
    if (data && compareLayerIndex >= data.circuit.length) {
      setCompareLayerIndex(Math.max(0, data.circuit.length - 1));
    }
  }, [data, currentTaskIndex, compareLayerIndex]);

  // Reset handler — clears all reconfig moves, keeps board and circuit
  const handleReset = useCallback(() => {
    if (!data) return;
    setData(d => d ? { ...d, plan: d.circuit.map(() => []) } : d);
    setCurrentTaskIndex(-1);
    setSelectedAtom(null);
  }, [data]);

  // Board config mode
  const isBoardConfig = currentTaskIndex === -1;

  // Current task data
  const currentGates = !isBoardConfig && data ? (data.circuit[currentTaskIndex] || []) : [];
  const currentTask = !isBoardConfig && computed ? computed.layers[currentTaskIndex] : null;
  const canonicalizedGates = currentTask?.canonicalizedGates || currentGates;

  // Active atoms (for tasks, atoms in gates; for board config, all atoms)
  const activeAtoms = useMemo(() => {
    if (!data) return new Set<number>();
    if (isBoardConfig) {
      return new Set(Object.keys(data.board.initialAtoms).map(Number));
    }
    const active = new Set<number>();
    currentGates.forEach(([a1, a2]) => {
      active.add(a1);
      active.add(a2);
    });
    return active;
  }, [data, isBoardConfig, currentGates]);

  // Atom positions
  const getAtomPositionsBeforeTask = useCallback((taskIdx: number): AtomPositions => {
    if (!data) return {};
    const result: AtomPositions = {};
    Object.entries(data.board.initialAtoms).forEach(([id, pos]) => {
      result[parseInt(id)] = { row: pos.row, col: pos.col };
    });
    for (let i = 0; i < taskIdx && i < data.plan.length; i++) {
      (data.plan[i] || []).forEach(move => {
        result[move.atom] = { row: move.to.row, col: move.to.col };
      });
    }
    return result;
  }, [data]);

  const atomsBeforeReconfig = useMemo(() => 
    isBoardConfig ? getAtomPositionsBeforeTask(0) : getAtomPositionsBeforeTask(currentTaskIndex),
    [getAtomPositionsBeforeTask, currentTaskIndex, isBoardConfig]
  );

  const atomsAfterReconfig = useMemo(() => {
    if (!data || isBoardConfig) return atomsBeforeReconfig;
    const result = { ...atomsBeforeReconfig };
    (data.plan[currentTaskIndex] || []).forEach(move => {
      result[move.atom] = { row: move.to.row, col: move.to.col };
    });
    return result;
  }, [atomsBeforeReconfig, data, currentTaskIndex, isBoardConfig]);

  const currentMoves = !isBoardConfig && data ? (data.plan[currentTaskIndex] || []) : [];

  // Handle atom click
  const handleAtomClick = useCallback((atomId: number) => {
    if (!activeAtoms.has(atomId)) return;
    setSelectedAtom(prev => prev === atomId ? null : atomId);
  }, [activeAtoms]);

  // Handle cell click (board config: move atom; task: add reconfig move)
  const handleCellClick = useCallback((pos: Position) => {
    if (!data || selectedAtom === null) return;
    if (!activeAtoms.has(selectedAtom)) return;

    if (isBoardConfig) {
      // Board config: move atom to new position
      const currentPos = data.board.initialAtoms[selectedAtom];
      if (currentPos?.row === pos.row && currentPos?.col === pos.col) {
        setSelectedAtom(null);
        return;
      }
      
      // Check if position is occupied
      for (const [id, atomPos] of Object.entries(data.board.initialAtoms)) {
        if (parseInt(id) !== selectedAtom && atomPos.row === pos.row && atomPos.col === pos.col) {
          return;
        }
      }

      setData(prev => {
        if (!prev) return prev;
        const newAtoms = { ...prev.board.initialAtoms };
        newAtoms[selectedAtom] = { row: pos.row, col: pos.col };
        return { ...prev, board: { ...prev.board, initialAtoms: newAtoms } };
      });
      setSelectedAtom(null);
      // Clear baseline since board changed
      setBaselineCost(null);
      localStorage.removeItem(BASELINE_COST_KEY);
    } else {
      // Task mode: add reconfig move
      const atomPos = atomsBeforeReconfig[selectedAtom];
      if (!atomPos) return;
      if (atomPos.row === pos.row && atomPos.col === pos.col) {
        setSelectedAtom(null);
        return;
      }

      const plannedDestinations = new Set(
        currentMoves.filter(m => m.atom !== selectedAtom).map(m => `${m.to.row},${m.to.col}`)
      );
      const vacatingAtoms = new Set(currentMoves.map(m => m.atom));

      if (plannedDestinations.has(`${pos.row},${pos.col}`)) return;

      for (const [idStr, atomP] of Object.entries(atomsBeforeReconfig)) {
        const id = parseInt(idStr);
        if (id === selectedAtom) continue;
        if (atomP.row === pos.row && atomP.col === pos.col && !vacatingAtoms.has(id)) return;
      }

      const newMove: Move = {
        atom: selectedAtom,
        from: { row: atomPos.row, col: atomPos.col },
        to: pos,
      };

      setData(prev => {
        if (!prev) return prev;
        const newPlan = [...prev.plan];
        while (newPlan.length <= currentTaskIndex) newPlan.push([]);
        newPlan[currentTaskIndex] = [
          ...newPlan[currentTaskIndex].filter(m => m.atom !== selectedAtom),
          newMove,
        ];
        return { ...prev, plan: newPlan };
      });
      setSelectedAtom(null);
    }
  }, [data, selectedAtom, activeAtoms, isBoardConfig, atomsBeforeReconfig, currentMoves, currentTaskIndex]);

  // Handle move click (remove)
  const handleMoveClick = useCallback((move: Move) => {
    if (isBoardConfig) return;
    setData(prev => {
      if (!prev) return prev;
      const newPlan = [...prev.plan];
      newPlan[currentTaskIndex] = newPlan[currentTaskIndex].filter(m => m.atom !== move.atom);
      return { ...prev, plan: newPlan };
    });
  }, [currentTaskIndex, isBoardConfig]);

  // Gates editing
  const handleEditGates = () => {
    const gatesStr = currentGates.map(g => `[${g[0]},${g[1]}]`).join(', ');
    setGatesInput(`[${gatesStr}]`);
    setEditingGates(true);
  };

  const handleSaveGates = () => {
    try {
      const parsed = JSON.parse(gatesInput);
      if (!Array.isArray(parsed)) throw new Error('Must be an array');

      const validGates: [number, number][] = [];
      for (const gate of parsed) {
        if (!Array.isArray(gate) || gate.length !== 2) throw new Error('Each gate must be [a, b]');
        const [a, b] = gate;
        if (typeof a !== 'number' || typeof b !== 'number') throw new Error('Gate atoms must be numbers');
        if (a < 0 || b < 0) throw new Error('Atom IDs must be non-negative');
        validGates.push([a, b]);
      }

      setData(prev => {
        if (!prev) return prev;
        const newCircuit = [...prev.circuit];
        newCircuit[currentTaskIndex] = validGates;
        const newPlan = [...prev.plan];
        newPlan[currentTaskIndex] = [];
        return { ...prev, circuit: newCircuit, plan: newPlan };
      });
      setEditingGates(false);
      setError(null);
    } catch (e) {
      setError(`Invalid gate format: ${(e as Error).message}`);
    }
  };

  // Add/delete tasks
  const handleAddTask = () => {
    setData(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        circuit: [...prev.circuit, []],
        plan: [...prev.plan, []],
      };
    });
    // Clear baseline since structure changed
    setBaselineCost(null);
    localStorage.removeItem(BASELINE_COST_KEY);
  };

  const handleDeleteLastTask = () => {
    if (!data || data.circuit.length === 0) return;
    const lastTask = data.circuit[data.circuit.length - 1];
    const hasContent = lastTask.length > 0 || (data.plan[data.circuit.length - 1]?.length > 0);
    
    if (hasContent && !confirm('Delete the last task? It has gates or moves.')) return;
    
    setData(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        circuit: prev.circuit.slice(0, -1),
        plan: prev.plan.slice(0, -1),
      };
    });
    if (currentTaskIndex >= data.circuit.length - 1) {
      setCurrentTaskIndex(Math.max(-1, data.circuit.length - 2));
    }
    setBaselineCost(null);
    localStorage.removeItem(BASELINE_COST_KEY);
  };

  // Generate template
  const handleGenerate = () => {
    const { rows, cols, qubits, tasks, gatesPerLayer } = generateParams;
    const maxGates = Math.floor(qubits / 2);
    const actualGates = Math.min(gatesPerLayer, maxGates);

    if (qubits > rows * cols) {
      setError(`Cannot place ${qubits} atoms in ${rows}x${cols} grid`);
      return;
    }

    // Generate random positions
    const allPositions: [number, number][] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        allPositions.push([r, c]);
      }
    }
    const shuffled = allPositions.sort(() => Math.random() - 0.5);
    const positions = shuffled.slice(0, qubits);

    // Generate random gates — fixed gatesPerLayer independent pairs per layer
    const circuit: [number, number][][] = [];
    for (let t = 0; t < tasks; t++) {
      const available = [...Array(qubits).keys()].sort(() => Math.random() - 0.5);
      const layer: [number, number][] = [];
      for (let i = 0; i < actualGates && available.length >= 2; i++) {
        layer.push([available.pop()!, available.pop()!]);
      }
      if (layer.length > 0) circuit.push(layer);
    }

    const initialAtoms: Record<string, { row: number; col: number }> = {};
    positions.forEach((pos, i) => {
      initialAtoms[i.toString()] = { row: pos[0], col: pos[1] };
    });

    const newData: SimulationData = {
      board: { rows, cols, initialAtoms },
      circuit,
      plan: circuit.map(() => []),
    };

    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(BASELINE_COST_KEY);
    setBaselineCost(null);
    setData(newData);
    setCurrentTaskIndex(-1);
    setSelectedAtom(null);
    setShowGenerateModal(false);
  };

  // Random board via backend (5x5, 3 layers, 4 gates/layer)
  const handleRandomBoard = async () => {
    try {
      const result = await randomBoard({ rows: 5, cols: 5, num_qubits: 12, num_layers: 3, gates_per_layer: 4 });
      const newData: SimulationData = { ...result as unknown as SimulationData, plan: result.circuit.map(() => []) };
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(BASELINE_COST_KEY);
      setBaselineCost(null);
      setData(newData);
      setCurrentTaskIndex(-1);
      setSelectedAtom(null);
    } catch (e: any) {
      setError(e.message);
    }
  };

  // Generate plan via backend
  const handleGeneratePlan = async () => {
    if (!data) return;
    const candidateId = `${planMethod}-${Date.now()}`;
    const params = planMethod === 'mcts'
      ? { model_id: mctsModelId, num_simulations: mctsNumSimulations, backend: 'fast', device: 'auto' }
      : undefined;
    const label = planMethod === 'mcts'
      ? `${methodLabel(planMethod)} ${mctsModelId}`
      : methodLabel(planMethod);

    setPlanGenerating(true);
    setPlanError(null);
    setCandidates(prev => [
      {
        id: candidateId,
        label,
        method: planMethod,
        data: null,
        computed: null,
        status: 'generating',
        created_at: Date.now(),
        params,
      },
      ...prev,
    ]);

    try {
      const result = await generatePlan({
        board: data.board,
        circuit: data.circuit,
        method: planMethod,
        ...(planMethod === 'mcts' ? {
          model_id: mctsModelId,
          num_simulations: mctsNumSimulations,
          backend: 'fast',
          device: 'auto',
          deterministic: true,
        } : {}),
      });
      const candidateData = normalizeGeneratedPlan(result);
      const candidateComputed = await computeAllLayers(candidateData);
      setCandidates(prev => prev.map(candidate =>
        candidate.id === candidateId
          ? {
              ...candidate,
              data: candidateData,
              computed: candidateComputed,
              status: 'ready',
              elapsed_ms: result.elapsed_ms,
              plan_cost: result.plan_cost ?? candidateComputed.totalCost,
            }
          : candidate
      ));
      setSelectedCandidateIds(prev => (
        prev.length < 2 && !prev.includes(candidateId) ? [...prev, candidateId] : prev
      ));
      setActiveTab('compare');
      setShowPlanModal(false);
    } catch (e: any) {
      const message = e.message || 'Plan generation failed';
      setPlanError(message);
      setCandidates(prev => prev.map(candidate =>
        candidate.id === candidateId
          ? { ...candidate, status: 'error', error: message }
          : candidate
      ));
      setActiveTab('compare');
      setShowPlanModal(false);
    } finally {
      setPlanGenerating(false);
    }
  };

  const handleToggleCandidate = (candidate: PlanCandidate) => {
    if (candidate.status !== 'ready') return;
    setSelectedCandidateIds(prev => {
      if (prev.includes(candidate.id)) return prev.filter(id => id !== candidate.id);
      if (prev.length >= 2) return prev;
      return [...prev, candidate.id];
    });
  };

  const handleRemoveCandidate = (candidateId: string) => {
    setCandidates(prev => prev.filter(candidate => candidate.id !== candidateId));
    setSelectedCandidateIds(prev => prev.filter(id => id !== candidateId));
  };

  const handleApplyCandidate = (candidate: PlanCandidate) => {
    if (!candidate.data) return;
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(BASELINE_COST_KEY);
    setBaselineCost(null);
    setData(candidate.data);
    setCurrentTaskIndex(-1);
    setSelectedAtom(null);
  };

  // Import/Export
  const handleExport = () => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'atom-viz-export.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        // Basic validation
        if (!json.board || !json.circuit) {
          throw new Error('Invalid format: missing board or circuit');
        }
        while (json.plan?.length < json.circuit.length) {
          json.plan = json.plan || [];
          json.plan.push([]);
        }
        localStorage.removeItem(BASELINE_COST_KEY);
        setBaselineCost(null);
        setData(json);
        setCurrentTaskIndex(-1);
        setSelectedAtom(null);
        setError(null);
      } catch (err) {
        setError(`Import failed: ${(err as Error).message}`);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (editingGates || activeTab !== 'edit') return;

      if (e.key === 'ArrowLeft') {
        setCurrentTaskIndex(i => Math.max(-1, i - 1));
        setSelectedAtom(null);
      } else if (e.key === 'ArrowRight' && data) {
        setCurrentTaskIndex(i => Math.min(data.circuit.length - 1, i + 1));
        setSelectedAtom(null);
      } else if (e.key === 'Escape') {
        setSelectedAtom(null);
        setEditingGates(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [data, editingGates, activeTab]);

  if (!data) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0a0e14',
        color: '#ff6b6b',
        fontSize: '18px',
      }}>
        Failed to load data. Check that public/data.json exists.
      </div>
    );
  }

  // Cost display
  const currentCost = computed?.totalCost;
  const costDiff = currentCost !== undefined && baselineCost !== null ? currentCost - baselineCost : null;
  const costColor = costDiff === null || costDiff === 0 ? '#e8e8e8' : costDiff < 0 ? '#00b894' : '#ff6b6b';
  const costBg = costDiff === null || costDiff === 0 ? '#1e2530' : costDiff < 0 ? '#00b89422' : '#ff6b6b22';
  const selectedCandidates = selectedCandidateIds
    .map(id => candidates.find(candidate => candidate.id === id))
    .filter((candidate): candidate is PlanCandidate => Boolean(candidate));
  const readyCandidateCount = candidates.filter(candidate => candidate.status === 'ready').length;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#0a0e14', color: '#e8e8e8' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px', borderBottom: '1px solid #1e2530' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <h1 style={{ fontSize: '20px', fontWeight: '600', color: '#4ecdc4', margin: 0 }}>
            Neutral Atoms Simulator
          </h1>
          
          {/* Tabs */}
          <div style={{ display: 'flex', gap: '4px', background: '#1e2530', borderRadius: '8px', padding: '4px' }}>
            {(['edit', 'compare', 'playback'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: '8px 20px',
                  fontSize: '14px',
                  fontWeight: '600',
                  background: activeTab === tab ? '#4ecdc4' : 'transparent',
                  color: activeTab === tab ? '#0a0e14' : '#6b7280',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                }}
              >
                {tab === 'compare' ? `Compare${readyCandidateCount ? ` (${readyCandidateCount})` : ''}` : tab}
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            fontSize: '12px',
            padding: '4px 10px',
            borderRadius: '4px',
            background: apiStatus === 'connected' ? '#00b89433' : apiStatus === 'disconnected' ? '#ff6b6b33' : '#ffd93d33',
            color: apiStatus === 'connected' ? '#00b894' : apiStatus === 'disconnected' ? '#ff6b6b' : '#ffd93d',
          }}>
            API: {apiStatus}
          </div>

          <div style={{ fontSize: '14px', fontWeight: '600', background: costBg, padding: '8px 16px', borderRadius: '6px', color: costColor, display: 'flex', gap: '8px' }}>
            <span>Total: {currentCost ?? '?'} moves</span>
            {costDiff !== null && costDiff !== 0 && (
              <span style={{ fontSize: '12px', opacity: 0.9 }}>({costDiff > 0 ? '+' : ''}{costDiff})</span>
            )}
          </div>

          <button onClick={() => setShowGenerateModal(true)} style={{ padding: '8px 12px', fontSize: '13px', background: '#6c5ce7', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
            Generate
          </button>

          <button onClick={handleRandomBoard} disabled={apiStatus !== 'connected'} style={{ padding: '8px 12px', fontSize: '13px', background: '#6c5ce7', color: '#fff', border: 'none', borderRadius: '6px', cursor: apiStatus === 'connected' ? 'pointer' : 'not-allowed', opacity: apiStatus === 'connected' ? 1 : 0.5 }}>
            Random 5×5
          </button>

          <button onClick={() => { setShowPlanModal(true); setPlanError(null); }} disabled={apiStatus !== 'connected' || !data} style={{ padding: '8px 12px', fontSize: '13px', background: '#e17055', color: '#fff', border: 'none', borderRadius: '6px', cursor: (apiStatus === 'connected' && data) ? 'pointer' : 'not-allowed', opacity: (apiStatus === 'connected' && data) ? 1 : 0.5 }}>
            Generate Plan
          </button>

          <input type="file" ref={fileInputRef} onChange={handleImport} accept=".json" style={{ display: 'none' }} />
          <button onClick={() => fileInputRef.current?.click()} style={{ padding: '8px 12px', fontSize: '13px', background: '#2a3444', color: '#e8e8e8', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
            Import
          </button>

          <button onClick={handleExport} style={{ padding: '8px 16px', fontSize: '13px', background: '#4ecdc4', color: '#0a0e14', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: '600' }}>
            Export
          </button>

          <button onClick={handleReset} style={{ padding: '8px 12px', fontSize: '13px', background: '#2a3444', color: '#e8e8e8', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
            Reset
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div style={{ padding: '12px 24px', background: '#ff6b6b22', borderBottom: '1px solid #ff6b6b', color: '#ff6b6b', display: 'flex', justifyContent: 'space-between' }}>
          {error}
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: '#ff6b6b', cursor: 'pointer', fontSize: '18px' }}>×</button>
        </div>
      )}

      {/* API warning */}
      {apiStatus === 'disconnected' && (
        <div style={{ padding: '12px 24px', background: '#ffd93d22', borderBottom: '1px solid #ffd93d', color: '#ffd93d', textAlign: 'center' }}>
          Python API not connected. Run: <code style={{ background: '#1e2530', padding: '2px 6px', borderRadius: '4px' }}>python api.py</code>
        </div>
      )}

      {/* Main Content */}
      <div style={{ flex: 1, padding: '20px', overflow: 'auto' }}>
        {activeTab === 'edit' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {/* Task Navigation */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '20px' }}>
              <button
                onClick={() => { setCurrentTaskIndex(i => i - 1); setSelectedAtom(null); }}
                disabled={currentTaskIndex === -1}
                style={{
                  padding: '8px 16px',
                  background: currentTaskIndex === -1 ? '#2a3444' : '#4ecdc4',
                  color: currentTaskIndex === -1 ? '#6b7280' : '#0a0e14',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: currentTaskIndex === -1 ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                }}
              >
                ← Prev
              </button>

              <div style={{ textAlign: 'center', minWidth: '180px' }}>
                <div style={{ fontSize: '18px', fontWeight: '600', color: isBoardConfig ? '#ffd93d' : '#e8e8e8' }}>
                  {isBoardConfig ? 'Board Configuration' : `Task ${currentTaskIndex + 1} / ${data.circuit.length}`}
                </div>
              </div>

              <button
                onClick={() => { setCurrentTaskIndex(i => i + 1); setSelectedAtom(null); }}
                disabled={currentTaskIndex === data.circuit.length - 1}
                style={{
                  padding: '8px 16px',
                  background: currentTaskIndex === data.circuit.length - 1 ? '#2a3444' : '#4ecdc4',
                  color: currentTaskIndex === data.circuit.length - 1 ? '#6b7280' : '#0a0e14',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: currentTaskIndex === data.circuit.length - 1 ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                }}
              >
                Next →
              </button>
            </div>

            {/* Gates Display (only for tasks) */}
            {!isBoardConfig && (
              <div style={{ textAlign: 'center', padding: '8px 16px', background: '#161c24', borderRadius: '8px', margin: '0 auto', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ color: '#6b7280' }}>Gates:</span>
                {editingGates ? (
                  <>
                    <input
                      type="text"
                      value={gatesInput}
                      onChange={e => setGatesInput(e.target.value)}
                      style={{ background: '#0a0e14', border: '1px solid #4ecdc4', borderRadius: '4px', padding: '4px 8px', color: '#e8e8e8', fontFamily: 'monospace', fontSize: '14px', minWidth: '300px' }}
                      onKeyDown={e => { if (e.key === 'Enter') handleSaveGates(); if (e.key === 'Escape') setEditingGates(false); }}
                    />
                    <button onClick={handleSaveGates} style={{ padding: '4px 12px', background: '#00b894', color: '#0a0e14', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: '600' }}>Save</button>
                    <button onClick={() => setEditingGates(false)} style={{ padding: '4px 12px', background: '#ff6b6b', color: '#0a0e14', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: '600' }}>Cancel</button>
                  </>
                ) : (
                  <>
                    <span style={{ fontFamily: 'monospace', color: '#4ecdc4' }}>
                      {currentGates.length > 0 ? currentGates.map(g => `[${g[0]},${g[1]}]`).join('  ') : '(none)'}
                    </span>
                    <button onClick={handleEditGates} style={{ padding: '4px 12px', background: '#2a3444', color: '#e8e8e8', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '12px' }}>Edit</button>
                  </>
                )}
              </div>
            )}

            {/* Grids */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '40px', alignItems: 'flex-start' }}>
              {isBoardConfig ? (
                /* Board Config: Single grid */
                <div style={{ background: '#161c24', padding: '20px', borderRadius: '12px', border: '2px solid #ffd93d' }}>
                  <Grid
                    rows={data.board.rows}
                    cols={data.board.cols}
                    atoms={atomsBeforeReconfig}
                    moves={[]}
                    moveGroups={[]}
                    gates={[]}
                    gateGroups={[]}
                    activeAtoms={activeAtoms}
                    isEditable={true}
                    selectedAtom={selectedAtom}
                    onAtomClick={handleAtomClick}
                    onCellClick={handleCellClick}
                    label="Initial Board"
                    cost={0}
                    showGateArrows={false}
                  />
                </div>
              ) : (
                /* Task mode: Two grids */
                <>
                  <div style={{ background: '#161c24', padding: '20px', borderRadius: '12px', border: '2px solid #4ecdc4' }}>
                    <Grid
                      rows={data.board.rows}
                      cols={data.board.cols}
                      atoms={atomsBeforeReconfig}
                      moves={currentMoves}
                      moveGroups={currentTask?.reconfigGroups || []}
                      gates={[]}
                      gateGroups={[]}
                      activeAtoms={activeAtoms}
                      isEditable={true}
                      selectedAtom={selectedAtom}
                      onAtomClick={handleAtomClick}
                      onCellClick={handleCellClick}
                      onMoveClick={handleMoveClick}
                      label="Reconfiguration"
                      cost={currentTask?.reconfigCost ?? 0}
                      showGateArrows={false}
                    />
                  </div>

                  <div style={{ background: '#161c24', padding: '20px', borderRadius: '12px', border: '1px solid #6c5ce7' }}>
                    <Grid
                      rows={data.board.rows}
                      cols={data.board.cols}
                      atoms={atomsAfterReconfig}
                      moves={[]}
                      moveGroups={[]}
                      gates={canonicalizedGates as [number, number][]}
                      gateGroups={currentTask?.gateGroups || []}
                      activeAtoms={activeAtoms}
                      isEditable={false}
                      label="Gate Execution"
                      cost={currentTask?.gateCost ?? 0}
                      showGateArrows={true}
                    />
                  </div>
                </>
              )}
            </div>

            {/* Instructions */}
            <div style={{ fontSize: '13px', color: '#6b7280', textAlign: 'center', maxWidth: '700px', margin: '0 auto' }}>
              {isBoardConfig
                ? 'Click atom to select, then click empty cell to move. Configure initial positions before defining tasks.'
                : 'Click colored atoms to select, then click empty cell to add move. Click ghost atom or arrow to remove. Same color = parallel.'}
            </div>

            {/* Task Tabs */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', padding: '12px', background: '#12171f', borderRadius: '8px', overflowX: 'auto', alignItems: 'center' }}>
              {/* Board Config Tab */}
              <button
                onClick={() => { setCurrentTaskIndex(-1); setSelectedAtom(null); setEditingGates(false); }}
                style={{
                  padding: '8px 16px',
                  minWidth: '80px',
                  fontSize: '12px',
                  background: isBoardConfig ? '#ffd93d' : '#2a3444',
                  color: isBoardConfig ? '#0a0e14' : '#e8e8e8',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: '600',
                }}
              >
                Board
              </button>

              <div style={{ width: '1px', height: '30px', background: '#4a5568', margin: '0 4px' }} />

              {/* Task Tabs */}
              {data.circuit.map((gates, i) => {
                const task = computed?.layers[i];
                const taskCost = (task?.reconfigCost ?? 0) + (task?.gateCost ?? 0);
                return (
                  <button
                    key={i}
                    onClick={() => { setCurrentTaskIndex(i); setSelectedAtom(null); setEditingGates(false); }}
                    style={{
                      padding: '8px 16px',
                      minWidth: '100px',
                      fontSize: '12px',
                      background: i === currentTaskIndex ? '#4ecdc4' : '#2a3444',
                      color: i === currentTaskIndex ? '#0a0e14' : '#e8e8e8',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    <span style={{ fontWeight: '600' }}>Task {i + 1}</span>
                    <span style={{ fontSize: '10px', color: i === currentTaskIndex ? '#0a0e14' : '#6b7280' }}>
                      {gates.length} gates · {taskCost} moves
                    </span>
                  </button>
                );
              })}

              <div style={{ width: '1px', height: '30px', background: '#4a5568', margin: '0 4px' }} />

              {/* Add/Delete buttons */}
              <button
                onClick={handleAddTask}
                style={{ padding: '8px 12px', fontSize: '16px', background: '#00b894', color: '#0a0e14', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: '600' }}
                title="Add Task"
              >
                +
              </button>
              <button
                onClick={handleDeleteLastTask}
                disabled={data.circuit.length === 0}
                style={{
                  padding: '8px 12px',
                  fontSize: '16px',
                  background: data.circuit.length === 0 ? '#2a3444' : '#ff6b6b',
                  color: data.circuit.length === 0 ? '#6b7280' : '#0a0e14',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: data.circuit.length === 0 ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                }}
                title="Delete Last Task"
              >
                −
              </button>
            </div>
          </div>
        ) : activeTab === 'compare' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 340px) 1fr', gap: '18px', alignItems: 'start' }}>
            <aside style={{ display: 'grid', gap: '12px', background: '#12171f', border: '1px solid #2a3444', borderRadius: '8px', padding: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: '#e8e8e8' }}>Plan Candidates</div>
                  <div style={{ marginTop: '3px', fontSize: '12px', color: '#6b7280' }}>Select up to two ready results.</div>
                </div>
                <button
                  onClick={() => { setShowPlanModal(true); setPlanError(null); }}
                  disabled={apiStatus !== 'connected'}
                  style={{ padding: '7px 10px', fontSize: '12px', background: '#e17055', color: '#fff', border: 'none', borderRadius: '6px', cursor: apiStatus === 'connected' ? 'pointer' : 'not-allowed', opacity: apiStatus === 'connected' ? 1 : 0.5 }}
                >
                  Generate
                </button>
              </div>

              {candidates.length === 0 ? (
                <div style={{ padding: '16px', background: '#0a0e14', border: '1px dashed #2a3444', borderRadius: '8px', color: '#6b7280', fontSize: '13px' }}>
                  Generate a greedy or model result to compare plans.
                </div>
              ) : (
                <div style={{ display: 'grid', gap: '8px' }}>
                  {candidates.map(candidate => {
                    const selected = selectedCandidateIds.includes(candidate.id);
                    const disabled = candidate.status !== 'ready' || (!selected && selectedCandidateIds.length >= 2);
                    const statusColor = candidate.status === 'ready' ? '#00b894' : candidate.status === 'error' ? '#ff6b6b' : '#ffd93d';
                    return (
                      <div key={candidate.id} style={{ display: 'grid', gap: '8px', padding: '10px', background: selected ? '#163331' : '#0a0e14', border: `1px solid ${selected ? '#4ecdc4' : '#2a3444'}`, borderRadius: '8px' }}>
                        <div style={{ display: 'flex', gap: '8px', alignItems: 'start' }}>
                          <input
                            type="checkbox"
                            checked={selected}
                            disabled={disabled}
                            onChange={() => handleToggleCandidate(candidate)}
                            style={{ marginTop: '3px' }}
                          />
                          <div style={{ minWidth: 0, flex: 1 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                              <span style={{ fontSize: '13px', fontWeight: 700, color: '#e8e8e8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{candidate.label}</span>
                              <span style={{ fontSize: '11px', color: statusColor, textTransform: 'uppercase' }}>{candidate.status}</span>
                            </div>
                            <div style={{ marginTop: '4px', fontSize: '12px', color: '#6b7280' }}>
                              {candidate.status === 'ready' && `Cost ${candidate.plan_cost ?? candidate.computed?.totalCost ?? '?'} · ${candidate.elapsed_ms?.toFixed(0) ?? '?'} ms`}
                              {candidate.status === 'generating' && 'Waiting for backend result...'}
                              {candidate.status === 'error' && candidate.error}
                            </div>
                            {candidate.params && (
                              <div style={{ marginTop: '3px', fontSize: '11px', color: '#6b7280' }}>
                                {candidate.params.num_simulations} sims · {candidate.params.backend} · {candidate.params.device}
                              </div>
                            )}
                          </div>
                        </div>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button
                            onClick={() => handleApplyCandidate(candidate)}
                            disabled={!candidate.data}
                            style={{ flex: 1, padding: '6px 8px', fontSize: '12px', background: candidate.data ? '#2a3444' : '#1e2530', color: candidate.data ? '#e8e8e8' : '#4a5568', border: 'none', borderRadius: '6px', cursor: candidate.data ? 'pointer' : 'not-allowed' }}
                          >
                            Apply
                          </button>
                          <button
                            onClick={() => handleRemoveCandidate(candidate.id)}
                            style={{ padding: '6px 8px', fontSize: '12px', background: '#2a1a1a', color: '#ff7675', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </aside>

            <section style={{ display: 'grid', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', alignItems: 'center', background: '#12171f', border: '1px solid #2a3444', borderRadius: '8px', padding: '8px 12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '13px', color: '#6b7280' }}>Layer</span>
                  <button
                    onClick={() => setCompareLayerIndex(i => Math.max(0, i - 1))}
                    disabled={compareLayerIndex === 0}
                    style={{ padding: '6px 10px', background: compareLayerIndex === 0 ? '#1e2530' : '#2a3444', color: compareLayerIndex === 0 ? '#4a5568' : '#e8e8e8', border: 'none', borderRadius: '6px', cursor: compareLayerIndex === 0 ? 'not-allowed' : 'pointer' }}
                  >
                    Prev
                  </button>
                  <span style={{ minWidth: '88px', textAlign: 'center', fontSize: '14px', fontWeight: 700 }}>
                    {compareLayerIndex + 1} / {data.circuit.length}
                  </span>
                  <button
                    onClick={() => setCompareLayerIndex(i => Math.min(data.circuit.length - 1, i + 1))}
                    disabled={compareLayerIndex >= data.circuit.length - 1}
                    style={{ padding: '6px 10px', background: compareLayerIndex >= data.circuit.length - 1 ? '#1e2530' : '#2a3444', color: compareLayerIndex >= data.circuit.length - 1 ? '#4a5568' : '#e8e8e8', border: 'none', borderRadius: '6px', cursor: compareLayerIndex >= data.circuit.length - 1 ? 'not-allowed' : 'pointer' }}
                  >
                    Next
                  </button>
                </div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                  {selectedCandidates.map(candidate => (
                    <span key={candidate.id} style={{ padding: '5px 9px', borderRadius: '999px', background: '#1e2530', color: '#9ca3af', fontSize: '12px' }}>
                      {candidate.label}
                    </span>
                  ))}
                </div>
              </div>

              {selectedCandidates.length === 0 ? (
                <div style={{ minHeight: '360px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#12171f', border: '1px dashed #2a3444', borderRadius: '8px', color: '#6b7280' }}>
                  Select one or two ready candidates to inspect their plans.
                </div>
              ) : (
                <div style={{ display: 'grid', gridTemplateRows: selectedCandidates.length === 2 ? '1fr 1fr' : '1fr', gap: selectedCandidates.length === 2 ? '8px' : '14px' }}>
                  {selectedCandidates.map(candidate => (
                    <LayerPlanView
                      key={candidate.id}
                      title={candidate.label}
                      subtitle={`${methodLabel(candidate.method)} · ${candidate.plan_cost ?? candidate.computed?.totalCost ?? '?'} total moves`}
                      data={candidate.data}
                      computed={candidate.computed}
                      layerIndex={compareLayerIndex}
                      compact={selectedCandidates.length === 2}
                    />
                  ))}
                </div>
              )}
            </section>
          </div>
        ) : (
          <PlaybackTab data={data} computed={computed} />
        )}
      </div>

      {/* Generate Plan Modal */}
      {showPlanModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <div style={{ background: '#161c24', padding: '24px', borderRadius: '12px', border: '1px solid #e17055', minWidth: '360px', maxWidth: '480px' }}>
            <h2 style={{ margin: '0 0 20px', color: '#e17055' }}>Generate Plan</h2>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ color: '#6b7280', fontSize: '13px', display: 'block', marginBottom: '8px' }}>Method</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {(['kouhei', 'smt', 'mcts'] as const).map(m => (
                  <button key={m} onClick={() => setPlanMethod(m)} style={{ flex: 1, padding: '8px', background: planMethod === m ? '#e17055' : '#2a3444', color: planMethod === m ? '#fff' : '#9ca3af', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: planMethod === m ? '600' : '400', textTransform: 'uppercase', fontSize: '12px' }}>
                    {m}
                  </button>
                ))}
              </div>
              <div style={{ marginTop: '8px', fontSize: '12px', color: '#6b7280' }}>
                {planMethod === 'kouhei' && 'Greedy placement using gate parallelism gain vectors. Fast, no model required.'}
                {planMethod === 'smt' && 'Z3 SMT solver — optimal within each layer under the exact AOD non-crossing constraint. Slower (~0.5–2s for 5×5).'}
                {planMethod === 'mcts' && 'MCTS with trained neural network. Requires a trained model checkpoint.'}
              </div>
            </div>

            {planMethod === 'mcts' && (
              <div style={{ display: 'grid', gap: '12px', marginBottom: '16px' }}>
                <label style={{ color: '#6b7280', fontSize: '13px', display: 'grid', gap: '6px' }}>
                  Model ID
                  <input
                    type="text"
                    value={mctsModelId}
                    onChange={e => setMctsModelId(e.target.value)}
                    style={{ background: '#0a0e14', border: '1px solid #2a3444', borderRadius: '4px', padding: '8px', color: '#e8e8e8', fontSize: '14px' }}
                  />
                </label>
                <label style={{ color: '#6b7280', fontSize: '13px', display: 'grid', gap: '6px' }}>
                  Simulations
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={mctsNumSimulations}
                    onChange={e => setMctsNumSimulations(Math.max(1, parseInt(e.target.value) || 1))}
                    style={{ background: '#0a0e14', border: '1px solid #2a3444', borderRadius: '4px', padding: '8px', color: '#e8e8e8', fontSize: '14px' }}
                  />
                </label>
              </div>
            )}

            {planError && (
              <div style={{ background: '#2a1a1a', border: '1px solid #e17055', borderRadius: '6px', padding: '10px', marginBottom: '12px', color: '#ff7675', fontSize: '13px' }}>
                {planError}
              </div>
            )}

            <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
              <button onClick={() => setShowPlanModal(false)} style={{ flex: 1, padding: '10px', background: '#2a3444', color: '#e8e8e8', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
                Cancel
              </button>
              <button onClick={handleGeneratePlan} disabled={planGenerating} style={{ flex: 1, padding: '10px', background: '#e17055', color: '#fff', border: 'none', borderRadius: '6px', cursor: planGenerating ? 'not-allowed' : 'pointer', fontWeight: '600', opacity: planGenerating ? 0.7 : 1 }}>
                {planGenerating ? 'Generating…' : 'Generate Candidate'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Generate Modal */}
      {showGenerateModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <div style={{ background: '#161c24', padding: '24px', borderRadius: '12px', border: '1px solid #4ecdc4', minWidth: '320px' }}>
            <h2 style={{ margin: '0 0 20px', color: '#4ecdc4' }}>Generate New Problem</h2>
            
            {[
              { label: 'Rows', key: 'rows', min: 1, max: 20 },
              { label: 'Columns', key: 'cols', min: 1, max: 20 },
              { label: 'Qubits', key: 'qubits', min: 1, max: 50 },
              { label: 'Layers', key: 'tasks', min: 1, max: 20 },
              { label: 'Gates/Layer', key: 'gatesPerLayer', min: 1, max: 25 },
            ].map(({ label, key, min, max }) => (
              <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <label style={{ width: '80px', color: '#6b7280' }}>{label}:</label>
                <input
                  type="number"
                  min={min}
                  max={max}
                  value={generateParams[key as keyof typeof generateParams]}
                  onChange={e => setGenerateParams(p => ({ ...p, [key]: parseInt(e.target.value) || min }))}
                  style={{ flex: 1, background: '#0a0e14', border: '1px solid #2a3444', borderRadius: '4px', padding: '8px', color: '#e8e8e8', fontSize: '14px' }}
                />
              </div>
            ))}

            <div style={{ display: 'flex', gap: '12px', marginTop: '20px' }}>
              <button onClick={() => setShowGenerateModal(false)} style={{ flex: 1, padding: '10px', background: '#2a3444', color: '#e8e8e8', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
                Cancel
              </button>
              <button onClick={handleGenerate} style={{ flex: 1, padding: '10px', background: '#4ecdc4', color: '#0a0e14', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: '600' }}>
                Generate
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
