// Build animation timeline from simulation data

import { SimulationData, Move, AtomPositions } from '../types';
import { ComputeAllResponse } from '../api';
import { AnimationTimeline, AnimationStep, AnimationMove, PlaybackSettings, DEFAULT_PLAYBACK_SETTINGS } from './types';

export function buildTimeline(
  data: SimulationData,
  computed: ComputeAllResponse,
  settings: PlaybackSettings = DEFAULT_PLAYBACK_SETTINGS
): AnimationTimeline {
  const steps: AnimationStep[] = [];
  let totalDuration = 0;

  // Track atom positions as we build the timeline
  const atomPositions: AtomPositions = {};
  Object.entries(data.board.initialAtoms).forEach(([id, pos]) => {
    atomPositions[parseInt(id)] = { row: pos.row, col: pos.col };
  });

  for (let layerIdx = 0; layerIdx < data.circuit.length; layerIdx++) {
    const layerResult = computed.layers[layerIdx];
    if (!layerResult) continue;

    // --- RECONFIG PHASE ---
    const reconfigMoves = layerResult.reconfigMoves || [];
    const reconfigGroups = layerResult.reconfigGroups || [];

    if (reconfigMoves.length > 0) {
      // Group moves by their parallel group
      const groupedMoves = groupMovesByGroup(reconfigMoves, reconfigGroups);
      const numGroups = Math.max(...reconfigGroups, -1) + 1;

      for (let groupIdx = 0; groupIdx < numGroups; groupIdx++) {
        const movesInGroup = groupedMoves.get(groupIdx) || [];
        if (movesInGroup.length === 0) continue;

        const animMoves: AnimationMove[] = movesInGroup.map(m => ({
          atomId: m.atom,
          from: { row: m.from.row, col: m.from.col },
          to: { row: m.to.row, col: m.to.col },
        }));

        steps.push({
          layerIndex: layerIdx,
          phase: 'reconfig',
          groupIndex: groupIdx,
          moves: animMoves,
          duration: settings.stepDuration,
        });
        totalDuration += settings.stepDuration;
      }

      // Apply reconfig moves to atom positions
      reconfigMoves.forEach(m => {
        atomPositions[m.atom] = { row: m.to.row, col: m.to.col };
      });
    }

    // --- GATE PHASE (single phase: forward + overlap + backward) ---
    const gates = layerResult.canonicalizedGates || layerResult.gates || [];
    const gateGroups = layerResult.gateGroups || [];

    if (gates.length > 0) {
      // Group gates by their parallel group
      const groupedGates = new Map<number, Array<{ gate: number[]; index: number }>>();
      gates.forEach((gate, idx) => {
        const group = gateGroups[idx] ?? 0;
        if (!groupedGates.has(group)) groupedGates.set(group, []);
        groupedGates.get(group)!.push({ gate, index: idx });
      });
      const numGroups = Math.max(...gateGroups, -1) + 1;

      // Single gate phase per group (includes forward, overlap, and backward)
      for (let groupIdx = 0; groupIdx < numGroups; groupIdx++) {
        const gatesInGroup = groupedGates.get(groupIdx) || [];
        if (gatesInGroup.length === 0) continue;

        const animMoves: AnimationMove[] = gatesInGroup.map(({ gate }) => {
          const [atom1, atom2] = gate;
          const pos1 = atomPositions[atom1];
          const pos2 = atomPositions[atom2];
          return {
            atomId: atom1,
            from: { row: pos1.row, col: pos1.col },
            to: { row: pos2.row, col: pos2.col },
            partnerAtomId: atom2,
          };
        });

        steps.push({
          layerIndex: layerIdx,
          phase: 'gate',
          groupIndex: groupIdx,
          moves: animMoves,
          duration: settings.stepDuration * 1.5,  // Slightly longer for full round-trip
        });
        totalDuration += settings.stepDuration * 1.5;
      }
    }
  }

  return { steps, totalDuration };
}

function groupMovesByGroup(moves: Move[], groups: number[]): Map<number, Move[]> {
  const grouped = new Map<number, Move[]>();
  moves.forEach((move, idx) => {
    const group = groups[idx] ?? 0;
    if (!grouped.has(group)) grouped.set(group, []);
    grouped.get(group)!.push(move);
  });
  return grouped;
}

// Get atom positions at a specific point in the timeline
export function getAtomPositionsAtStep(
  data: SimulationData,
  timeline: AnimationTimeline,
  stepIndex: number,
  _stepProgress: number = 0  // Reserved for future interpolation use
): { positions: AtomPositions; movingAtoms: Map<number, AnimationMove> } {
  // Start with initial positions
  const positions: AtomPositions = {};
  Object.entries(data.board.initialAtoms).forEach(([id, pos]) => {
    positions[parseInt(id)] = { row: pos.row, col: pos.col };
  });

  const movingAtoms = new Map<number, AnimationMove>();

  // Apply all completed steps
  for (let i = 0; i < stepIndex && i < timeline.steps.length; i++) {
    const step = timeline.steps[i];
    applyStepToPositions(positions, step);
  }

  // For the current step, track which atoms are moving
  if (stepIndex < timeline.steps.length) {
    const currentStep = timeline.steps[stepIndex];
    currentStep.moves.forEach(move => {
      movingAtoms.set(move.atomId, move);
    });
  }

  return { positions, movingAtoms };
}

function applyStepToPositions(positions: AtomPositions, step: AnimationStep): void {
  if (step.phase === 'reconfig') {
    // Reconfig moves permanently update positions
    step.moves.forEach(move => {
      positions[move.atomId] = { row: move.to.row, col: move.to.col };
    });
  }
  // Gate moves don't permanently change positions (round trip)
}
