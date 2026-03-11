// Animation-specific types

import { Position } from '../types';

// A single animation step (one parallel group executing)
export interface AnimationStep {
  layerIndex: number;
  phase: 'reconfig' | 'gate';  // Simplified: gate includes forward + overlap + backward
  groupIndex: number;
  moves: AnimationMove[];  // Atoms moving in this step
  duration: number;        // Duration in ms
}

// Move with additional info for animation
export interface AnimationMove {
  atomId: number;
  from: Position;
  to: Position;
  // For gate moves, track the partner atom
  partnerAtomId?: number;
}

// Full timeline of all steps
export interface AnimationTimeline {
  steps: AnimationStep[];
  totalDuration: number;
}

// Current animation state
export interface AnimationState {
  status: 'idle' | 'playing' | 'paused' | 'complete';
  currentStepIndex: number;
  stepProgress: number;  // 0 to 1 within current step
  globalProgress: number; // 0 to 1 across entire timeline
}

// Interpolated atom position during animation
export interface AnimatedAtom {
  atomId: number;
  x: number;  // Interpolated column
  y: number;  // Interpolated row
  phase: 'idle' | 'pickup' | 'moving' | 'settling' | 'gate-overlap';
  color: string;
  // For gate overlap, show both atom IDs
  overlappingWith?: number;
}

// Laser line for visualization
export interface LaserLine {
  type: 'horizontal' | 'vertical';
  // For horizontal: row is fixed, col goes from start to end
  // For vertical: col is fixed, row goes from start to end
  fixed: number;      // The fixed row (for H) or col (for V)
  start: number;      // Start position
  end: number;        // End position
  atomIds: number[];  // Atoms on this laser
  color: string;
  opacity: number;
}

// Playback settings
export interface PlaybackSettings {
  speed: number;        // 0.5, 1, 2, etc.
  stepDuration: number; // Base duration per step in ms
  pickupDuration: number;
  settlingDuration: number;
}

export const DEFAULT_PLAYBACK_SETTINGS: PlaybackSettings = {
  speed: 1,
  stepDuration: 1200,    // Slower base speed
  pickupDuration: 250,
  settlingDuration: 200,
};
