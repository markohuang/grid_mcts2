// Playback tab with animation controls and timeline

import React, { useMemo } from 'react';
import { SimulationData, AtomPositions } from '../types';
import { ComputeAllResponse } from '../api';
import { AnimatedGrid } from './AnimatedGrid';
import { useAnimation } from './useAnimation';
import { buildTimeline, getAtomPositionsAtStep } from './timelineBuilder';
import { AnimationTimeline } from './types';

interface PlaybackTabProps {
  data: SimulationData;
  computed: ComputeAllResponse | null;
}

export const PlaybackTab: React.FC<PlaybackTabProps> = ({ data, computed }) => {
  // Build timeline from data
  const timeline = useMemo<AnimationTimeline | null>(() => {
    if (!computed) return null;
    return buildTimeline(data, computed);
  }, [data, computed]);

  // Animation state and controls
  const {
    state,
    play,
    pause,
    stop,
    stepForward,
    stepBackward,
    seekToProgress,
    setSpeed,
    settings,
  } = useAnimation(timeline);

  // Get current positions and step
  const { basePositions, currentStep } = useMemo(() => {
    if (!timeline) {
      // Build initial positions
      const positions: AtomPositions = {};
      Object.entries(data.board.initialAtoms).forEach(([id, pos]) => {
        positions[parseInt(id)] = { row: pos.row, col: pos.col };
      });
      return { basePositions: positions, currentStep: null };
    }

    const { positions } = getAtomPositionsAtStep(data, timeline, state.currentStepIndex, state.stepProgress);
    const step = timeline.steps[state.currentStepIndex] || null;
    return { basePositions: positions, currentStep: step };
  }, [data, timeline, state.currentStepIndex, state.stepProgress]);

  // Get step description and phase
  const getStepInfo = () => {
    if (!currentStep) return { description: 'Ready', isGate: false };
    
    const taskNum = currentStep.layerIndex + 1;
    const groupNum = currentStep.groupIndex + 1;
    const isGate = currentStep.phase === 'gate';
    
    const description = isGate
      ? `Task ${taskNum}: Gate Execution (Group ${groupNum})`
      : `Task ${taskNum}: Reconfiguration (Group ${groupNum})`;
    
    return { description, isGate };
  };

  const stepInfo = getStepInfo();

  if (!computed) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: '400px',
        color: '#6b7280',
      }}>
        Waiting for API response...
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '24px',
      padding: '20px',
    }}>
      {/* Step description with phase indicator */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
      }}>
        {currentStep && (
          <div style={{
            padding: '4px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: '600',
            background: stepInfo.isGate ? '#6c5ce733' : '#4ecdc433',
            color: stepInfo.isGate ? '#a29bfe' : '#4ecdc4',
            border: `1px solid ${stepInfo.isGate ? '#6c5ce755' : '#4ecdc455'}`,
          }}>
            {stepInfo.isGate ? 'GATE' : 'RECONFIG'}
          </div>
        )}
        <div style={{
          fontSize: '18px',
          fontWeight: '600',
          color: stepInfo.isGate ? '#a29bfe' : '#4ecdc4',
          minHeight: '28px',
        }}>
          {stepInfo.description}
        </div>
      </div>

      {/* Animated Grid */}
      <AnimatedGrid
        rows={data.board.rows}
        cols={data.board.cols}
        basePositions={basePositions}
        currentStep={currentStep}
        stepProgress={state.stepProgress}
        settings={settings}
      />

      {/* Timeline */}
      <div style={{ width: '100%', maxWidth: '700px' }}>
        {/* Progress bar */}
        <div
          style={{
            position: 'relative',
            height: '24px',
            background: '#1e2530',
            borderRadius: '12px',
            cursor: 'pointer',
            overflow: 'hidden',
          }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const progress = (e.clientX - rect.left) / rect.width;
            seekToProgress(progress);
          }}
        >
          {/* Progress fill */}
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: `${state.globalProgress * 100}%`,
              background: 'linear-gradient(90deg, #4ecdc4, #6c5ce7)',
              borderRadius: '12px',
              transition: state.status === 'playing' ? 'none' : 'width 0.1s',
            }}
          />
          
          {/* Playhead */}
          <div
            style={{
              position: 'absolute',
              left: `${state.globalProgress * 100}%`,
              top: '50%',
              transform: 'translate(-50%, -50%)',
              width: '16px',
              height: '16px',
              background: '#ffffff',
              borderRadius: '50%',
              boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
            }}
          />
        </div>
      </div>

      {/* Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
      }}>
        {/* Stop */}
        <button
          onClick={stop}
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            border: 'none',
            background: '#2a3444',
            color: '#e8e8e8',
            cursor: 'pointer',
            fontSize: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title="Stop"
        >
          ⏹
        </button>

        {/* Step backward */}
        <button
          onClick={stepBackward}
          disabled={state.currentStepIndex === 0 && state.stepProgress === 0}
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            border: 'none',
            background: '#2a3444',
            color: state.currentStepIndex === 0 ? '#6b7280' : '#e8e8e8',
            cursor: state.currentStepIndex === 0 ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title="Previous Step"
        >
          ⏮
        </button>

        {/* Play/Pause */}
        <button
          onClick={state.status === 'playing' ? pause : play}
          style={{
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            border: 'none',
            background: '#4ecdc4',
            color: '#0a0e14',
            cursor: 'pointer',
            fontSize: '24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(78, 205, 196, 0.3)',
          }}
          title={state.status === 'playing' ? 'Pause' : 'Play'}
        >
          {state.status === 'playing' ? '⏸' : '▶'}
        </button>

        {/* Step forward */}
        <button
          onClick={stepForward}
          disabled={!timeline || state.currentStepIndex >= timeline.steps.length - 1}
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            border: 'none',
            background: '#2a3444',
            color: !timeline || state.currentStepIndex >= timeline.steps.length - 1 ? '#6b7280' : '#e8e8e8',
            cursor: !timeline || state.currentStepIndex >= timeline.steps.length - 1 ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title="Next Step"
        >
          ⏭
        </button>

        {/* Speed selector */}
        <div style={{
          marginLeft: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <span style={{ fontSize: '12px', color: '#6b7280' }}>Speed:</span>
          {[0.5, 1, 2].map(speed => (
            <button
              key={speed}
              onClick={() => setSpeed(speed)}
              style={{
                padding: '4px 12px',
                borderRadius: '4px',
                border: 'none',
                background: settings.speed === speed ? '#4ecdc4' : '#2a3444',
                color: settings.speed === speed ? '#0a0e14' : '#e8e8e8',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: '600',
              }}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div style={{
        display: 'flex',
        gap: '24px',
        fontSize: '13px',
        color: '#6b7280',
      }}>
        <div>
          Step: {state.currentStepIndex + 1} / {timeline?.steps.length || 0}
        </div>
        <div>
          Total Cost: {computed.totalCost} moves
        </div>
        <div>
          Status: {state.status}
        </div>
      </div>

      {/* Legend */}
      <div style={{
        display: 'flex',
        gap: '24px',
        fontSize: '12px',
        color: '#6b7280',
        padding: '12px 20px',
        background: '#161c24',
        borderRadius: '8px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '20px', height: '3px', background: '#ff6b6b', borderRadius: '2px' }} />
          <span>H laser (traps vertically)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '3px', height: '20px', background: '#4ecdc4', borderRadius: '2px' }} />
          <span>V laser (traps horizontally)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '16px', height: '16px', background: '#ffd93d', borderRadius: '50%' }} />
          <span>Gate overlap</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '16px', height: '16px', background: '#0f1a1f', borderRadius: '4px', border: '1px solid #4ecdc455' }} />
          <span>Reconfig</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '16px', height: '16px', background: '#1a0f1f', borderRadius: '4px', border: '1px solid #6c5ce755' }} />
          <span>Gate</span>
        </div>
      </div>
    </div>
  );
};
