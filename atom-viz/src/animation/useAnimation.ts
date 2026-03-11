// Animation playback hook

import { useState, useCallback, useRef, useEffect } from 'react';
import { AnimationTimeline, AnimationState, PlaybackSettings, DEFAULT_PLAYBACK_SETTINGS } from './types';

interface UseAnimationReturn {
  state: AnimationState;
  play: () => void;
  pause: () => void;
  stop: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  seekToStep: (stepIndex: number) => void;
  seekToProgress: (progress: number) => void;
  setSpeed: (speed: number) => void;
  settings: PlaybackSettings;
}

export function useAnimation(timeline: AnimationTimeline | null): UseAnimationReturn {
  const [state, setState] = useState<AnimationState>({
    status: 'idle',
    currentStepIndex: 0,
    stepProgress: 0,
    globalProgress: 0,
  });

  const [settings, setSettings] = useState<PlaybackSettings>(DEFAULT_PLAYBACK_SETTINGS);
  
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  // Calculate global progress from step index and step progress
  const calculateGlobalProgress = useCallback((stepIndex: number, stepProgress: number): number => {
    if (!timeline || timeline.steps.length === 0) return 0;
    const stepsCompleted = stepIndex + stepProgress;
    return stepsCompleted / timeline.steps.length;
  }, [timeline]);

  // Animation loop
  const animate = useCallback((currentTime: number) => {
    if (!timeline || timeline.steps.length === 0) return;

    const deltaTime = lastTimeRef.current ? currentTime - lastTimeRef.current : 0;
    lastTimeRef.current = currentTime;

    setState(prev => {
      if (prev.status !== 'playing') return prev;

      const currentStep = timeline.steps[prev.currentStepIndex];
      if (!currentStep) {
        return { ...prev, status: 'complete' };
      }

      // Calculate progress increment based on speed and step duration
      const effectiveDuration = currentStep.duration / settings.speed;
      const progressIncrement = deltaTime / effectiveDuration;
      let newStepProgress = prev.stepProgress + progressIncrement;
      let newStepIndex = prev.currentStepIndex;

      // Check if we've completed the current step
      while (newStepProgress >= 1 && newStepIndex < timeline.steps.length - 1) {
        newStepProgress -= 1;
        newStepIndex++;
      }

      // Check if animation is complete
      if (newStepIndex >= timeline.steps.length - 1 && newStepProgress >= 1) {
        return {
          status: 'complete',
          currentStepIndex: timeline.steps.length - 1,
          stepProgress: 1,
          globalProgress: 1,
        };
      }

      return {
        status: 'playing',
        currentStepIndex: newStepIndex,
        stepProgress: Math.min(newStepProgress, 1),
        globalProgress: calculateGlobalProgress(newStepIndex, Math.min(newStepProgress, 1)),
      };
    });

    animationRef.current = requestAnimationFrame(animate);
  }, [timeline, settings.speed, calculateGlobalProgress]);

  // Start/stop animation loop based on status
  useEffect(() => {
    if (state.status === 'playing') {
      lastTimeRef.current = 0;
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [state.status, animate]);

  const play = useCallback(() => {
    if (!timeline || timeline.steps.length === 0) return;
    
    setState(prev => {
      // If complete, restart from beginning
      if (prev.status === 'complete') {
        return {
          status: 'playing',
          currentStepIndex: 0,
          stepProgress: 0,
          globalProgress: 0,
        };
      }
      return { ...prev, status: 'playing' };
    });
  }, [timeline]);

  const pause = useCallback(() => {
    setState(prev => ({ ...prev, status: 'paused' }));
  }, []);

  const stop = useCallback(() => {
    setState({
      status: 'idle',
      currentStepIndex: 0,
      stepProgress: 0,
      globalProgress: 0,
    });
  }, []);

  const stepForward = useCallback(() => {
    if (!timeline) return;
    
    setState(prev => {
      const newIndex = Math.min(prev.currentStepIndex + 1, timeline.steps.length - 1);
      const isComplete = newIndex === timeline.steps.length - 1;
      return {
        status: isComplete ? 'complete' : 'paused',
        currentStepIndex: newIndex,
        stepProgress: isComplete ? 1 : 0,
        globalProgress: calculateGlobalProgress(newIndex, isComplete ? 1 : 0),
      };
    });
  }, [timeline, calculateGlobalProgress]);

  const stepBackward = useCallback(() => {
    if (!timeline) return;
    
    setState(prev => {
      const newIndex = Math.max(prev.currentStepIndex - 1, 0);
      return {
        status: 'paused',
        currentStepIndex: newIndex,
        stepProgress: 0,
        globalProgress: calculateGlobalProgress(newIndex, 0),
      };
    });
  }, [timeline, calculateGlobalProgress]);

  const seekToStep = useCallback((stepIndex: number) => {
    if (!timeline) return;
    
    const clampedIndex = Math.max(0, Math.min(stepIndex, timeline.steps.length - 1));
    setState({
      status: 'paused',
      currentStepIndex: clampedIndex,
      stepProgress: 0,
      globalProgress: calculateGlobalProgress(clampedIndex, 0),
    });
  }, [timeline, calculateGlobalProgress]);

  const seekToProgress = useCallback((progress: number) => {
    if (!timeline || timeline.steps.length === 0) return;
    
    const clampedProgress = Math.max(0, Math.min(progress, 1));
    const totalSteps = timeline.steps.length;
    const exactStep = clampedProgress * totalSteps;
    const stepIndex = Math.min(Math.floor(exactStep), totalSteps - 1);
    const stepProgress = exactStep - stepIndex;
    
    setState({
      status: clampedProgress >= 1 ? 'complete' : 'paused',
      currentStepIndex: stepIndex,
      stepProgress: stepProgress,
      globalProgress: clampedProgress,
    });
  }, [timeline]);

  const setSpeed = useCallback((speed: number) => {
    setSettings(prev => ({ ...prev, speed }));
  }, []);

  return {
    state,
    play,
    pause,
    stop,
    stepForward,
    stepBackward,
    seekToStep,
    seekToProgress,
    setSpeed,
    settings,
  };
}
