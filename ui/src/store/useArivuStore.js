import { create } from 'zustand';

export const useArivuStore = create((set) => ({
  traceMode: 'IDLE', // 'IDLE' | 'LIVE' | 'REPLAY'
  activeStage: 'FEED',
  stageData: {},
  statusLine: { state: 'IDLE', action: 'HOLD', reason: 'WAITING' },

  setTraceMode: (mode) => set({ traceMode: mode }),
  setActiveStage: (stage) => set({ activeStage: stage }),
  setStageData: (stage, data) => set((state) => ({
    stageData: { ...state.stageData, [stage]: data }
  })),
  setStatusLine: (status) => set({ statusLine: status }),
}));
