import { create } from 'zustand';

export const useCrossCuttingStore = create((set) => ({
  escapeValve: {
    armed: false,
    fired: false,
    holdsCount: 0,
    holdsThreshold: 12,
    positionFraction: 0,
    feedbackEnabled: false
  },
  regime: {
    status: 'bootstrap', // 'bootstrap' | 'retraining' | 'trained'
    k: 0,
    clusterId: null,
    confidence: null,
    populationSize: 0
  },

  setEscapeValve: (data) => set({ escapeValve: data }),
  setRegime: (data) => set({ regime: data }),
}));
