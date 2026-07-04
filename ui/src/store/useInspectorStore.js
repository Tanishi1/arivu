import { create } from 'zustand';

export const useInspectorStore = create((set) => ({
  pinnedObject: null,
  autoFollow: true,

  setPinnedObject: (obj) => set({ pinnedObject: obj, autoFollow: false }),
  clearPinned: () => set({ pinnedObject: null, autoFollow: true }),
  setAutoFollow: (val) => set({ autoFollow: val }),
}));
