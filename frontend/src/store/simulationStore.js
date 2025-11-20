import { create } from 'zustand';

export const useSimulationStore = create((set, get) => ({
  isPlaying: false,
  currentIndex: 0,
  maxIndex: 0,
  
  actions: {
    setDataLength: (len) => set({ maxIndex: len - 1 }),
    togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
    setIndex: (idx) => set({ currentIndex: idx }),
    nextFrame: () => set((state) => {
      // Loop functionality
      if (state.currentIndex >= state.maxIndex) return { isPlaying: false, currentIndex: 0 };
      return { currentIndex: state.currentIndex + 1 };
    }),
    reset: () => set({ currentIndex: 0, isPlaying: false }),
  }
}));