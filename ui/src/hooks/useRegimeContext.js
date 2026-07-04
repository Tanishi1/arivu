import { useCrossCuttingStore } from '../store/useCrossCuttingStore';

export function useRegimeContext() {
  const { regime } = useCrossCuttingStore();
  return regime;
}
