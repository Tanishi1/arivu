import { useEffect, useRef } from 'react';
import { useArivuStore } from '../store/useArivuStore';
import { useCrossCuttingStore } from '../store/useCrossCuttingStore';
import replayFixture from '../mocks/fixtures/replay_experiment_01.json';

const STAGES = [
  'FEED', 'BARS', 'GRAPH', 'THESIS', 'READINESS', 
  'TWIN', 'LEDGER', 'TRUST', 'OPTIMIZER'
];

export function useTraceController() {
  const { traceMode, activeStage, setActiveStage, setStageData, setStatusLine, setTraceMode } = useArivuStore();
  const { setEscapeValve, setRegime } = useCrossCuttingStore();
  const timerRef = useRef(null);

  useEffect(() => {
    if (traceMode === 'REPLAY') {
      let stageIndex = 0;
      
      // Load cross-cutting data upfront for the replay
      setEscapeValve(replayFixture.escape_valve);
      setRegime(replayFixture.regime);

      const playNextStage = () => {
        if (stageIndex >= STAGES.length) {
          setStatusLine({ state: 'DONE', action: 'IDLE', reason: 'REPLAY_COMPLETE' });
          setTraceMode('IDLE');
          return;
        }

        const stage = STAGES[stageIndex];
        setActiveStage(stage);
        
        // Mock data loading for the stage
        const stageDataKey = stage.toLowerCase();
        if (replayFixture[stageDataKey]) {
          setStageData(stage, replayFixture[stageDataKey]);
        }
        
        setStatusLine({ state: `LOADING_${stage}`, action: 'PLAYING', reason: 'REPLAY_MODE' });

        stageIndex++;
        timerRef.current = setTimeout(playNextStage, 2000); // 2 second per stage
      };

      playNextStage();

      return () => {
        if (timerRef.current) clearTimeout(timerRef.current);
      };
    } else if (traceMode === 'LIVE') {
      setStatusLine({ state: 'CHECKING_SELF', action: 'HOLD', reason: 'WAITING_FOR_WS' });
      // WS wiring would go here
    }
  }, [traceMode, setActiveStage, setStageData, setStatusLine, setEscapeValve, setRegime, setTraceMode]);
}
