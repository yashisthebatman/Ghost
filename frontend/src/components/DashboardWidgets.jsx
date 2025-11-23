import { motion } from "framer-motion";

// --- 1. HEADS UP DISPLAY (New KPH Widget) ---
export const TelemetryHUD = ({ speed, throttle, brake, gear }) => (
    <div className="flex items-end gap-6">
        {/* Large Speed */}
        <div>
            <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">Speed</div>
            <div className="flex items-baseline gap-2">
                <span className="text-6xl font-mono font-bold text-white tracking-tighter">
                    {Math.round(speed || 0)}
                </span>
                <span className="text-sm text-white/40 font-bold">KPH</span>
            </div>
        </div>
        
        {/* Inputs Summary */}
        <div className="flex gap-3 pb-2">
            <div className="flex flex-col items-center gap-1">
                <div className="w-1.5 h-12 bg-white/10 rounded-full overflow-hidden flex items-end">
                    <motion.div className="w-full bg-green-500" animate={{ height: `${throttle}%` }}/>
                </div>
                <span className="text-[9px] text-white/40 font-bold">THR</span>
            </div>
            <div className="flex flex-col items-center gap-1">
                <div className="w-1.5 h-12 bg-white/10 rounded-full overflow-hidden flex items-end">
                    <motion.div className="w-full bg-red-500" animate={{ height: `${brake}%` }}/>
                </div>
                <span className="text-[9px] text-white/40 font-bold">BRK</span>
            </div>
        </div>
    </div>
);

// --- 2. COMPARISON TOOL (Updated for Solo Mode) ---
export const TelemetryComparison = ({ isCompareMode, optimalFrame = {}, humanFrame = {} }) => {
  
  if (!isCompareMode) {
      return (
          <div className="h-full flex flex-col items-center justify-center text-white/30 space-y-3">
              <div className="w-12 h-12 rounded-full border-2 border-dashed border-white/20 flex items-center justify-center">
                  <span className="text-xl font-bold">?</span>
              </div>
              <div className="text-xs text-center">
                  <p className="font-bold text-white/50">SOLO MODE</p>
                  <p className="mt-1 text-[10px]">Select "Compare" to see<br/>performance deltas.</p>
              </div>
          </div>
      );
  }

  // ... (Keep existing logic for deltas from previous response) ...
  const speedDelta = (humanFrame.speed || 0) - (optimalFrame.speed || 0);
  const optThr = optimalFrame.ath || 0; 
  const humThr = humanFrame.ath || 0;
  const optBrk = optimalFrame.pbrake_f || 0; 
  const humBrk = humanFrame.pbrake_f || 0;

  // Logic Engine
  let insight = "Matching Pace";
  let insightColor = "text-white/40";
  
  if (Math.abs(humBrk - optBrk) > 15) {
      if (humBrk > optBrk) { insight = "BRAKING TOO HARD"; insightColor = "text-red-500"; }
      else { insight = "UNDER-BRAKING"; insightColor = "text-yellow-500"; }
  } else if (Math.abs(speedDelta) > 5) {
      if (speedDelta > 0) { insight = "CARRYING SPEED"; insightColor = "text-green-400"; }
      else { insight = "LOSING SPEED"; insightColor = "text-red-400"; }
  }

  return (
    <div className="flex flex-col h-full gap-3">
        {/* Headline */}
        <div className="flex items-center justify-between bg-white/5 p-3 rounded-xl border border-white/10">
            <div>
                <div className="text-[9px] text-white/40 font-bold uppercase tracking-widest">Speed Delta</div>
                <div className={`text-3xl font-mono font-bold ${speedDelta >= 0 ? 'text-green-400' : 'text-red-500'}`}>
                    {speedDelta > 0 ? '+' : ''}{speedDelta.toFixed(0)} <span className="text-sm text-white/40">KPH</span>
                </div>
            </div>
            <div className="text-right">
                 <div className="text-[9px] text-white/40 font-bold uppercase tracking-widest">Analysis</div>
                 <div className={`text-sm font-bold font-mono ${insightColor}`}>{insight}</div>
            </div>
        </div>

        {/* Bars */}
        <div className="flex-1 flex flex-col justify-center gap-4 p-1">
            <Comparator label="THROTTLE" optimal={optThr} human={humThr} color="bg-green-500" />
            <Comparator label="BRAKE" optimal={optBrk} human={humBrk} color="bg-red-500" />
        </div>
    </div>
  );
};

const Comparator = ({ label, optimal, human, color }) => (
    <div>
        <div className="flex justify-between mb-1">
            <span className="text-[9px] font-bold text-white/50">{label}</span>
        </div>
        <div className="h-3 bg-white/5 rounded relative overflow-hidden border border-white/5">
            <div className="absolute top-0 bottom-0 w-0.5 bg-white/50 z-10" style={{ left: `${optimal}%` }} />
            <motion.div className={`h-full ${color} opacity-90`} animate={{ width: `${human}%` }} />
        </div>
    </div>
);

export const Bar = ({ label, val, col }) => (
    <div className="w-full">
      <div className="flex justify-between text-[9px] font-bold text-white/40 mb-1"><span>{label}</span></div>
      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
        <motion.div className={`h-full ${col}`} animate={{ width: `${val}%` }} />
      </div>
    </div>
);