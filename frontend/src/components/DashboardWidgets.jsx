import { motion } from "framer-motion";

// --- PROFESSIONAL ANALYSIS TOOL ---
export const TelemetryComparison = ({ optimalFrame = {}, humanFrame = {} }) => {
  // Safe Defaults
  const optSpeed = optimalFrame.speed || 0;
  const humSpeed = humanFrame.speed || 0;
  const speedDelta = humSpeed - optSpeed;

  const optThr = optimalFrame.ath || 0;
  const humThr = humanFrame.ath || 0;

  const optBrk = optimalFrame.pbrake_f || 0;
  const humBrk = humanFrame.pbrake_f || 0;

  // --- INSIGHT ENGINE (Plain English Logic) ---
  let insight = "Matching Pace";
  let insightColor = "text-white/40";

  // Logic priorities: Braking > Throttle > Speed
  if (Math.abs(humBrk - optBrk) > 15) {
      if (humBrk > optBrk) {
          insight = "BRAKING TOO HARD";
          insightColor = "text-red-500";
      } else {
          insight = "UNDER-BRAKING";
          insightColor = "text-yellow-500";
      }
  } else if (Math.abs(humThr - optThr) > 20) {
      if (humThr < optThr) {
          insight = "HESITANT THROTTLE";
          insightColor = "text-orange-500";
      } else {
          insight = "AGGRESSIVE THROTTLE";
          insightColor = "text-blue-400";
      }
  } else if (Math.abs(speedDelta) > 5) {
      if (speedDelta > 0) {
          insight = "CARRYING MORE SPEED";
          insightColor = "text-green-400";
      } else {
          insight = "LOSING CORNER SPEED";
          insightColor = "text-red-400";
      }
  }

  return (
    <div className="flex flex-col h-full gap-3">
        
        {/* 1. THE HEADLINE METRIC */}
        <div className="flex items-center justify-between bg-white/5 p-3 rounded-xl border border-white/10">
            <div>
                <div className="text-[9px] text-white/40 font-bold uppercase tracking-widest">Speed Delta</div>
                <div className={`text-3xl font-mono font-bold ${speedDelta >= 0 ? 'text-green-400' : 'text-red-500'}`}>
                    {speedDelta > 0 ? '+' : ''}{speedDelta.toFixed(0)} <span className="text-sm text-white/40">KPH</span>
                </div>
            </div>
            <div className="text-right">
                 <div className="text-[9px] text-white/40 font-bold uppercase tracking-widest">Live Insight</div>
                 <div className={`text-sm font-bold font-mono ${insightColor} animate-pulse`}>{insight}</div>
            </div>
        </div>

        {/* 2. COMPARISON BARS (Clearer Labels) */}
        <div className="flex-1 flex flex-col justify-center gap-4 p-1">
            <Comparator label="THROTTLE" optimal={optThr} human={humThr} color="bg-green-500" />
            <Comparator label="BRAKE" optimal={optBrk} human={humBrk} color="bg-red-500" />
            <Comparator label="STEERING" optimal={Math.abs(optimalFrame.Steering_Angle || 0)/5} human={Math.abs(humanFrame.Steering_Angle || 0)/5} color="bg-blue-500" />
        </div>

        {/* 3. LEGEND */}
        <div className="flex justify-center gap-6 text-[9px] font-bold uppercase tracking-widest text-white/30 border-t border-white/10 pt-2">
            <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full border border-white/50"/> Optimal Target</span>
            <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-white"/> Your Input</span>
        </div>
    </div>
  );
};

// --- HELPER COMPONENT ---
const Comparator = ({ label, optimal, human, color }) => (
    <div>
        <div className="flex justify-between mb-1">
            <span className="text-[9px] font-bold text-white/50">{label}</span>
            <span className="text-[9px] font-mono text-white/80">
                {human.toFixed(0)}% <span className="text-white/30">/ {optimal.toFixed(0)}%</span>
            </span>
        </div>
        <div className="h-4 bg-white/5 rounded relative overflow-hidden border border-white/5">
            {/* Optimal Marker (The Ghost Target) */}
            <div 
                className="absolute top-0 bottom-0 w-1 bg-white/30 z-10"
                style={{ left: `${Math.min(optimal, 98)}%` }}
            />
            {/* Human Bar (The Fill) */}
            <motion.div 
                className={`h-full ${color} opacity-90`}
                animate={{ width: `${Math.min(human, 100)}%` }}
                transition={{ duration: 0.05 }}
            />
        </div>
    </div>
);

// Export simple bar for bottom controls
export const Bar = ({ label, val, col }) => (
    <div className="w-full">
      <div className="flex justify-between text-[9px] font-bold text-white/40 mb-1">
        <span>{label}</span>
      </div>
      <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
        <motion.div className={`h-full ${col}`} animate={{ width: `${Math.min(val || 0, 100)}%` }} transition={{ duration: 0.1 }} />
      </div>
    </div>
);