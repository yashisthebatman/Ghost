import { motion } from "framer-motion";
import { clsx } from "clsx";

// --- 1. CIRCULAR SPEEDOMETER (V2 RESTORED) ---
export const CircularSpeedo = ({ speed = 0 }) => {
  const safeSpeed = isNaN(speed) ? 0 : speed;
  const pct = Math.min(safeSpeed / 300, 1);
  const circumference = 2 * Math.PI * 45;

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" stroke="rgba(255,255,255,0.1)" strokeWidth="6" fill="none" />
        <motion.circle
          cx="50" cy="50" r="45"
          stroke="url(#grad)"
          strokeWidth="6"
          fill="none"
          strokeLinecap="round"
          style={{ strokeDasharray: circumference, strokeDashoffset: circumference - pct * circumference }}
          transition={{ duration: 0.05 }}
        />
        <defs>
          <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#0a84ff" />
            <stop offset="100%" stopColor="#ff3b30" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-5xl font-bold tracking-tighter text-white drop-shadow-2xl">{Math.round(safeSpeed)}</span>
        <span className="text-[10px] font-bold text-white/40 tracking-widest">KPH</span>
      </div>
    </div>
  );
};

// --- 2. INTERACTIVE LAP TABLE ---
export const LapTable = ({ laps = [], onSelectLap, selectedLapId }) => {
  return (
    <div className="w-full h-full overflow-auto">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-[#0a0a0a] text-white/30 font-mono uppercase tracking-wider border-b border-white/10">
          <tr><th className="p-3">Lap</th><th className="p-3">S1</th><th className="p-3">S2</th><th className="p-3 text-right">Time</th></tr>
        </thead>
        <tbody className="font-mono">
          {laps.map((lap, i) => (
            <tr 
              key={i} 
              onClick={() => onSelectLap(lap)}
              className={clsx(
                "border-b border-white/5 cursor-pointer transition-all hover:bg-white/10",
                selectedLapId === lap.lap_number ? "bg-white/10 text-white" : "text-white/50"
              )}
            >
              <td className="p-3 flex items-center gap-2">
                {selectedLapId === lap.lap_number && <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />}
                {lap.lap_number}
              </td>
              <td className="p-3">{lap.s1_time}</td>
              <td className="p-3">{lap.s2_time}</td>
              <td className="p-3 text-right font-bold">{lap.lap_time?.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// --- 3. G-FORCE (V2 RESTORED) ---
export const GForceMeter = ({ steer = 0, accel = 0 }) => {
  const x = (steer / 100) * 30; 
  const y = (accel * -1) * 30; 
  return (
    <div className="relative w-20 h-20 bg-white/5 rounded-full border border-white/10 flex items-center justify-center">
      <div className="absolute w-full h-px bg-white/10" />
      <div className="absolute h-full w-px bg-white/10" />
      <motion.div 
        className="w-3 h-3 bg-ghost rounded-full shadow-[0_0_10px_#ff3b30] z-10"
        animate={{ x, y }}
        transition={{ type: 'spring', stiffness: 200 }}
      />
    </div>
  );
};

// --- 4. INPUT BAR ---
export const InputBar = ({ label, value, color }) => (
  <div className="w-full">
    <div className="flex justify-between text-[9px] font-bold text-white/40 mb-1">
      <span>{label}</span>
    </div>
    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
      <motion.div className={`h-full ${color}`} animate={{ width: `${value}%` }} transition={{ duration: 0.05 }} />
    </div>
  </div>
);