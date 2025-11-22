import { motion } from "framer-motion";

const TrackMap = ({ className, progress = 0, ghostProgress = 0 }) => {
  // EXACT GEOMETRY (Replica)
  const pathData = `
    M 700 800 L 300 800 C 150 800 120 750 120 600 L 120 550 Q 120 480 400 550 
    L 650 620 Q 700 630 700 580 L 700 520 Q 700 480 600 480 L 550 480 
    Q 450 480 450 400 L 380 320 C 300 250 250 200 280 150 C 300 100 400 100 520 150 
    L 550 170 L 850 380 Q 950 450 850 520 L 850 520 Q 800 580 820 620 
    C 840 680 860 750 830 780 Q 800 800 700 800 Z
  `;

  return (
    <div className={`${className} flex items-center justify-center select-none`}>
      <svg viewBox="0 0 1200 900" className="w-full h-full overflow-visible drop-shadow-2xl" preserveAspectRatio="xMidYMid meet">
        
        {/* Base Track (Clean, not blobby) */}
        <path d={pathData} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="28" strokeLinecap="round" strokeLinejoin="round" />
        <path d={pathData} fill="none" stroke="#1a1a1a" strokeWidth="20" strokeLinecap="round" strokeLinejoin="round" />

        {/* Racing Line (Ghost) */}
        <motion.path
          d={pathData} fill="none" stroke="#ef4444" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1, opacity: 0.5 }} transition={{ duration: 2, ease: "easeInOut" }}
          style={{ filter: 'drop-shadow(0 0 8px #ef4444)' }}
        />

        {/* Ghost Head (White Dot) */}
        <motion.path
          d={pathData} fill="none" stroke="#fff" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: ghostProgress }} transition={{ duration: 0.05, ease: "linear" }}
        />

        {/* Real Car (Blue Dot) */}
        <motion.path
          d={pathData} fill="none" stroke="#3b82f6" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: progress }} transition={{ duration: 0.05, ease: "linear" }}
          style={{ filter: 'drop-shadow(0 0 8px #3b82f6)' }}
        />

        {/* Markers */}
        <text x="730" y="550" fill="white" opacity="0.4" fontSize="20" fontFamily="sans-serif" fontWeight="bold">S1</text>
        <text x="550" y="120" fill="white" opacity="0.4" fontSize="20" fontFamily="sans-serif" fontWeight="bold">S2</text>
        <text x="650" y="850" fill="white" opacity="0.6" fontSize="16" fontFamily="monospace" fontWeight="bold" textAnchor="middle">FINISH</text>
        <line x1="650" y1="785" x2="650" y2="815" stroke="white" strokeWidth="2" strokeDasharray="4 4" />

      </svg>
    </div>
  );
};

export default TrackMap;