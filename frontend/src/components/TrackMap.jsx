import { motion } from "framer-motion";

const TrackMap = ({ className, progress = 0, ghostProgress = 0 }) => {
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
        
        <path d={pathData} fill="none" stroke="rgba(0,0,0,0.3)" strokeWidth="40" strokeLinecap="round" strokeLinejoin="round" />
        <path d={pathData} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="24" strokeLinecap="round" strokeLinejoin="round" />

        {/* Ghost Trail */}
        <motion.path
          d={pathData} fill="none" stroke="#ef4444" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1, opacity: 0.3 }} transition={{ duration: 2, ease: "easeInOut" }}
          style={{ filter: 'drop-shadow(0 0 10px rgba(239,68,68,0.8))' }}
        />

        <motion.path
          d={pathData} fill="none" stroke="#fff" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: ghostProgress }} transition={{ duration: 0.05, ease: "linear" }}
          style={{ filter: 'drop-shadow(0 0 8px white)' }}
        />

        <motion.path
          d={pathData} fill="none" stroke="#3b82f6" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: progress }} transition={{ duration: 0.05, ease: "linear" }}
          style={{ filter: 'drop-shadow(0 0 12px #3b82f6)' }}
        />

        <text x="730" y="550" fill="white" opacity="0.4" fontSize="24" fontFamily="sans-serif" fontWeight="bold">S1</text>
        <text x="550" y="120" fill="white" opacity="0.4" fontSize="24" fontFamily="sans-serif" fontWeight="bold">S2</text>
        <text x="900" y="580" fill="white" opacity="0.4" fontSize="24" fontFamily="sans-serif" fontWeight="bold">S3</text>
        <line x1="650" y1="780" x2="650" y2="820" stroke="white" strokeWidth="3" strokeDasharray="6 6" opacity="0.7" />
        <text x="650" y="860" fill="white" opacity="0.6" fontSize="20" fontFamily="sans-serif" fontWeight="bold" textAnchor="middle" letterSpacing="2px">FINISH</text>

      </svg>
    </div>
  );
};

export default TrackMap;