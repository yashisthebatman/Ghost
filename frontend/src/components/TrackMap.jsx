import { memo, useEffect, useRef } from "react";
import { motion } from "framer-motion";

const TRACK_PATH = "M 470 900 L 250 900 C 150 900 120 850 120 750 L 130 600 L 130 500 Q 130 400 250 420 L 500 450 L 600 460 Q 680 470 680 400 L 670 300 Q 660 200 550 200 L 500 200 Q 400 200 400 150 L 400 100 C 400 50 500 50 600 80 L 800 150 L 900 300 L 850 500 L 850 700 Q 850 800 750 850 L 600 900 Z";

const TrackMap = ({ className, progress = 0, ghostProgress = null }) => {
  const pathRef = useRef(null);
  const ghostDotRef = useRef(null);
  const realDotRef = useRef(null);

  useEffect(() => {
    if (!pathRef.current) return;
    const length = pathRef.current.getTotalLength();
    
    // Real Dot
    if (realDotRef.current) {
      const point = pathRef.current.getPointAtLength(length * progress);
      realDotRef.current.setAttribute("cx", point.x);
      realDotRef.current.setAttribute("cy", point.y);
    }

    // Ghost Dot (Only if enabled)
    if (ghostDotRef.current && ghostProgress !== null) {
      const point = pathRef.current.getPointAtLength(length * ghostProgress);
      ghostDotRef.current.setAttribute("cx", point.x);
      ghostDotRef.current.setAttribute("cy", point.y);
      ghostDotRef.current.style.opacity = "1";
    } else if (ghostDotRef.current) {
      ghostDotRef.current.style.opacity = "0";
    }
  }, [progress, ghostProgress]);

  return (
    <div className={`${className} flex items-center justify-center select-none relative`}>
      <svg viewBox="0 0 1000 1000" className="w-full h-full drop-shadow-2xl" style={{ overflow: 'visible' }}>
        <path d={TRACK_PATH} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="40" strokeLinecap="round" strokeLinejoin="round" />
        <path ref={pathRef} d={TRACK_PATH} fill="none" stroke="none" />
        
        {/* Ghost Trail & Dot */}
        {ghostProgress !== null && (
            <>
                <motion.path d={TRACK_PATH} fill="none" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"
                    initial={{ pathLength: 0 }} animate={{ pathLength: ghostProgress, opacity: 0.3 }} transition={{ duration: 0, ease: "linear" }}/>
                <circle ref={ghostDotRef} r="10" fill="#ffffff" stroke="none" className="transition-opacity" />
            </>
        )}

        {/* Real Trail & Dot */}
        <motion.path d={TRACK_PATH} fill="none" stroke="#10b981" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: progress }} transition={{ duration: 0, ease: "linear" }}
          style={{ filter: 'drop-shadow(0 0 8px #10b981)' }} />
        <circle ref={realDotRef} r="12" fill="#10b981" stroke="white" strokeWidth="2" />
        
        <text x="500" y="950" fill="white" opacity="0.5" fontSize="24" textAnchor="middle" fontWeight="bold">START / FINISH</text>
      </svg>
    </div>
  );
};

export default memo(TrackMap);