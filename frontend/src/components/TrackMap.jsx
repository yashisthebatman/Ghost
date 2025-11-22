import { memo, useEffect, useRef } from "react";
import { motion } from "framer-motion";

// ROAD AMERICA SHAPE (Approximate SVG Path)
// Normalized to 1000x1000 coordinate space
const TRACK_PATH = "M 470 900 L 250 900 C 150 900 120 850 120 750 L 130 600 L 130 500 Q 130 400 250 420 L 500 450 L 600 460 Q 680 470 680 400 L 670 300 Q 660 200 550 200 L 500 200 Q 400 200 400 150 L 400 100 C 400 50 500 50 600 80 L 800 150 L 900 300 L 850 500 L 850 700 Q 850 800 750 850 L 600 900 Z";

const TrackMap = ({ className, progress = 0, ghostProgress = 0 }) => {
  
  // We use a ref to access the DOM element of the path to calculate position
  const pathRef = useRef(null);
  const ghostDotRef = useRef(null);
  const realDotRef = useRef(null);

  // MANUAL ANIMATION LOOP for smooth 60fps movement
  // (Bypasses React State for position updates to fix lag)
  useEffect(() => {
    if (!pathRef.current) return;
    
    const length = pathRef.current.getTotalLength();
    
    // 1. Move Ghost Dot
    if (ghostDotRef.current) {
      const point = pathRef.current.getPointAtLength(length * ghostProgress);
      ghostDotRef.current.setAttribute("cx", point.x);
      ghostDotRef.current.setAttribute("cy", point.y);
    }

    // 2. Move Real Dot
    if (realDotRef.current) {
      const point = pathRef.current.getPointAtLength(length * progress);
      realDotRef.current.setAttribute("cx", point.x);
      realDotRef.current.setAttribute("cy", point.y);
    }
  }, [progress, ghostProgress]); // Updates whenever progress changes (passed from parent rAF loop)

  return (
    <div className={`${className} flex items-center justify-center select-none relative`}>
      <svg viewBox="0 0 1000 1000" className="w-full h-full drop-shadow-2xl" style={{ overflow: 'visible' }}>
        
        {/* Track Outline (Base) */}
        <path d={TRACK_PATH} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="40" strokeLinecap="round" strokeLinejoin="round" />
        
        {/* The Invisible Path used for Math */}
        <path ref={pathRef} d={TRACK_PATH} fill="none" stroke="none" />

        {/* Animated Trail (Ghost) */}
        <motion.path
          d={TRACK_PATH} fill="none" stroke="#ef4444" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: ghostProgress, opacity: 0.4 }} transition={{ duration: 0, ease: "linear" }}
        />

        {/* Animated Trail (Real) */}
        <motion.path
          d={TRACK_PATH} fill="none" stroke="#3b82f6" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: progress }} transition={{ duration: 0, ease: "linear" }}
          style={{ filter: 'drop-shadow(0 0 8px #3b82f6)' }}
        />

        {/* Dots (Controlled by useEffect ref) */}
        <circle ref={ghostDotRef} r="12" fill="#ef4444" stroke="white" strokeWidth="2" />
        <circle ref={realDotRef} r="12" fill="#3b82f6" stroke="white" strokeWidth="2" />

        {/* Metadata Labels */}
        <text x="500" y="950" fill="white" opacity="0.5" fontSize="24" textAnchor="middle" fontWeight="bold">START / FINISH</text>
        <text x="200" y="400" fill="white" opacity="0.3" fontSize="30" fontWeight="bold">T5</text>
        <text x="880" y="250" fill="white" opacity="0.3" fontSize="30" fontWeight="bold">KINK</text>

      </svg>
    </div>
  );
};

export default memo(TrackMap);