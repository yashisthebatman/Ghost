import { motion } from "framer-motion";

const TrackMap = ({ className, progress = 0 }) => {
  // Road America Geometry (Accurate Topology)
  // ViewBox 0 0 200 150
  const pathData = `
    M 50 135 
    L 150 135  
    Q 160 135 160 125 
    L 160 100 
    Q 160 90 150 90 
    L 130 92 
    Q 115 94 115 80 
    L 115 70 
    Q 115 60 105 60 
    L 95 60 
    Q 85 60 85 70 
    L 85 75 
    Q 85 90 65 90 
    Q 45 90 45 70 
    Q 45 50 65 40 
    L 90 35 
    L 120 30 
    Q 130 28 130 38 
    L 130 50 
    L 150 100 
    L 150 115 
    Q 150 125 140 125 
    L 60 125 
    Q 50 125 50 135
  `;

  return (
    <div className={`relative ${className} flex items-center justify-center`}>
      <svg viewBox="0 0 200 160" className="w-full h-full drop-shadow-[0_0_20px_rgba(255,255,255,0.1)]">
        
        {/* 1. Track Base (Dark) */}
        <path d={pathData} fill="none" stroke="rgba(255, 255, 255, 0.1)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
        
        {/* 2. Ghost Line (Active Lap) */}
        <motion.path
          d={pathData}
          fill="none"
          stroke="#ff3b30"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0, opacity: 0.5 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 2, ease: "easeInOut" }}
        />

        {/* 3. Car Position Indicator (Glowing Dot) */}
        <path d={pathData} id="trackPath" fill="none" stroke="none" />
        <motion.circle r="4" fill="#fff" className="drop-shadow-[0_0_10px_rgba(255,255,255,1)] z-50">
          {/* We use pathLength on a stroke usually, but for a dot we need an offset.
              For simplicity in this architecture, we simulate position by drawing a white stroke
              on top that ends at the 'progress' point.
          */}
        </motion.circle>

        {/* Progress Overlay (The actual moving line showing distance covered) */}
        <motion.path
          d={pathData}
          fill="none"
          stroke="#fff"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: progress }}
          transition={{ duration: 0.1, ease: "linear" }}
        />

        {/* Sector Labels */}
        <Marker x={155} y={135} label="T1" />
        <Marker x={115} y={80} label="T5" />
        <Marker x={50} y={60} label="CAROUSEL" />
        <Marker x={120} y={30} label="KINK" />
        <Marker x={150} y={100} label="CANADA" />
        <Marker x={50} y={135} label="FINISH" color="#0a84ff" />
      </svg>
    </div>
  );
};

const Marker = ({ x, y, label, color = "rgba(255,255,255,0.4)" }) => (
  <g>
    <text x={x + 4} y={y + 2} fill={color} fontSize="5" fontFamily="monospace" fontWeight="bold">{label}</text>
    <circle cx={x} cy={y} r="1.5" fill={color} />
  </g>
);

export default TrackMap;