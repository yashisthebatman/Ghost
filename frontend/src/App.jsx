import { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import { Play, Pause, Activity, Zap, Trophy, Timer, ToggleLeft, ToggleRight, Target } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import TrackMap from './components/TrackMap';
import { TelemetryComparison, TelemetryHUD, Bar } from './components/DashboardWidgets';
import { useSimulationStore } from './store/simulationStore';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
  // State
  const [status, setStatus] = useState("BOOT");
  const [session, setSession] = useState(null);
  const [laps, setLaps] = useState([]);
  
  // Selection State
  const [selectedLapId, setSelectedLapId] = useState(null); // ID or "optimal"
  const [isCompareMode, setIsCompareMode] = useState(false); // Toggle state

  // Data
  const [displayData, setDisplayData] = useState([]);
  const [optimalDataCache, setOptimalDataCache] = useState([]);

  // Simulation Refs
  const { isPlaying, currentIndex, actions } = useSimulationStore();
  const requestRef = useRef();
  const lastTimeRef = useRef(0);

  // --- 1. BOOT ---
  useEffect(() => {
    const boot = async () => {
      try {
        const [ctx, lapsRes, optRes] = await Promise.all([
            axios.get(`${API_URL}/session/context`),
            axios.get(`${API_URL}/session/laps`),
            axios.get(`${API_URL}/laps/optimal`),
        ]);

        setStatus("READY");
        setSession(ctx.data);
        setLaps(lapsRes.data);
        setOptimalDataCache(optRes.data);

        // Default: Load first human lap in SOLO mode
        if (lapsRes.data.length > 0) {
            loadSoloLap(lapsRes.data[0].id, lapsRes.data[0].id);
        }
      } catch (e) {
        console.error(e);
        setStatus("ERROR");
      }
    };
    boot();
  }, []);

  // --- 2. DATA LOADERS ---
  
  // A. SOLO MODE: Just load one file
  const loadSoloLap = async (id) => {
      setIsCompareMode(false);
      setSelectedLapId(id);
      actions.reset();

      let data = [];
      if (id === "optimal") {
          data = optimalDataCache;
      } else {
          try {
              const res = await axios.get(`${API_URL}/laps/human/${id}`);
              data = res.data;
          } catch { data = []; }
      }

      // Standardize structure
      const processed = data.map(p => ({
          time: p.time,
          // Primary is the one selected
          primary_speed: p.speed,
          primary_obj: p,
          // Secondary is empty in solo
          ref_speed: 0,
          ref_obj: {}
      }));

      setDisplayData(processed);
      actions.setDataLength(processed.length);
  };

  // B. COMPARE MODE: Load Selected vs Optimal
  const loadComparison = async (humanId) => {
      if (humanId === "optimal") return; // Can't compare optimal vs optimal
      
      setIsCompareMode(true);
      setSelectedLapId(humanId);
      actions.reset();

      let humanData = [];
      try {
          const res = await axios.get(`${API_URL}/laps/human/${humanId}`);
          humanData = res.data;
      } catch { humanData = []; }

      const maxLength = Math.max(optimalDataCache.length, humanData.length);
      const merged = Array.from({ length: maxLength }).map((_, i) => {
          const opt = optimalDataCache[i] || {};
          const hum = humanData[i] || {};
          return {
              time: i * 0.01,
              primary_speed: hum.speed || 0,
              primary_obj: hum,
              ref_speed: opt.speed || 0,
              ref_obj: opt
          };
      });

      setDisplayData(merged);
      actions.setDataLength(merged.length);
  };

  // Toggle Handler
  const toggleCompare = () => {
      if (selectedLapId === "optimal") return; // Disable toggle if viewing optimal
      if (!isCompareMode) {
          loadComparison(selectedLapId);
      } else {
          loadSoloLap(selectedLapId);
      }
  };

  // --- 3. ANIMATION LOOP ---
  const animate = (time) => {
    if (!isPlaying) return;
    if (lastTimeRef.current === 0) lastTimeRef.current = time;
    const deltaTime = time - lastTimeRef.current;
    const framesToAdvance = Math.floor(deltaTime / 10);

    if (framesToAdvance > 0) {
        for(let i=0; i<framesToAdvance; i++) actions.nextFrame();
        lastTimeRef.current = time - (deltaTime % 10);
    }
    requestRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    if (isPlaying) {
        lastTimeRef.current = 0;
        requestRef.current = requestAnimationFrame(animate);
    } else {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
    return () => cancelAnimationFrame(requestRef.current);
  }, [isPlaying]);

  // --- VIEW STATE ---
  const frame = displayData[currentIndex] || {};
  const progress = displayData.length > 0 ? currentIndex / displayData.length : 0;
  
  // Memoize Chart
  const chartWindow = useMemo(() => {
      const start = Math.max(0, currentIndex - 300);
      const end = Math.min(displayData.length, start + 600);
      return displayData.slice(start, end);
  }, [currentIndex, displayData]);

  if (status === "BOOT") return <div className="h-screen w-screen bg-black flex items-center justify-center text-white/30 font-mono animate-pulse">LOADING TELEMETRY...</div>;

  return (
    <div className="dashboard-container bg-black text-white font-sans">
      <div className="noise-bg" />

      {/* HEADER */}
      <header style={{ gridArea: 'header' }} className="liquid-glass rounded-2xl flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-4">
            <div className="w-9 h-9 bg-emerald-600 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-900/50">
                <Activity size={18} className="text-white"/>
            </div>
            <div>
                <h1 className="text-sm font-bold tracking-widest text-white/90">GHOST ENGINEER</h1>
                <div className="flex items-center gap-2 text-[10px] text-white/50 font-mono">
                    <span>{session?.vehicle_id}</span>
                    <span className="text-emerald-400">ONLINE</span>
                </div>
            </div>
        </div>
        
        {/* COMPARISON TOGGLE */}
        <div 
            onClick={toggleCompare}
            className={`cursor-pointer px-4 py-2 rounded-full border flex items-center gap-3 transition-all select-none
            ${isCompareMode ? 'bg-emerald-500/20 border-emerald-500/50' : 'bg-white/5 border-white/10 hover:bg-white/10'}`}
        >
            <span className={`text-xs font-bold ${isCompareMode ? 'text-emerald-400' : 'text-white/40'}`}>
                COMPARE MODE
            </span>
            {isCompareMode ? <ToggleRight size={20} className="text-emerald-400"/> : <ToggleLeft size={20} className="text-white/40"/>}
        </div>
      </header>

      {/* SIDEBAR */}
      <div style={{ gridArea: 'sidebar' }} className="liquid-glass rounded-3xl flex flex-col overflow-hidden z-40">
         <div className="p-5 border-b border-white/10 bg-white/5">
            <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">Lap Selector</div>
         </div>
         <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
            
            {/* OPTIMAL LAP OPTION */}
            <div 
                onClick={() => loadSoloLap("optimal")}
                className={`p-3 rounded-2xl cursor-pointer border transition-all flex justify-between items-center
                ${selectedLapId === "optimal" ? 'bg-purple-500/20 border-purple-500/50 text-white' : 'border-transparent hover:bg-white/5 text-purple-400/60'}`}
            >
                <div className="flex items-center gap-3">
                    <Target size={16}/>
                    <span className="text-xs font-bold uppercase tracking-wider">Optimal Lap</span>
                </div>
            </div>

            <div className="h-px bg-white/10 my-2 mx-2"/>

            {/* HUMAN LAPS */}
            {laps.map(lap => (
                <div key={lap.id} 
                     onClick={() => isCompareMode ? loadComparison(lap.id) : loadSoloLap(lap.id)}
                     className={`p-3 rounded-2xl cursor-pointer border transition-all duration-200
                        ${selectedLapId === lap.id ? 'liquid-glass-active border-emerald-500/30' : 'border-transparent hover:bg-white/5'}`}>
                    <div className="flex justify-between items-center mb-1">
                        <span className={`text-[10px] font-bold uppercase tracking-wider ${selectedLapId === lap.id ? 'text-emerald-400' : 'text-white/60'}`}>
                            {lap.name}
                        </span>
                        {lap.status === "PB" && <Trophy size={12} className="text-yellow-400"/>}
                    </div>
                    <div className="flex justify-between items-end">
                        <span className={`text-xl font-mono font-medium ${selectedLapId === lap.id ? 'text-white' : 'text-white/50'}`}>
                            {lap.lap_time.toFixed(3)}s
                        </span>
                    </div>
                </div>
            ))}
         </div>
      </div>

      {/* MAP */}
      <div style={{ gridArea: 'map' }} className="liquid-glass rounded-3xl relative flex items-center justify-center overflow-hidden">
          <div className="w-full h-full p-6">
              <TrackMap 
                  className="w-full h-full" 
                  progress={progress} 
                  // Only show ghost dot if in compare mode
                  ghostProgress={isCompareMode ? progress : null} 
              />
          </div>
          
          {/* HUD OVERLAY */}
          <div className="absolute bottom-8 left-8">
               <TelemetryHUD 
                   speed={frame.primary_speed} 
                   throttle={frame.primary_obj?.ath || 0} 
                   brake={frame.primary_obj?.pbrake_f || 0} 
               />
          </div>

          <div className="absolute top-6 left-6 liquid-glass px-5 py-2 rounded-full flex items-center gap-3">
              <Timer size={18} className="text-emerald-400"/>
              <span className="text-2xl font-mono font-bold tracking-tight">
                  {(frame.time || 0).toFixed(2)}<span className="text-sm text-white/40 ml-1">s</span>
              </span>
          </div>
      </div>

      {/* ANALYSIS */}
      <div style={{ gridArea: 'analysis' }} className="liquid-glass rounded-3xl flex flex-col overflow-hidden z-40">
          <div className="p-5 border-b border-white/10 bg-white/5">
             <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">Telemetry Analysis</div>
             <div className="text-xs text-white/60">
                 {isCompareMode ? "Optimal vs. Human" : "Solo View"}
             </div>
          </div>
          <div className="flex-1 p-4">
              <TelemetryComparison 
                  isCompareMode={isCompareMode}
                  optimalFrame={frame.ref_obj} 
                  humanFrame={frame.primary_obj} 
              />
          </div>
      </div>

      {/* BOTTOM CONTROLS */}
      <div style={{ gridArea: 'telemetry' }} className="liquid-glass rounded-3xl flex flex-col p-1 z-30 relative">
          <div className="flex-1 relative w-full px-6 pt-4">
              <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartWindow}>
                      <defs>
                          <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity={0.3}/>
                            <stop offset="100%" stopColor="#10b981" stopOpacity={0}/>
                          </linearGradient>
                      </defs>
                      <YAxis hide domain={[0, 280]}/>
                      
                      {/* Comparison Trace (Ghost) - Only show in compare mode */}
                      {isCompareMode && (
                          <Area type="monotone" dataKey="ref_speed" stroke="#ffffff" strokeWidth={1} strokeOpacity={0.4} fill="transparent" isAnimationActive={false} />
                      )}
                      
                      {/* Primary Trace */}
                      <Area type="monotone" dataKey="primary_speed" stroke="#10b981" strokeWidth={2} fill="url(#grad)" isAnimationActive={false} />
                  </AreaChart>
              </ResponsiveContainer>
          </div>

          <div className="h-16 border-t border-white/10 bg-black/20 flex items-center px-8 gap-8">
               <button onClick={actions.togglePlay} className="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 transition shadow-lg">
                  {isPlaying ? <Pause size={16} fill="black"/> : <Play size={16} fill="black" className="ml-0.5"/>}
               </button>
               
               <div className="flex-1 grid grid-cols-3 gap-12">
                  <Bar label="THROTTLE" val={frame.primary_obj?.ath || 0} col="bg-emerald-500"/>
                  <Bar label="BRAKE" val={frame.primary_obj?.pbrake_f || 0} col="bg-red-500"/>
                  <Bar label="STEER" val={(Math.abs(frame.primary_obj?.Steering_Angle || 0)/5)} col="bg-blue-500"/>
               </div>
          </div>
      </div>
    </div>
  );
}

export default App;