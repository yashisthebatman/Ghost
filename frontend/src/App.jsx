import { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import { Play, Pause, Activity, Zap, Trophy, Timer, AlertCircle, MousePointerClick } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import TrackMap from './components/TrackMap';
import { TelemetryComparison, Bar } from './components/DashboardWidgets';
import { useSimulationStore } from './store/simulationStore';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
  const [state, setState] = useState({ status: "BOOT", session: null, laps: [] });
  const [activeLapId, setActiveLapId] = useState(null);
  const [displayData, setDisplayData] = useState([]);
  const [optimalData, setOptimalData] = useState([]); // Was 'ghostData'

  const { isPlaying, currentIndex, actions } = useSimulationStore();
  const requestRef = useRef();
  const lastTimeRef = useRef(0);

  // --- 1. BOOT SEQUENCE (STRICT) ---
  useEffect(() => {
    const boot = async () => {
      try {
        await axios.get(`${API_URL}/`); 
        
        const [ctx, lapsRes, optimal] = await Promise.all([
            axios.get(`${API_URL}/session/context`),
            axios.get(`${API_URL}/session/laps`),
            axios.get(`${API_URL}/laps/optimal`), // Updated Endpoint
        ]);

        setState({ 
            status: "READY", 
            session: ctx.data, 
            laps: lapsRes.data // NO FALLBACKS. If empty, it's empty.
        });
        setOptimalData(optimal.data);

        // Select the 3rd lap (PB) by default if it exists, else the first available
        if (lapsRes.data.length > 0) {
            const target = lapsRes.data.find(l => l.lap_number === 3) || lapsRes.data[0];
            loadCompareLap(target.lap_number, optimal.data, lapsRes.data);
        }

      } catch (e) {
        console.error("Boot Error:", e);
        setState(s => ({ ...s, status: "ERROR" }));
      }
    };
    boot();
  }, []);

  // --- 2. LOAD COMPARISON (Human vs Optimal) ---
  const loadCompareLap = async (humanLapId, optimalBase, history) => {
    setActiveLapId(humanLapId);
    actions.reset();
    
    try {
        let humanData;
        try {
            // Fetch specific human attempt
            const res = await axios.get(`${API_URL}/laps/human/${humanLapId}`);
            humanData = res.data;
        } catch {
            console.warn(`Human lap ${humanLapId} missing.`);
            humanData = []; // Don't fake it, just show empty
        }

        // Merge Logic
        const maxLength = Math.max(optimalBase.length, humanData.length);
        
        const merged = Array.from({ length: maxLength }).map((_, i) => {
            const opt = optimalBase[i] || {};
            const hum = humanData[i] || {};
            
            return {
                time: i * 0.01,
                // Flatted for Charts
                opt_speed: opt.speed || 0,
                hum_speed: hum.speed || 0,
                
                // Objects for Analysis Widget
                opt_obj: opt,
                hum_obj: hum
            };
        });

        setDisplayData(merged);
        actions.setDataLength(merged.length);
    } catch (e) { console.error(e); }
  };

  // --- 3. SIMULATION LOOP (Performance Optimized) ---
  const animate = (time) => {
    if (!isPlaying) return;
    if (lastTimeRef.current === 0) lastTimeRef.current = time;
    const deltaTime = time - lastTimeRef.current;
    const framesToAdvance = Math.floor(deltaTime / 10); // 10ms = 100Hz

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

  // --- DERIVED STATE ---
  const frame = displayData[currentIndex] || {};
  const progress = displayData.length > 0 ? currentIndex / displayData.length : 0;
  
  const chartWindow = useMemo(() => {
      const windowSize = 300; 
      const start = Math.max(0, currentIndex - windowSize/2);
      const end = Math.min(displayData.length, start + windowSize);
      return displayData.slice(start, end);
  }, [currentIndex, displayData]);

  if (state.status === "ERROR") return <div className="h-screen w-screen bg-black flex items-center justify-center text-red-500 font-mono">Backend Offline</div>;
  if (state.status === "BOOT") return <div className="h-screen w-screen bg-black flex items-center justify-center text-white/30 tracking-widest animate-pulse">INITIALIZING...</div>;

  return (
    <div className="dashboard-container bg-black text-white font-sans">
      <div className="noise-bg" />

      {/* HEADER */}
      <header style={{ gridArea: 'header' }} className="liquid-glass rounded-2xl flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-4">
            <div className="w-9 h-9 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(16,185,129,0.3)]">
                <Activity size={18} className="text-white"/>
            </div>
            <div>
                <h1 className="text-sm font-bold tracking-widest text-white/90">GHOST ENGINEER</h1>
                <div className="flex items-center gap-2 text-[10px] text-white/50 font-mono">
                    <span>{state.session?.vehicle_id}</span>
                    <span className="text-emerald-400">CONNECTED</span>
                </div>
            </div>
        </div>
        <div className="liquid-glass px-4 py-1.5 rounded-full flex items-center gap-2">
            <Zap size={14} className="text-yellow-400"/>
            <span className="text-xs font-bold">{state.session?.weather?.track_temp}°C</span>
        </div>
      </header>

      {/* SIDEBAR: HUMAN LAPS */}
      <div style={{ gridArea: 'sidebar' }} className="liquid-glass rounded-3xl flex flex-col overflow-hidden z-40">
         <div className="p-5 border-b border-white/10 bg-white/5">
            <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">Human Attempts</div>
            <div className="text-xs text-white/60">Select to Compare</div>
         </div>
         <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
            {state.laps.length === 0 && (
                <div className="text-center text-white/20 text-xs mt-10">No laps recorded.</div>
            )}
            {state.laps.map(lap => (
                <div key={lap.lap_number} 
                     onClick={() => loadCompareLap(lap.lap_number, optimalData, state.laps)}
                     className={`p-3 rounded-2xl cursor-pointer border transition-all duration-200 relative overflow-hidden
                        ${activeLapId === lap.lap_number ? 'liquid-glass-active border-emerald-500/30' : 'border-transparent hover:bg-white/5'}`}>
                    
                    {activeLapId === lap.lap_number && <div className="absolute inset-0 bg-emerald-500/5 pointer-events-none"/>}
                    
                    <div className="flex justify-between items-center mb-1 relative z-10">
                        <span className={`text-[10px] font-bold uppercase tracking-wider ${activeLapId === lap.lap_number ? 'text-emerald-400' : 'text-white/60'}`}>
                            Attempt {lap.lap_number}
                        </span>
                        {lap.status === "PB" && <Trophy size={12} className="text-yellow-400"/>}
                    </div>
                    <div className="flex justify-between items-end relative z-10">
                        <span className={`text-xl font-mono font-medium ${activeLapId === lap.lap_number ? 'text-white' : 'text-white/50'}`}>
                            {lap.lap_time ? lap.lap_time.toFixed(3) : '--'}s
                        </span>
                        {activeLapId !== lap.lap_number && <MousePointerClick size={14} className="text-white/20"/>}
                    </div>
                </div>
            ))}
         </div>
      </div>

      {/* CENTER: MAP */}
      <div style={{ gridArea: 'map' }} className="liquid-glass rounded-3xl relative flex items-center justify-center overflow-hidden">
          <div className="w-full h-full p-6">
              <TrackMap className="w-full h-full" progress={progress} ghostProgress={progress} />
          </div>
          <div className="absolute top-6 left-6 liquid-glass px-5 py-2 rounded-full flex items-center gap-3">
              <Timer size={18} className="text-blue-400"/>
              <span className="text-2xl font-mono font-bold tracking-tight">
                  {(frame.time || 0).toFixed(2)}<span className="text-sm text-white/40 ml-1">s</span>
              </span>
          </div>
      </div>

      {/* RIGHT: ANALYSIS */}
      <div style={{ gridArea: 'analysis' }} className="liquid-glass rounded-3xl flex flex-col overflow-hidden z-40">
          <div className="p-5 border-b border-white/10 bg-white/5">
             <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">Telemetry Analysis</div>
             <div className="text-xs text-white/60">Optimal vs. Attempt {activeLapId}</div>
          </div>
          <div className="flex-1 p-4">
              <TelemetryComparison 
                  optimalFrame={frame.opt_obj} 
                  humanFrame={frame.hum_obj} 
              />
          </div>
      </div>

      {/* BOTTOM: CHART & CONTROLS */}
      <div style={{ gridArea: 'telemetry' }} className="liquid-glass rounded-3xl flex flex-col p-1 z-30 relative">
          <div className="flex-1 relative w-full px-6 pt-4">
              <div className="absolute top-2 left-6 flex gap-6 text-[10px] font-bold uppercase tracking-widest z-10">
                  <span className="flex items-center gap-2 text-white/50"><div className="w-1.5 h-1.5 rounded-full border border-white/50"/> OPTIMAL</span>
                  <span className="flex items-center gap-2 text-emerald-400"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_#10b981]"/> YOU</span>
              </div>
              <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartWindow}>
                      <defs>
                          <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity={0.2}/>
                            <stop offset="100%" stopColor="#10b981" stopOpacity={0}/>
                          </linearGradient>
                      </defs>
                      <YAxis hide domain={[0, 280]}/>
                      {/* Optimal (Ghost) - White Line */}
                      <Area type="monotone" dataKey="opt_speed" stroke="#ffffff" strokeWidth={1} strokeOpacity={0.3} fill="transparent" isAnimationActive={false} />
                      {/* Human - Green Line */}
                      <Area type="monotone" dataKey="hum_speed" stroke="#10b981" strokeWidth={2} fill="url(#grad)" isAnimationActive={false} />
                  </AreaChart>
              </ResponsiveContainer>
          </div>

          <div className="h-16 border-t border-white/10 bg-black/20 flex items-center px-8 gap-8">
               <button onClick={actions.togglePlay} className="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 transition shadow-lg">
                  {isPlaying ? <Pause size={16} fill="black"/> : <Play size={16} fill="black" className="ml-0.5"/>}
               </button>
               
               <div className="flex-1 grid grid-cols-3 gap-12">
                  <Bar label="THROTTLE" val={frame.hum_obj?.ath || 0} col="bg-emerald-500"/>
                  <Bar label="BRAKE" val={frame.hum_obj?.pbrake_f || 0} col="bg-red-500"/>
                  <Bar label="STEER" val={(Math.abs(frame.hum_obj?.Steering_Angle || 0)/5)} col="bg-blue-500"/>
               </div>
          </div>
      </div>

    </div>
  );
}

export default App;